from dataclasses import dataclass
from typing import List, Optional, Dict

import cv2
import numpy as np
np.float = float

import torch
from tqdm import tqdm
from ultralytics import YOLO

import sys
sys.path.append('./ByteTrack')
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch

from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections
from supervision.geometry.dataclasses import Point


# =============================================================
#  CONFIG
# =============================================================
SOURCE_VIDEO_PATH          = "CRK01.mp4"
TARGET_VIDEO_PATH          = "shop_output_v2.mp4"

MODEL_NAME                 = "yolo11s.pt"
CONFIDENCE                 = 0.1
PERSON_CLASS_ID            = 0

SKIP                       = 2
SELLER_LOCK_WINDOW_SECONDS = 90
SELLER_RADIUS              = 350 #300 
SELLER_ANCHOR_RADIUS       = 350     # 300 px — seller can't be claimed outside this zone from their home
SELLER_ANCHOR_EMA          = 0.03    # how fast the anchor drifts (very slow — seller stays near counter)
ROLE_INHERIT_IOU_SELLER    = 0.1  #0.2
ROLE_INHERIT_IOU_CLIENT    = 0.2
GHOST_TTL_FRAMES           = 120
CLIENT_RADIUS              = 150      # px positional fallback for clients
SELLER_ALONE_SECONDS       = 5       # person alone this long → reassign as seller

REQUIRE_GPU    = True
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF_PRECISION = torch.cuda.is_available()

LINE_START = Point(1726, 820)
LINE_END   = Point(1626, 1078)


# =============================================================
#  ByteTrack args
# =============================================================
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh:        float = 0.1
    track_buffer:        int   = 60
    match_thresh:        float = 0.3
    aspect_ratio_thresh: float = 3.0
    min_box_area:        float = 10.0
    mot20:               bool  = False


def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([t.tlbr for t in tracks], dtype=float)


def match_detections_with_tracks(detections: Detections,
                                 tracks: List[STrack]) -> List[Optional[int]]:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return [None] * len(detections)
    iou       = box_iou_batch(tracks2boxes(tracks), detections.xyxy)
    track2det = np.argmax(iou, axis=1)
    ids: List[Optional[int]] = [None] * len(detections)
    for ti, di in enumerate(track2det):
        if iou[ti, di] != 0:
            ids[di] = tracks[ti].track_id
    return ids


# =============================================================
#  RolePerson
# =============================================================
@dataclass
class RolePerson:
    pid:             int
    bbox:            np.ndarray
    role:            str
    last_seen_frame: int
    bytetrack_id:    Optional[int] = None
    confidence:      float = 0.0


# =============================================================
#  RoleTracker — IoU matching + seller positional fallback
# =============================================================
class RoleTracker:
    def __init__(self, iou_threshold_seller: float, iou_threshold_client: float,
                 ghost_ttl: int, seller_lock_window_seconds: float, seller_radius: float):
        self.iou_threshold_seller = iou_threshold_seller
        self.iou_threshold_client = iou_threshold_client
        self.ghost_ttl     = ghost_ttl
        self.seller_window = seller_lock_window_seconds
        self.seller_radius = seller_radius
        self.persons: Dict[int, RolePerson] = {}
        self.next_pid      = 1
        self.seller_locked = False
        self.seller_pid: Optional[int] = None
        self._alone_since: Optional[float] = None
        self._anchor_cx:   Optional[float] = None  # seller home position
        self._anchor_cy:   Optional[float] = None

    def _apply(self, p: RolePerson, bbox: np.ndarray, conf: float,
               bt_id, frame: int):
        p.bbox            = bbox.copy()
        p.confidence      = float(conf)
        p.last_seen_frame = frame
        p.bytetrack_id    = int(bt_id) if bt_id is not None else None

    def _move_anchor(self, cx: float, cy: float):
        if self._anchor_cx is None:
            self._anchor_cx, self._anchor_cy = cx, cy
        else:
            self._anchor_cx = (1 - SELLER_ANCHOR_EMA) * self._anchor_cx + SELLER_ANCHOR_EMA * cx
            self._anchor_cy = (1 - SELLER_ANCHOR_EMA) * self._anchor_cy + SELLER_ANCHOR_EMA * cy

    def _anchor_ok(self, dcx: float, dcy: float) -> bool:
        if self._anchor_cx is None:
            return True
        return np.sqrt((dcx - self._anchor_cx) ** 2 + (dcy - self._anchor_cy) ** 2) <= SELLER_ANCHOR_RADIUS

    def update(self, det_xyxy: np.ndarray, det_conf: np.ndarray,
               det_bytetrack_ids: np.ndarray,
               processed_frame_idx: int, timestamp: float) -> List[RolePerson]:
        n = len(det_xyxy)
        if n == 0:
            self._evict(processed_frame_idx)
            return []

        # Alone-10s rule: single person visible for ≥ SELLER_ALONE_SECONDS → seller
        if n == 1 and self.seller_locked and self.seller_pid in self.persons:
            if self._alone_since is None:
                self._alone_since = timestamp
            elif timestamp - self._alone_since >= SELLER_ALONE_SECONDS:
                seller = self.persons[self.seller_pid]
                self._apply(seller, det_xyxy[0], det_conf[0],
                            det_bytetrack_ids[0], processed_frame_idx)
                acx = (det_xyxy[0][0] + det_xyxy[0][2]) / 2
                acy = (det_xyxy[0][1] + det_xyxy[0][3]) / 2
                self._move_anchor(acx, acy)
                self._evict(processed_frame_idx)
                return [seller]
        else:
            self._alone_since = None

        result:   List[Optional[RolePerson]] = [None] * n
        used_det: set = set()

        # Pre-compute client list here so Step 1 can check it
        client_pids = [p for p in self.persons if p != self.seller_pid]

        # Step 1: seller gets absolute first pick.
        # IoU match is only accepted if the detection center is within seller_radius
        # (prevents jumping onto a nearby client when they are close).
        if self.seller_locked and self.seller_pid in self.persons:
            seller = self.persons[self.seller_pid]
            scx = (seller.bbox[0] + seller.bbox[2]) / 2
            scy = (seller.bbox[1] + seller.bbox[3]) / 2
            iou_s  = box_iou_batch(det_xyxy, seller.bbox[np.newaxis])[:, 0]
            best_i = -1
            best_iou = 0.0
            for i in range(n):
                if iou_s[i] <= self.iou_threshold_seller:
                    continue
                dcx = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
                dcy = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
                if not self._anchor_ok(dcx, dcy):
                    continue  # outside seller home zone
                dist_to_seller = np.sqrt((dcx - scx) ** 2 + (dcy - scy) ** 2)
                if dist_to_seller > self.seller_radius:
                    continue
                # Reject if any client is closer to this detection than the seller is
                closer_client = False
                for cpid in client_pids:
                    cp  = self.persons[cpid]
                    ccx = (cp.bbox[0] + cp.bbox[2]) / 2
                    ccy = (cp.bbox[1] + cp.bbox[3]) / 2
                    if np.sqrt((dcx - ccx) ** 2 + (dcy - ccy) ** 2) < dist_to_seller:
                        closer_client = True
                        break
                if closer_client:
                    continue
                if iou_s[i] > best_iou:
                    best_iou = iou_s[i]
                    best_i   = i
            if best_i >= 0:
                mcx = (det_xyxy[best_i][0] + det_xyxy[best_i][2]) / 2
                mcy = (det_xyxy[best_i][1] + det_xyxy[best_i][3]) / 2
                self._apply(seller, det_xyxy[best_i], det_conf[best_i],
                            det_bytetrack_ids[best_i], processed_frame_idx)
                self._move_anchor(mcx, mcy)
                result[best_i] = seller
                used_det.add(best_i)
            else:
                # Positional fallback — never steal a detection near a known client,
                # and must be within the seller's anchor zone
                best_dist = float(self.seller_radius)
                best_i    = -1
                for i in range(n):
                    dcx  = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
                    dcy  = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
                    if not self._anchor_ok(dcx, dcy):
                        continue  # outside seller home zone
                    dist = np.sqrt((dcx - scx) ** 2 + (dcy - scy) ** 2)
                    if dist >= best_dist:
                        continue
                    near_client = False
                    for cpid in client_pids:
                        cp  = self.persons[cpid]
                        ccx = (cp.bbox[0] + cp.bbox[2]) / 2
                        ccy = (cp.bbox[1] + cp.bbox[3]) / 2
                        if np.sqrt((dcx - ccx) ** 2 + (dcy - ccy) ** 2) < CLIENT_RADIUS:
                            near_client = True
                            break
                    if near_client:
                        continue
                    best_dist = dist
                    best_i    = i
                if best_i >= 0:
                    mcx = (det_xyxy[best_i][0] + det_xyxy[best_i][2]) / 2
                    mcy = (det_xyxy[best_i][1] + det_xyxy[best_i][3]) / 2
                    self._apply(seller, det_xyxy[best_i], det_conf[best_i],
                                det_bytetrack_ids[best_i], processed_frame_idx)
                    self._move_anchor(mcx, mcy)
                    result[best_i] = seller
                    used_det.add(best_i)

        # Step 2: clients — IoU + distance check, then positional fallback.
        # Distance check prevents a client from jumping onto the seller's body.
        free_dets   = [i for i in range(n) if i not in used_det]
        if client_pids and free_dets:
            free_xyxy   = det_xyxy[free_dets]
            cand_bboxes = np.stack([self.persons[p].bbox for p in client_pids])
            iou_mat     = box_iou_batch(free_xyxy, cand_bboxes)
            assigned    = [False] * len(client_pids)
            order       = np.argsort(-iou_mat.max(axis=1))
            for fi in order:
                i = free_dets[fi]
                dcx = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
                dcy = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
                for j in np.argsort(-iou_mat[fi]):
                    if iou_mat[fi, j] <= self.iou_threshold_client:
                        break
                    if assigned[j]:
                        continue
                    p   = self.persons[client_pids[j]]
                    pcx = (p.bbox[0] + p.bbox[2]) / 2
                    pcy = (p.bbox[1] + p.bbox[3]) / 2
                    if np.sqrt((dcx - pcx) ** 2 + (dcy - pcy) ** 2) > CLIENT_RADIUS:
                        continue  # IoU match too far — skip, try next candidate
                    self._apply(p, det_xyxy[i], det_conf[i],
                                det_bytetrack_ids[i], processed_frame_idx)
                    result[i]   = p
                    assigned[j] = True
                    used_det.add(i)
                    break

            # Positional fallback for clients whose IoU match failed
            free_dets = [i for i in range(n) if i not in used_det]
            for i in free_dets:
                dcx = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
                dcy = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
                best_dist = float(CLIENT_RADIUS)
                best_j    = -1
                for j, pid in enumerate(client_pids):
                    if assigned[j]:
                        continue
                    p   = self.persons[pid]
                    pcx = (p.bbox[0] + p.bbox[2]) / 2
                    pcy = (p.bbox[1] + p.bbox[3]) / 2
                    dist = np.sqrt((dcx - pcx) ** 2 + (dcy - pcy) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_j    = j
                if best_j >= 0:
                    p = self.persons[client_pids[best_j]]
                    self._apply(p, det_xyxy[i], det_conf[i],
                                det_bytetrack_ids[i], processed_frame_idx)
                    result[i]        = p
                    assigned[best_j] = True
                    used_det.add(i)

        # Step 3: remaining unmatched -> new persons
        for i in range(n):
            if result[i] is not None:
                continue
            if not self.seller_locked and timestamp <= self.seller_window:
                role               = "SELLER"
                self.seller_locked = True
                self.seller_pid    = self.next_pid
                icx = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
                icy = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
                self._move_anchor(icx, icy)
                print(f"  [SELLER LOCKED] pid={self.next_pid} at t={timestamp:.1f}s  anchor=({icx:.0f},{icy:.0f})")
            else:
                role = "CLIENT"
            new_p = RolePerson(
                pid=self.next_pid,
                bbox=det_xyxy[i].copy(),
                role=role,
                last_seen_frame=processed_frame_idx,
                bytetrack_id=(int(det_bytetrack_ids[i])
                              if det_bytetrack_ids[i] is not None else None),
                confidence=float(det_conf[i]),
            )
            self.persons[self.next_pid] = new_p
            result[i]                   = new_p
            self.next_pid              += 1

        self._evict(processed_frame_idx)
        return result  # type: ignore

    def _evict(self, current_frame: int):
        to_drop = [pid for pid, p in self.persons.items()
                   if pid != self.seller_pid
                   and current_frame - p.last_seen_frame > self.ghost_ttl]
        for pid in to_drop:
            del self.persons[pid]


# =============================================================
#  Manual line crossing
# =============================================================
def is_in_side(point, ls: Point, le: Point) -> bool:
    cross = (le.x - ls.x) * (point[1] - ls.y) - (le.y - ls.y) * (point[0] - ls.x)
    return cross < 0


# =============================================================
#  Annotator
# =============================================================
def draw_frame(frame: np.ndarray, persons: List[RolePerson],
               in_count: int, out_count: int) -> np.ndarray:
    SELLER_COLOR = (0, 0, 255)
    CLIENT_COLOR = (0, 200, 0)
    for p in persons:
        x1, y1, x2, y2 = map(int, p.bbox)
        color      = SELLER_COLOR if p.role == "SELLER" else CLIENT_COLOR
        display_id = p.bytetrack_id if p.bytetrack_id is not None else p.pid
        label      = (f"SELLER {p.confidence:.2f}" if p.role == "SELLER"
                      else f"CLIENT #{display_id} {p.confidence:.2f}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.line(frame, (LINE_START.x, LINE_START.y), (LINE_END.x, LINE_END.y),
             (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"IN: {in_count}  OUT: {out_count}",
                (LINE_START.x - 160, LINE_START.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


# =============================================================
#  Main
# =============================================================
def main():
    print("=" * 50)
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
    else:
        if REQUIRE_GPU:
            raise RuntimeError("[GPU] NO CUDA.")
        print("[GPU] running on CPU.")
    print("=" * 50)

    model = YOLO(MODEL_NAME)
    model.to(DEVICE)
    model.fuse()

    byte_tracker = BYTETracker(BYTETrackerArgs())
    role_tracker = RoleTracker(
        iou_threshold_seller=ROLE_INHERIT_IOU_SELLER,
        iou_threshold_client=ROLE_INHERIT_IOU_CLIENT,
        ghost_ttl=GHOST_TTL_FRAMES,
        seller_lock_window_seconds=SELLER_LOCK_WINDOW_SECONDS,
        seller_radius=SELLER_RADIUS,
    )

    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    fps        = video_info.fps
    generator  = get_video_frames_generator(SOURCE_VIDEO_PATH)

    prev_sides:  Dict[int, bool]  = {}
    entry_times: Dict[int, float] = {}
    exit_times:  Dict[int, float] = {}
    in_count  = 0
    out_count = 0

    last_persons: List[RolePerson] = []
    last_in  = 0
    last_out = 0
    processed_frame_idx = -1

    with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame_idx, frame in enumerate(tqdm(generator, total=video_info.total_frames)):

            # FROZEN frame
            if frame_idx % SKIP != 0:
                draw_frame(frame, last_persons, last_in, last_out)
                sink.write_frame(frame)
                continue

            processed_frame_idx += 1
            timestamp = frame_idx / fps

            # 1) YOLO
            results    = model(frame, conf=CONFIDENCE, classes=[PERSON_CLASS_ID],
                               device=DEVICE, half=HALF_PRECISION, verbose=False)
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
            )

            # 2) ByteTrack — stable display IDs
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections),
                img_info=frame.shape,
                img_size=frame.shape,
            )
            detections.tracker_id = np.array(
                match_detections_with_tracks(detections, tracks), dtype=object)

            # 3) RoleTracker — role assignment (seller vs client)
            persons = role_tracker.update(
                det_xyxy=detections.xyxy,
                det_conf=detections.confidence,
                det_bytetrack_ids=detections.tracker_id,
                processed_frame_idx=processed_frame_idx,
                timestamp=timestamp,
            )

            # 4) Line crossing — clients only, keyed by stable RoleTracker pid
            for p in persons:
                if p.role == "SELLER":
                    continue
                cx = (p.bbox[0] + p.bbox[2]) / 2
                cy = (p.bbox[1] + p.bbox[3]) / 2
                curr_in = is_in_side((cx, cy), LINE_START, LINE_END)
                if p.pid in prev_sides and prev_sides[p.pid] != curr_in:
                    display = p.bytetrack_id if p.bytetrack_id is not None else p.pid
                    if not curr_in:
                        in_count += 1
                        entry_times[p.pid] = timestamp
                        print(f"  [ENTER] CLIENT #{display} at "
                              f"{int(timestamp)//60:02d}:{int(timestamp)%60:02d}")
                    else:
                        out_count += 1
                        exit_times[p.pid] = timestamp
                        print(f"  [EXIT]  CLIENT #{display} at "
                              f"{int(timestamp)//60:02d}:{int(timestamp)%60:02d}")
                prev_sides[p.pid] = curr_in

            # Prune prev_sides to alive tracks only
            alive      = set(role_tracker.persons.keys())
            prev_sides = {k: v for k, v in prev_sides.items() if k in alive}

            # 5) Draw + write
            draw_frame(frame, persons, in_count, out_count)
            sink.write_frame(frame)

            last_persons = persons
            last_in      = in_count
            last_out     = out_count

    del model, byte_tracker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n========== TRACKING SUMMARY ==========")
    print(f"Seller locked       : {role_tracker.seller_locked}")
    print(f"Seller pid          : {role_tracker.seller_pid}")
    print(f"Total unique persons: {role_tracker.next_pid - 1}")
    print(f"Clients entered     : {in_count}")
    print(f"Clients exited      : {out_count}")
    print(f"Output video        : {TARGET_VIDEO_PATH}")

    all_pids = sorted(set(entry_times) | set(exit_times))
    if all_pids:
        print("\n---------- CLIENT LOG ----------")
        print(f"{'PID':>5}  {'ENTRY':>8}  {'EXIT':>8}  {'DURATION':>10}")
        for pid in all_pids:
            ent = entry_times.get(pid)
            ext = exit_times.get(pid)
            ent_str = f"{int(ent)//60:02d}:{int(ent)%60:02d}" if ent is not None else "  --:--"
            ext_str = f"{int(ext)//60:02d}:{int(ext)%60:02d}" if ext is not None else "  --:--"
            dur_str = f"{ext - ent:.0f}s" if (ent is not None and ext is not None) else "      -"
            print(f"{pid:>5}  {ent_str:>8}  {ext_str:>8}  {dur_str:>10}")
        print("--------------------------------")
    print("======================================")


if __name__ == "__main__":
    main()
