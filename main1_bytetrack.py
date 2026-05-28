from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

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

import supervision as sv


# =============================================================
#  CONFIG
# =============================================================
SOURCE_VIDEO_PATH          = "CRK01.mp4"
TARGET_VIDEO_PATH          = "shop_output_bt.mp4"

MODEL_NAME                 = "yolo11s.pt"
CONF_SELLER                = 0.1
CONF_CLIENT                = 0.2
PERSON_CLASS_ID            = 0

SKIP                       = 2
SELLER_LOCK_WINDOW_SECONDS = 90
SELLER_RADIUS              = 300
ROLE_INHERIT_IOU_SELLER    = 0.1
SELLER_ALONE_SECONDS       = 5

GHOST_TTL_FRAMES           = 120

REQUIRE_GPU    = True
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF_PRECISION = torch.cuda.is_available()

LINE_START = sv.Point(1726, 820)
LINE_END   = sv.Point(1626, 1078)


# =============================================================
#  ByteTrack args
# =============================================================
@dataclass
class BYTETrackerArgs:
    track_thresh:        float = 0.2
    track_buffer:        int   = 90
    match_thresh:        float = 0.6
    aspect_ratio_thresh: float = 3.0
    min_box_area:        float = 1.0
    mot20:               bool  = False


# =============================================================
#  Helpers
# =============================================================
def detections2boxes(detections: sv.Detections) -> np.ndarray:
    return np.hstack([detections.xyxy, detections.confidence[:, np.newaxis]])


def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([t.tlbr for t in tracks], dtype=float)


def match_detections_with_tracks(
    detections: sv.Detections, tracks: List[STrack]
) -> sv.Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return detections
    track_boxes = tracks2boxes(tracks)
    iou = box_iou_batch(track_boxes, detections.xyxy)
    track2det = np.argmax(iou, axis=1)
    tracker_ids = np.full(len(detections), -1, dtype=int)
    for ti, di in enumerate(track2det):
        if iou[ti, di] > 0:
            tracker_ids[di] = tracks[ti].track_id
    detections.tracker_id = tracker_ids
    return detections


# =============================================================
#  RolePerson
# =============================================================
@dataclass
class RolePerson:
    pid:             int
    bbox:            np.ndarray
    role:            str
    last_seen_frame: int
    confidence:      float = 0.0

    @property
    def first_pid(self):
        return self.pid


# =============================================================
#  RoleTracker
# =============================================================
class RoleTracker:
    def __init__(self, iou_threshold_seller: float,
                 seller_lock_window_seconds: float, seller_radius: float,
                 client_conf_threshold: float = 0.2):
        self.iou_threshold_seller  = iou_threshold_seller
        self.seller_window         = seller_lock_window_seconds
        self.seller_radius         = seller_radius
        self.client_conf_threshold = client_conf_threshold

        self.persons: Dict[int, RolePerson] = {}
        self.bt_to_first_pid: Dict[int, int] = {}
        self.evicted_cache: Dict[int, Tuple[RolePerson, int]] = {}
        self.pid_chains: Dict[int, List[int]] = {}

        self.seller_locked = False
        self.seller_pid: Optional[int] = None
        self._alone_since: Optional[float] = None
        self._rescue_count: int = 0
        self._rescue_fp:    Optional[int] = None

    def _apply(self, p: RolePerson, bbox: np.ndarray, conf: float, frame: int):
        p.bbox            = bbox.copy()
        p.confidence      = float(conf)
        p.last_seen_frame = frame

    def _get_or_link(self, bt_id: int) -> Optional[int]:
        return self.bt_to_first_pid.get(bt_id)

    def _link(self, bt_id: int, stable_pid: int):
        self.bt_to_first_pid[bt_id] = stable_pid
        if stable_pid not in self.pid_chains:
            self.pid_chains[stable_pid] = []
        if bt_id not in self.pid_chains[stable_pid]:
            self.pid_chains[stable_pid].append(bt_id)

    def update(self, det_xyxy: np.ndarray, det_conf: np.ndarray,
               det_tids: np.ndarray,
               frame_idx: int, timestamp: float) -> List[RolePerson]:
        n = len(det_xyxy)
        if n == 0:
            return []

        active_bt_ids = set(int(t) for t in det_tids)
        client_pids   = [p for p in self.persons if p != self.seller_pid]
        out: List[RolePerson] = []
        used: set = set()

        # Alone rule
        if n == 1 and self.seller_locked and self.seller_pid in self.persons:
            if self._alone_since is None:
                self._alone_since = timestamp
            elif timestamp - self._alone_since >= SELLER_ALONE_SECONDS:
                seller = self.persons[self.seller_pid]
                self._apply(seller, det_xyxy[0], det_conf[0], frame_idx)
                for sp in list(self.persons.keys()):
                    if sp != self.seller_pid:
                        del self.persons[sp]
                return [seller]
        else:
            self._alone_since = None

        # Step 1: seller
        if self.seller_locked and self.seller_pid in self.persons:
            seller = self.persons[self.seller_pid]
            scx = (seller.bbox[0] + seller.bbox[2]) / 2
            scy = (seller.bbox[1] + seller.bbox[3]) / 2

            best_i = -1
            for i in range(n):
                bt = int(det_tids[i])
                if self._get_or_link(bt) == self.seller_pid:
                    best_i = i
                    break

            if best_i < 0:
                iou_s    = box_iou_batch(det_xyxy, seller.bbox[np.newaxis])[:, 0]
                best_iou = 0.0
                for i in range(n):
                    if iou_s[i] <= self.iou_threshold_seller:
                        continue
                    dcx = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
                    dcy = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
                    dist = np.sqrt((dcx - scx) ** 2 + (dcy - scy) ** 2)
                    if dist > self.seller_radius:
                        continue
                    closer = any(
                        np.sqrt(((det_xyxy[i][0]+det_xyxy[i][2])/2
                                 - (self.persons[p].bbox[0]+self.persons[p].bbox[2])/2)**2
                                + ((det_xyxy[i][1]+det_xyxy[i][3])/2
                                   - (self.persons[p].bbox[1]+self.persons[p].bbox[3])/2)**2)
                        < dist
                        for p in client_pids if p in self.persons
                    )
                    if closer:
                        continue
                    if iou_s[i] > best_iou:
                        best_iou = iou_s[i]
                        best_i   = i

            if best_i < 0:
                best_dist = float(self.seller_radius)
                for i in range(n):
                    dcx  = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
                    dcy  = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
                    dist = np.sqrt((dcx - scx) ** 2 + (dcy - scy) ** 2)
                    if dist >= best_dist:
                        continue
                    near_client = any(
                        np.sqrt(((det_xyxy[i][0]+det_xyxy[i][2])/2
                                 - (self.persons[p].bbox[0]+self.persons[p].bbox[2])/2)**2
                                + ((det_xyxy[i][1]+det_xyxy[i][3])/2
                                   - (self.persons[p].bbox[1]+self.persons[p].bbox[3])/2)**2)
                        < 150
                        for p in client_pids if p in self.persons
                    )
                    if near_client:
                        continue
                    best_dist = dist
                    best_i    = i

            if best_i >= 0:
                bt = int(det_tids[best_i])
                self._link(bt, self.seller_pid)
                self._apply(seller, det_xyxy[best_i], det_conf[best_i], frame_idx)
                used.add(best_i)
                out.append(seller)

        # Step 2 & 3: clients
        for i in range(n):
            if i in used:
                continue
            bt   = int(det_tids[i])
            conf = float(det_conf[i])

            stable_pid = self._get_or_link(bt)

            if stable_pid is not None and stable_pid in self.persons:
                p = self.persons[stable_pid]
                self._apply(p, det_xyxy[i], conf, frame_idx)
                out.append(p)
            elif stable_pid is None:
                matched_evicted = None
                best_iou = 0.0
                for ep, (ep_person, ep_frame) in list(self.evicted_cache.items()):
                    if frame_idx - ep_frame > GHOST_TTL_FRAMES:
                        del self.evicted_cache[ep]
                        continue
                    iou = box_iou_batch(det_xyxy[i:i+1], ep_person.bbox[np.newaxis])[0, 0]
                    if iou > 0.65 and iou > best_iou:
                        best_iou        = iou
                        matched_evicted = ep

                if matched_evicted is not None:
                    p, _ = self.evicted_cache.pop(matched_evicted)
                    self._link(bt, matched_evicted)
                    self.persons[matched_evicted] = p
                    self._apply(p, det_xyxy[i], conf, frame_idx)
                    out.append(p)
                else:
                    if not self.seller_locked and timestamp <= self.seller_window:
                        role               = "SELLER"
                        self.seller_locked = True
                        stable_pid         = bt
                        self.seller_pid    = stable_pid
                        print(f"  [SELLER LOCKED] bt_id={bt} stable_pid={stable_pid} t={timestamp:.1f}s")
                    else:
                        if conf < self.client_conf_threshold:
                            continue
                        role       = "CLIENT"
                        stable_pid = bt

                    self._link(bt, stable_pid)
                    p = RolePerson(
                        pid=stable_pid, bbox=det_xyxy[i].copy(),
                        role=role, last_seen_frame=frame_idx, confidence=conf,
                    )
                    self.persons[stable_pid] = p
                    out.append(p)

        # Seller rescue (3 consecutive frames of IoU > 0.65)
        if self.seller_locked and self.seller_pid in self.persons:
            seller = self.persons[self.seller_pid]
            if not any(p.pid == self.seller_pid for p in out):
                rescue_found = False
                for idx, p in enumerate(out):
                    if p.role != 'CLIENT':
                        continue
                    iou = box_iou_batch(p.bbox[np.newaxis], seller.bbox[np.newaxis])[0, 0]
                    if iou > 0.65:
                        rescue_found = True
                        if self._rescue_fp == p.pid:
                            self._rescue_count += 1
                        else:
                            self._rescue_fp    = p.pid
                            self._rescue_count = 1
                        if self._rescue_count >= 3:
                            self._apply(seller, p.bbox, p.confidence, frame_idx)
                            if p.pid in self.persons:
                                del self.persons[p.pid]
                            out[idx] = seller
                            self._rescue_count = 0
                            self._rescue_fp    = None
                        break
                if not rescue_found:
                    self._rescue_count = 0
                    self._rescue_fp    = None
            else:
                self._rescue_count = 0
                self._rescue_fp    = None

        # Evict stable pids whose ByteTrack ids are all gone
        for sp in list(self.persons.keys()):
            if sp == self.seller_pid:
                continue
            bt_ids = self.pid_chains.get(sp, [sp])
            if not any(b in active_bt_ids for b in bt_ids):
                p = self.persons.pop(sp)
                self.evicted_cache[sp] = (p, frame_idx)

        return out


# =============================================================
#  Manual line crossing
# =============================================================
def is_in_side(point, ls: sv.Point, le: sv.Point) -> bool:
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
        color = SELLER_COLOR if p.role == "SELLER" else CLIENT_COLOR
        label = (f"SELLER {p.confidence:.2f}" if p.role == "SELLER"
                 else f"CLIENT #{p.pid} {p.confidence:.2f}")
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
        seller_lock_window_seconds=SELLER_LOCK_WINDOW_SECONDS,
        seller_radius=SELLER_RADIUS,
        client_conf_threshold=CONF_CLIENT,
    )

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    fps        = video_info.fps
    generator  = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    prev_sides:  Dict[int, bool]  = {}
    entry_times: Dict[int, float] = {}
    exit_times:  Dict[int, float] = {}
    in_count  = 0
    out_count = 0

    last_persons: List[RolePerson] = []
    last_in  = 0
    last_out = 0
    processed_frame_idx = -1

    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame_idx, frame in enumerate(tqdm(generator, total=video_info.total_frames)):

            if frame_idx % SKIP != 0:
                draw_frame(frame, last_persons, last_in, last_out)
                sink.write_frame(frame)
                continue

            processed_frame_idx += 1
            timestamp = frame_idx / fps

            results = model(
                frame,
                conf=CONF_SELLER, classes=[PERSON_CLASS_ID],
                iou=0.5,
                device=DEVICE, half=HALF_PRECISION, verbose=False,
            )

            detections = sv.Detections.from_ultralytics(results[0])
            detections = detections[detections.confidence >= CONF_SELLER]

            tracks = byte_tracker.update(
                output_results=detections2boxes(detections),
                img_info=frame.shape[:2],
                img_size=frame.shape[:2],
            )
            detections = match_detections_with_tracks(detections, tracks)
            detections = detections[detections.tracker_id > 0]

            det_xyxy = detections.xyxy
            det_conf = detections.confidence
            det_tids = detections.tracker_id

            persons = role_tracker.update(
                det_xyxy=det_xyxy,
                det_conf=det_conf,
                det_tids=det_tids,
                frame_idx=processed_frame_idx,
                timestamp=timestamp,
            )

            for p in persons:
                if p.role == "SELLER":
                    continue
                cx = (p.bbox[0] + p.bbox[2]) / 2
                cy = (p.bbox[1] + p.bbox[3]) / 2
                curr_in = is_in_side((cx, cy), LINE_START, LINE_END)
                if p.first_pid in prev_sides and prev_sides[p.first_pid] != curr_in:
                    if not curr_in:
                        in_count += 1
                        entry_times[p.first_pid] = timestamp
                        print(f"  [ENTER] CLIENT #{p.first_pid} at "
                              f"{int(timestamp)//60:02d}:{int(timestamp)%60:02d}")
                    else:
                        out_count += 1
                        exit_times[p.first_pid] = timestamp
                        print(f"  [EXIT]  CLIENT #{p.first_pid} at "
                              f"{int(timestamp)//60:02d}:{int(timestamp)%60:02d}")
                prev_sides[p.first_pid] = curr_in

            alive = set(role_tracker.persons.keys())
            prev_sides = {k: v for k, v in prev_sides.items() if k in alive}

            draw_frame(frame, persons, in_count, out_count)
            sink.write_frame(frame)

            last_persons = persons
            last_in      = in_count
            last_out     = out_count

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n========== TRACKING SUMMARY ==========")
    print(f"Seller locked       : {role_tracker.seller_locked}")
    print(f"Seller stable_pid   : {role_tracker.seller_pid}")
    print(f"Clients entered     : {in_count}")
    print(f"Clients exited      : {out_count}")
    print(f"Output video        : {TARGET_VIDEO_PATH}")

    all_pids = sorted(set(entry_times) | set(exit_times))
    if all_pids:
        print("\n---------- CLIENT LOG ----------")
        print(f"{'CLIENT':<8}  {'ENTRY':>7}  {'EXIT':>7}  {'DURATION':>9}")
        for pid in all_pids:
            ent = entry_times.get(pid)
            ext = exit_times.get(pid)
            duration = ext - ent if (ent is not None and ext is not None) else None
            if duration is not None and duration <= 3:
                continue
            ent_str = f"{int(ent)//60:02d}:{int(ent)%60:02d}" if ent is not None else "--:--"
            ext_str = f"{int(ext)//60:02d}:{int(ext)%60:02d}" if ext is not None else "--:--"
            dur_str = f"{duration:.0f}s" if duration is not None else "-"
            print(f"#{pid:<7}  {ent_str:>7}  {ext_str:>7}  {dur_str:>9}")
        print("--------------------------------")
    print("======================================")


if __name__ == "__main__":
    main()
