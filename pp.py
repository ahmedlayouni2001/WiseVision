from dataclasses import dataclass
from typing import List, Optional, Dict

import cv2
import numpy as np
np.float = float

import torch
from tqdm import tqdm
from ultralytics import YOLO

# ByteTrack
import sys
sys.path.append('./ByteTrack')
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch

# Supervision (v0.1.0)
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections
from supervision.geometry.dataclasses import Point


# =============================================================
#  CONFIG
# =============================================================
SOURCE_VIDEO_PATH          = "CRK01.mp4"
TARGET_VIDEO_PATH          = "shop_output15.mp4"

MODEL_NAME                 = "yolo11m.pt"
CONFIDENCE                 = 0.05
PERSON_CLASS_ID            = 0

SKIP                       = 2
SELLER_LOCK_WINDOW_SECONDS = 90
SELLER_RADIUS              = 350   # raja7a 300 mbaaed  px — positional fallback for seller re-id
ROLE_INHERIT_IOU           = 0.1 #raja7a 0.2
GHOST_TTL_FRAMES           = 90

REQUIRE_GPU    = True
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF_PRECISION = torch.cuda.is_available()

LINE_START = Point(1755, 838)
LINE_END   = Point(1671, 1078)


# =============================================================
#  ByteTrack args
# =============================================================
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float        = 0.05
    track_buffer: int          = 60
    match_thresh: float        = 0.2
    aspect_ratio_thresh: float = 3.0
    min_box_area: float        = 10.0
    mot20: bool                = False


def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([t.tlbr for t in tracks], dtype=float)


def match_detections_with_tracks(detections: Detections,
                                 tracks: List[STrack]) -> List[Optional[int]]:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return [None] * len(detections)
    iou = box_iou_batch(tracks2boxes(tracks), detections.xyxy)
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
    pid: int
    bbox: np.ndarray
    role: str
    last_seen_frame: int
    bytetrack_id: Optional[int] = None
    confidence: float = 0.0


# =============================================================
#  RoleTracker — IoU matching + seller positional fallback
# =============================================================
class RoleTracker:
    def __init__(self, iou_threshold: float, ghost_ttl: int,
                 seller_lock_window_seconds: float, seller_radius: float):
        self.iou_threshold = iou_threshold
        self.ghost_ttl     = ghost_ttl
        self.seller_window = seller_lock_window_seconds
        self.seller_radius = seller_radius
        self.persons: Dict[int, RolePerson] = {}
        self.next_pid      = 1
        self.seller_locked = False
        self.seller_pid: Optional[int] = None
        self._alone_since: Optional[float] = None  # timestamp when single-person streak started

    def update(self, det_xyxy: np.ndarray, det_conf: np.ndarray,
               det_bytetrack_ids: np.ndarray, processed_frame_idx: int,
               timestamp: float) -> List[RolePerson]:
        n = len(det_xyxy)
        if n == 0:
            self._evict(processed_frame_idx)
            return []

        # If only one person visible and seller is locked:
        # only force SELLER after they've been alone for at least 30s continuously.
        if n == 1 and self.seller_locked and self.seller_pid in self.persons:
            if self._alone_since is None:
                self._alone_since = timestamp
            elif timestamp - self._alone_since >= 30:
                seller = self.persons[self.seller_pid]
                seller.bbox = det_xyxy[0].copy()
                seller.last_seen_frame = processed_frame_idx
                seller.bytetrack_id = (int(det_bytetrack_ids[0])
                                       if det_bytetrack_ids[0] is not None else None)
                seller.confidence = float(det_conf[0])
                self._evict(processed_frame_idx)
                return [seller]
        else:
            self._alone_since = None  # reset streak when more than 1 person is visible

        candidate_pids = list(self.persons.keys())
        if candidate_pids:
            cand_bboxes = np.stack([self.persons[p].bbox for p in candidate_pids])
            iou_mat = box_iou_batch(det_xyxy, cand_bboxes)
        else:
            iou_mat = np.zeros((n, 0))

        assigned = [False] * len(candidate_pids)
        result: List[Optional[RolePerson]] = [None] * n

        order = (np.argsort(-iou_mat.max(axis=1)) if len(candidate_pids) > 0
                 else range(n))

        for i in order:
            if len(candidate_pids) > 0:
                for j in np.argsort(-iou_mat[i]):
                    if iou_mat[i, j] <= self.iou_threshold:
                        break
                    if assigned[j]:
                        continue
                    p = self.persons[candidate_pids[j]]
                    p.bbox = det_xyxy[i].copy()
                    p.last_seen_frame = processed_frame_idx
                    p.bytetrack_id = (int(det_bytetrack_ids[i])
                                      if det_bytetrack_ids[i] is not None else None)
                    p.confidence = float(det_conf[i])
                    result[i] = p
                    assigned[j] = True
                    break

        # Seller positional fallback: when IoU match fails but detection is near
        # the seller's last known position, re-assign it to the seller.
        if self.seller_locked and self.seller_pid in self.persons:
            seller = self.persons[self.seller_pid]
            scx = (seller.bbox[0] + seller.bbox[2]) / 2
            scy = (seller.bbox[1] + seller.bbox[3]) / 2
            for i in range(n):
                if result[i] is not None:
                    continue
                ncx = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
                ncy = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
                if np.sqrt((scx - ncx) ** 2 + (scy - ncy) ** 2) < self.seller_radius:
                    seller.bbox = det_xyxy[i].copy()
                    seller.last_seen_frame = processed_frame_idx
                    seller.bytetrack_id = (int(det_bytetrack_ids[i])
                                           if det_bytetrack_ids[i] is not None else None)
                    seller.confidence = float(det_conf[i])
                    result[i] = seller
                    break

        # Remaining unmatched → new persons
        for i in range(n):
            if result[i] is not None:
                continue
            if not self.seller_locked and timestamp <= self.seller_window:
                role = "SELLER"
                self.seller_locked = True
                self.seller_pid = self.next_pid
                print(f"  [SELLER LOCKED] pid={self.next_pid} at t={timestamp:.1f}s")
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
            result[i] = new_p
            self.next_pid += 1

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
    """Same convention as supervision's Vector.is_in()."""
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
        display_id = p.bytetrack_id if p.bytetrack_id is not None else p.pid
        label = (f"SELLER {p.confidence:.2f}" if p.role == "SELLER"
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
        iou_threshold=ROLE_INHERIT_IOU,
        ghost_ttl=GHOST_TTL_FRAMES,
        seller_lock_window_seconds=SELLER_LOCK_WINDOW_SECONDS,
        seller_radius=SELLER_RADIUS,
    )
    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    fps        = video_info.fps
    generator  = get_video_frames_generator(SOURCE_VIDEO_PATH)

    prev_sides:   Dict[int, bool]       = {}  # RoleTracker pid → side
    pid_to_btids: Dict[int, List[int]] = {}  # RoleTracker pid → [ByteTrack IDs seen]
    entry_times:  Dict[int, float]     = {}  # RoleTracker pid → entry timestamp
    exit_times:   Dict[int, float]     = {}  # RoleTracker pid → exit timestamp
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
            results = model(frame, conf=CONFIDENCE, classes=[PERSON_CLASS_ID],
                            device=DEVICE, half=HALF_PRECISION, verbose=False)
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
            )

            # 2) ByteTrack (for display IDs only)
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections),
                img_info=frame.shape, img_size=frame.shape,
            )
            detections.tracker_id = np.array(
                match_detections_with_tracks(detections, tracks), dtype=object)

            # 3) RoleTracker
            persons = role_tracker.update(
                det_xyxy=detections.xyxy,
                det_conf=detections.confidence,
                det_bytetrack_ids=detections.tracker_id,
                processed_frame_idx=processed_frame_idx,
                timestamp=timestamp,
            )

            # 4) Silently collect all ByteTrack IDs seen for each RoleTracker pid
            for p in persons:
                if p.role == "SELLER" or p.bytetrack_id is None:
                    continue
                if p.pid not in pid_to_btids:
                    pid_to_btids[p.pid] = []
                if p.bytetrack_id not in pid_to_btids[p.pid]:
                    pid_to_btids[p.pid].append(p.bytetrack_id)

            # 5) Line crossing detection — clients only
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

            # prune by tracker memory, not just current visibility
            alive = set(role_tracker.persons.keys())
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
        print(f"{'CANON':>6}  {'ALL IDs':<26}  {'ENTRY':>8}  {'EXIT':>8}  {'DURATION':>10}")
        for pid in all_pids:
            btids = pid_to_btids.get(pid, [pid])
            canon = min(btids)
            ent = entry_times.get(pid)
            ext = exit_times.get(pid)
            ent_str = f"{int(ent)//60:02d}:{int(ent)%60:02d}" if ent is not None else "--:--"
            ext_str = f"{int(ext)//60:02d}:{int(ext)%60:02d}" if ext is not None else "--:--"
            dur_str = f"{ext - ent:.0f}s" if (ent is not None and ext is not None) else "-"
            ids_str = str(sorted(btids))
            print(f"{canon:>6}  {ids_str:<26}  {ent_str:>8}  {ext_str:>8}  {dur_str:>10}")
        print("--------------------------------")
    print("======================================")


if __name__ == "__main__":
    main()
