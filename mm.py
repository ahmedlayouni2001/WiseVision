from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

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
TARGET_VIDEO_PATH          = "shop_output_v2.mp4"

MODEL_NAME                 = "yolo11m.pt"
CONFIDENCE                 = 0.1
PERSON_CLASS_ID            = 0

SKIP                       = 2
SELLER_LOCK_WINDOW_SECONDS = 90
SELLER_RADIUS              = 300      # positional fallback radius (px)
SELLER_IOU_THRESHOLD       = 0.05     # IoU threshold for seller re-id
SELLER_ALONE_SECONDS       = 30       # alone this long → force seller

REQUIRE_GPU    = True
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF_PRECISION = torch.cuda.is_available()

LINE_START = Point(1755, 838)
LINE_END   = Point(1671, 1078)


# =============================================================
#  ByteTrack args  (track_buffer=75 → ~6s at SKIP=2/25fps)
# =============================================================
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh:        float = 0.10
    track_buffer:        int   = 75
    match_thresh:        float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area:        float = 10.0
    mot20:               bool  = False


# =============================================================
#  SellerTracker
#  Same logic as the old RoleTracker but only for the seller:
#  IoU matching → positional fallback → alone-30s rule
# =============================================================
class SellerTracker:
    def __init__(self, iou_threshold: float, lock_window_seconds: float,
                 radius: float, alone_seconds: float):
        self.iou_threshold = iou_threshold
        self.lock_window   = lock_window_seconds
        self.radius        = radius
        self.alone_seconds = alone_seconds
        self.locked        = False
        self.bbox:         Optional[np.ndarray] = None
        self.confidence:   float                = 0.0
        self._alone_since: Optional[float]      = None

    def update(self, det_xyxy: np.ndarray, det_conf: np.ndarray,
               timestamp: float) -> Tuple[Optional[int], Optional[np.ndarray], float]:
        """
        Returns (seller_det_idx, seller_bbox, seller_conf).
        seller_det_idx — index in det_xyxy matched as seller (None = not found this frame).
        seller_bbox    — current or last known bbox (None before first lock).
        """
        n = len(det_xyxy)
        if n == 0:
            return None, self.bbox, self.confidence

        # Alone-since-30s: single person for ≥ alone_seconds → force seller
        if n == 1 and self.locked and self.bbox is not None:
            if self._alone_since is None:
                self._alone_since = timestamp
            elif timestamp - self._alone_since >= self.alone_seconds:
                self.bbox       = det_xyxy[0].copy()
                self.confidence = float(det_conf[0])
                return 0, self.bbox, self.confidence
        else:
            self._alone_since = None

        if self.locked and self.bbox is not None:
            # IoU matching
            iou    = box_iou_batch(det_xyxy, self.bbox[np.newaxis])[:, 0]
            best_i = int(np.argmax(iou))
            if iou[best_i] > self.iou_threshold:
                self.bbox       = det_xyxy[best_i].copy()
                self.confidence = float(det_conf[best_i])
                return best_i, self.bbox, self.confidence

            # Positional fallback
            scx = (self.bbox[0] + self.bbox[2]) / 2
            scy = (self.bbox[1] + self.bbox[3]) / 2
            best_dist = float(self.radius)
            best_i    = -1
            for i in range(n):
                ncx  = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
                ncy  = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
                dist = np.sqrt((ncx - scx) ** 2 + (ncy - scy) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_i    = i
            if best_i >= 0:
                self.bbox       = det_xyxy[best_i].copy()
                self.confidence = float(det_conf[best_i])
                return best_i, self.bbox, self.confidence

            # Seller not found this frame — return last known bbox
            return None, self.bbox, self.confidence

        # Lock on first detection within the window
        if not self.locked and n > 0 and timestamp <= self.lock_window:
            self.bbox       = det_xyxy[0].copy()
            self.confidence = float(det_conf[0])
            self.locked     = True
            print(f"  [SELLER LOCKED] at t={timestamp:.1f}s")
            return 0, self.bbox, self.confidence

        return None, self.bbox, self.confidence


# =============================================================
#  RolePerson — lightweight container for drawing / counting
# =============================================================
@dataclass
class RolePerson:
    pid:        int           # ByteTrack track_id for clients, -1 for seller
    bbox:       np.ndarray
    role:       str           # "SELLER" or "CLIENT"
    confidence: float = 0.0


# =============================================================
#  ByteTrack helpers
# =============================================================
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([t.tlbr for t in tracks], dtype=float)


def match_tracks_to_dets(detections: Detections,
                          tracks: List[STrack]) -> List[Optional[int]]:
    """Returns a list of track_ids aligned with detections (None = unmatched)."""
    if not len(detections) or not len(tracks):
        return [None] * len(detections)
    iou       = box_iou_batch(tracks2boxes(tracks), detections.xyxy)
    track2det = np.argmax(iou, axis=1)
    ids: List[Optional[int]] = [None] * len(detections)
    for ti, di in enumerate(track2det):
        if iou[ti, di] > 0:
            ids[di] = tracks[ti].track_id
    return ids


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

    byte_tracker   = BYTETracker(BYTETrackerArgs())
    seller_tracker = SellerTracker(
        iou_threshold=SELLER_IOU_THRESHOLD,
        lock_window_seconds=SELLER_LOCK_WINDOW_SECONDS,
        radius=SELLER_RADIUS,
        alone_seconds=SELLER_ALONE_SECONDS,
    )

    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    fps        = video_info.fps
    generator  = get_video_frames_generator(SOURCE_VIDEO_PATH)

    # Which ByteTrack track_id corresponds to the seller (updated when seller is detected)
    seller_byte_tid: Optional[int] = None

    prev_sides:      Dict[int, bool]  = {}   # ByteTrack track_id → side
    entry_times:     Dict[int, float] = {}
    exit_times:      Dict[int, float] = {}
    in_count         = 0
    out_count        = 0
    client_det_total = 0

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
            results  = model(frame, conf=CONFIDENCE, classes=[PERSON_CLASS_ID],
                             device=DEVICE, half=HALF_PRECISION, verbose=False)
            det_xyxy = results[0].boxes.xyxy.cpu().numpy()
            det_conf = results[0].boxes.conf.cpu().numpy()
            detections = Detections(
                xyxy=det_xyxy,
                confidence=det_conf,
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
            )

            # 2) SellerTracker — IoU + positional fallback + alone-30s (same as old RoleTracker)
            seller_det_idx, seller_bbox, seller_conf = seller_tracker.update(
                det_xyxy=det_xyxy,
                det_conf=det_conf,
                timestamp=timestamp,
            )

            # 3) ByteTrack — tracks all persons for stable client IDs
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections),
                img_info=frame.shape,
                img_size=frame.shape,
            )

            # Map detections → track_ids
            det_track_ids = match_tracks_to_dets(detections, tracks)

            # Keep seller_byte_tid up to date whenever seller is detected
            if seller_det_idx is not None and seller_det_idx < len(det_track_ids):
                tid = det_track_ids[seller_det_idx]
                if tid is not None:
                    seller_byte_tid = tid

            # Build confidence lookup
            track_conf: Dict[int, float] = {}
            for i, tid in enumerate(det_track_ids):
                if tid is not None:
                    track_conf[tid] = float(det_conf[i])

            # 4) Build persons list
            persons: List[RolePerson] = []

            # Seller — bbox from SellerTracker (stable, IoU-based)
            if seller_tracker.locked and seller_bbox is not None:
                persons.append(RolePerson(
                    pid=-1, bbox=seller_bbox, role="SELLER", confidence=seller_conf))

            # Clients — ByteTrack track_ids directly (Kalman filter = stable IDs)
            for tr in tracks:
                if tr.track_id == seller_byte_tid:
                    continue  # seller already added above
                bbox = np.array(tr.tlbr, dtype=float)
                conf = track_conf.get(tr.track_id, 0.0)
                persons.append(RolePerson(
                    pid=tr.track_id, bbox=bbox, role="CLIENT", confidence=conf))

            # 5) Line crossing — clients only
            for p in persons:
                if p.role == "SELLER":
                    continue
                client_det_total += 1
                cx = (p.bbox[0] + p.bbox[2]) / 2
                cy = (p.bbox[1] + p.bbox[3]) / 2
                curr_in = is_in_side((cx, cy), LINE_START, LINE_END)
                if p.pid in prev_sides and prev_sides[p.pid] != curr_in:
                    if not curr_in:
                        in_count += 1
                        entry_times[p.pid] = timestamp
                        print(f"  [ENTER] CLIENT #{p.pid} at "
                              f"{int(timestamp)//60:02d}:{int(timestamp)%60:02d}")
                    else:
                        out_count += 1
                        exit_times[p.pid] = timestamp
                        print(f"  [EXIT]  CLIENT #{p.pid} at "
                              f"{int(timestamp)//60:02d}:{int(timestamp)%60:02d}")
                prev_sides[p.pid] = curr_in

            # 6) Draw + write
            draw_frame(frame, persons, in_count, out_count)
            sink.write_frame(frame)

            last_persons = persons
            last_in      = in_count
            last_out     = out_count

    del model, byte_tracker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n========== TRACKING SUMMARY ==========")
    print(f"Seller locked       : {seller_tracker.locked}")
    print(f"Client detections   : {client_det_total}")
    print(f"Clients entered     : {in_count}")
    print(f"Clients exited      : {out_count}")
    print(f"Output video        : {TARGET_VIDEO_PATH}")

    all_pids = sorted(set(entry_times) | set(exit_times))
    if all_pids:
        print("\n---------- CLIENT LOG ----------")
        print(f"{'TID':>5}  {'ENTRY':>8}  {'EXIT':>8}  {'DURATION':>10}")
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
