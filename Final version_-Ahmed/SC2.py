#!/usr/bin/env python3
"""
SC.py - Shop SELLER vs CLIENT + entry counting + PEC (interaction) counting.
STANDALONE: no dependency on bot-sort.py -- everything is inlined here.

What it does, per processed frame:
  1) SEED (first SEED_SECONDS): everyone detected is a SELLER (bootstraps staff).
  2) ROLE: a person is SELLER if their lower body matches the learned UNIFORM
     beige band AND they show a white top. This VOTES per BoT-SORT track: after
     SELLER_VOTES matching frames the track is locked SELLER and stays SELLER on
     every later frame (anti-flicker). A staff-desk ZONE also forces SELLER (for a
     seated staffer whose pants are hidden). Everyone else is CLIENT (no id shown).
  3) ENTRY COUNT: a CLIENT is counted +1 when their box center crosses the
     doorway line INTO the shop (crossing the other way = exit). Sellers are never
     counted. Keyed on the BoT-SORT track id (only to detect the crossing edge).
  4) INTERACTION: a CLIENT within INTERACTION_DIST px of any seller for
     >= INTERACTION_MIN_SECS of continuous contact = ONE interaction, counted
     once per client track id (the same client is never double-counted).
  5) MIRROR zones drop reflections; IGNORE zones drop static false detections.

Tracking uses BoT-SORT (boxmot) with OSNet ReID -- ONLY to give track ids; the
SELLER decision is uniform COLOR, not ReID.

PRODUCTION: the uniform color band is BAKED IN (UNIFORM_BAND_*), learned once
from reference crops, so NO image folder is needed at runtime.
"""

import os
import glob
import time
from pathlib import Path

import cv2
import numpy as np
np.float = float                      # compat shim for older boxmot/numpy

import torch
from ultralytics import YOLO
import supervision as sv
from boxmot.trackers.tracker_zoo import create_tracker

# =============================================================
#  CONFIG  (everything is here)
# =============================================================
# --- videos (set these to your files) ---------------------------------------
SOURCE_VIDEO_PATH = "testvid_playable.mp4"   # <-- INPUT video
TARGET_VIDEO_PATH = "SC_output23.mp4"          # <-- OUTPUT annotated video

# --- models ------------------------------------------------------------------
MODEL_NAME = "yolo11m-pose.pt"                # YOLO-POSE: person boxes + 17 keypoints (auto-downloads)
REID_MODEL = "osnet_ain_x1_0_msmt17.pt"       # OSNet ReID for BoT-SORT (auto-downloads)
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF_PRECISION = torch.cuda.is_available()

# --- detection / general -----------------------------------------------------
DETECT_CONF     = 0.07    # single YOLO conf -> keep ALL detected people ;
                        # no per-role confidence filtering. detection -> classify -> track.
PERSON_CLASS_ID = 0
SKIP            = 2    # process EVERY frame -> steadiest tracking (2-3x slower)

# --- BoT-SORT / ReID tuning --------------------------------------------------
TRACK_BUFFER         = 180
APPEARANCE_THRESH    = 0.3
PROXIMITY_THRESH     = 0.8
MATCH_THRESH         = 0.8
CMC_METHOD           = "sof"   # sparse optical flow; robust on a fixed camera (ecc spammed "did not converge")
TRACK_HIGH_THRESH    = 0.4
TRACK_LOW_THRESH     = 0.1
NEW_TRACK_THRESH     = 0.2
APPEARANCE_EMA_ALPHA = 0.95

# --- UNIFORM color (the SELLER cue) ------------------------------------------
# PRODUCTION: band baked in below -> no images needed.
#   USE_REF_COLORS = False -> use UNIFORM_BAND_* (production, image-free).
#   USE_REF_COLORS = True  -> re-learn from SELLER_REF_DIR (dev), then copy the
#                             printed band into UNIFORM_BAND_* and set back to False.
USE_REF_COLORS  = False
SELLER_REF_DIR  = "reference_image"           # only used if USE_REF_COLORS=True
UNIFORM_BAND_LO = np.array([18,  0,  51])      # baked-in HSV band (lo), learned once
UNIFORM_BAND_HI = np.array([49, 54, 240])      # baked-in HSV band (hi)
COLOR_PCT_LO = 5      # [^] percentile band of the uniform's beige pixels (used only when re-learning)
COLOR_PCT_HI = 95     # [^]
COLOR_MARGIN = np.array([4, 15, 30])           # H,S,V padding around the learned band (re-learn only)
COLOR_TAKE   = 0.15  # (legacy, unused) fraction of lower-body pixels inside the band
WHITE_MIN    = 0.1    # [^] WHOLE-BOX white fraction for the SELLER-ZONE white check (body_white_ratio)
SELLER_VOTES = 2  # [^] beige-keypoint frames before a track locks SELLER (anti-flicker)

# --- SELLER cue = BEIGE PANTS at the LEG KEYPOINTS (yolo-pose) -----------------
# White-top logic is GONE; beige pants are the only appearance cue. For each of the
# hip/knee keypoints (COCO), sample a small patch of pixels around it and test it
# against the beige band UNIFORM_BAND_LO..HI. If >= BEIGE_KP_MIN of them are beige,
# the person is uniform. Ankles (15,16) are skipped -- they sit on shoes/floor.
LOWER_KP_IDX  = [11, 12, 13, 14]   # left/right hip, left/right knee (COCO 17-pt)
KP_CONF       = 0.07            # a keypoint must be at least this confident to be tested
KP_PATCH_R    = 3                 # half-window -> a (2r+1)^2 = 7x7 px patch around the keypoint
KP_PATCH_FRAC = 0.5                # >= this fraction of the patch inside the band -> that kp is "beige"
BEIGE_KP_MIN  = 1                  # >= this many beige leg keypoints -> pants ok (2 of 4)

# WHITE SHIRT check on the SHOULDER keypoints (second cue). SELLER requires beige
# pants AND a white shirt: >= WHITE_KP_MIN shoulders on white. White = low saturation
# (<= WHITE_S_MAX) and high value (>= WHITE_V_MIN). Hips are NOT used -- they sit at
# the waist/pants transition and read unreliably.
UPPER_KP_IDX     = [5, 6]          # shoulders only (COCO 17-pt)
WHITE_KP_MIN     = 1               # >= this many shoulders on white -> shirt ok (1 of 2)
WHITE_PATCH_FRAC = 0.5             # >= this fraction of the patch white -> that kp is "white"

# beige pants HSV gate (broad pre-gate when re-learning) + white-top gate
BEIGE_H_MIN, BEIGE_H_MAX = 22, 55
BEIGE_S_MAX = 95
BEIGE_V_MIN = 75
WHITE_S_MAX = 80
WHITE_V_MIN = 110 

# --- SEED window -------------------------------------------------------------
SEED_SECONDS   = 3.0     # everyone detected before this is SELLER (+ teaches color if re-learning)
SEED_BEIGE_MIN = 0.05    # only sample pants color from seed people who actually show beige

# --- ENTRY = ORDERED TWO-ZONE pass (direction comes from the ORDER, no line) --
# Two zones across the doorway. A CLIENT is counted +1 only when they pass
#   ENTER_ZONE_1  THEN  ENTER_ZONE_2   (this direction = ENTERING the shop).
# Passing 2 THEN 1 (leaving) or touching only ONE zone => NOT counted.
# "In a zone" = >= ENTER_ZONE_FRAC of the person BOX overlaps it (grid-sampled).
# Direction is purely the visiting ORDER -- no LINE / signed-distance math.
# DRAW BOTH with "tools for help/pick_zones.py":
#   ZONE 1 = the side crossed FIRST when entering (door / outside).
#   ZONE 2 = the side reached SECOND when entering (shop interior).
ENTER_ZONE_1 = [
    [(1909, 708), (1909, 1068), (1715, 1065), (1822, 685), (1908, 708)]    # door side (crossed first)
]
ENTER_ZONE_2 = [
    [(1682, 1055), (1790, 680), (1732, 663), (1602, 1066), (1686, 1053)]   # shop side (reached second)
]
ENTER_ZONE_FRAC = 0.10  # [^] >= this fraction of the BOX must overlap a zone to be "in" it
_ENTER_POLYS_1 = [np.array(z, np.int32) for z in ENTER_ZONE_1 if len(z) >= 3]
_ENTER_POLYS_2 = [np.array(z, np.int32) for z in ENTER_ZONE_2 if len(z) >= 3]
# id-FREE matching (so a client with NO BoT-SORT id is still tracked across zones):
ZONE_MATCH_DIST =200  # [^] px to link a client to its track between frames (no id)
ZONE_TTL        = 20   # processed-frames a track survives unseen before expiring

# --- REVERT a mis-counted entry ----------------------------------------------
# A real staffer can be briefly misread as a CLIENT at the doorway and counted
# +1. If that SAME track id later locks SELLER within ENTRY_REVERT_SECS, undo the
# +1 (they were staff, not a customer). Needs a track id at count time.
ENTRY_REVERT_SECS = 4.0  # [^] window after an entry in which a SELLER flip cancels it
REVERT_MIN_SELLER_SECS = 0.5  # [^] the SELLER role must PERSIST this long to revert; a
                            #     shorter flip is a momentary misread of a real client
                            #     and does NOT cancel the +1

# --- INTERACTION (client <-> seller) -----------------------------------------
# A 200 px circle is drawn at EACH person's box CENTER. A CLIENT is "in contact"
# with a seller when their two circles OVERLAP by >= INTERACTION_OVERLAP_MIN of a
# circle's area (equal radii -> centers within ~162 px at 0.5). Near ANY seller
# counts (OMNIDIRECTIONAL). Contact time need NOT be unbroken: a no-contact gap up
# to INTERACTION_GRACE_SECS does NOT reset the timer (accumulated seconds are kept
# and continue on the next in-contact frame); only a longer gap resets to zero.
# Once accumulated contact reaches INTERACTION_MIN_SECS it counts as ONE
# interaction, and the client track id is remembered so it is counted ONCE (id
# checked before every count) and a "+1" is drawn on its box thereafter.
INTERACTION_RADIUS      = 200   # [^] px circle radius at each person's box CENTER
INTERACTION_OVERLAP_MIN = 0.35  # [^] client & seller circles must overlap >= this fraction
INTERACTION_MIN_SECS    = 5   # [^] accumulated contact required before it counts
INTERACTION_GRACE_SECS  = 2     # [^] a no-contact gap up to this long does NOT reset the timer

# --- zones (mark with a polygon picker; feet-in-polygon) ----------------------
# MIRROR: drop reflected people. IGNORE: drop static false detections (e.g. a bag).
# SELLER: anyone standing/seated here is SELLER (covers a seated staffer).
MIRROR_ZONES = [
    [(36, 125), (195, 59), (419, 635), (168, 745), (34, 131)],
    [(1732, 3), (1888, 4), (1788, 493), (1654, 448), (1730, 8)],
    [(904, 3), (1124, 4), (1155, 61), (1161, 240), (929, 276), (902, 10)],
    
]
SELLER_ZONES = [
    [(728, 237), (791, 362), (648, 452), (571, 302), (725, 244)],
    [(218, 697), (502, 684), (1078, 1068), (274, 1068), (212, 695)],
    [(892, 1065), (952, 876), (1291, 873), (1268, 1068), (891, 1068)],
    [(1000, 264), (992, 189), (1152, 174), (1168, 244), (1002, 268)],
    [(676, 607), (548, 498), (528, 385), (614, 353), (680, 603)],
    [(1429, 358), (1468, 247), (1598, 263), (1574, 407), (1428, 360)],
    [(1562, 364), (1580, 296), (1658, 325), (1599, 513), (1562, 364)]
]
SELLER_ZONE_FRAC = 0.50   # [^] fraction of a person's BOX that must be inside a SELLER zone
                        #     (draw the zone over the staff BODY area, not just floor)
IGNORE_ZONES = [
    [(1789, 663), (1838, 680), (1819, 768), (1769, 750), (1788, 665)],
    [(302, 832), (261, 750), (468, 663), (496, 701), (302, 832)],
    [(498, 701), (572, 640), (780, 810), (664, 919), (498, 700)],
    [(492, 710), (456, 659), (549, 618), (610, 690), (531, 745), (486, 707)]
]
IGNORE_ZONE_FRAC = 0.1    # [^] > this fraction of a CLIENT box inside an IGNORE zone -> drop it
                          #     (box-overlap, same method as the ENTER / SELLER zones)
_MIRROR_POLYS = [np.array(z, np.int32) for z in MIRROR_ZONES if len(z) >= 3]
_SELLER_POLYS = [np.array(z, np.int32) for z in SELLER_ZONES if len(z) >= 3]
_IGNORE_POLYS = [np.array(z, np.int32) for z in IGNORE_ZONES if len(z) >= 3]

# --- debug -------------------------------------------------------------------
DEBUG_CALIB = True      # print per-track stats + near-line events at the end
DEBUG_ZONE  = True      # overlay each person's seller-zone% and whole-box white% (turn OFF later)
MAX_FRAMES  = 0         # >0: stop after this many PROCESSED frames (quick test)

_MAX_PIX = 400_000      # cap on accumulated beige pixels when re-learning
_BG_H_MIN, _BG_H_MAX = BEIGE_H_MIN, BEIGE_H_MAX
_BG_S_MAX, _BG_V_MIN = BEIGE_S_MAX, BEIGE_V_MIN


# =============================================================
#  BoT-SORT builder (OSNet ReID) -- inlined
# =============================================================
def _patch_appearance_ema(alpha: float):
    """Override BoT-SORT's per-track feature-EMA momentum so a single bad crop
    barely perturbs a track's appearance template."""
    import boxmot.trackers.bbox.botsort.botsort_track as bt
    if getattr(bt.STrack, "_ema_patched", False):
        bt.STrack.alpha = alpha
        return
    _orig_init = bt.STrack.__init__
    def _init(self, *a, **k):
        _orig_init(self, *a, **k)
        self.alpha = alpha
    bt.STrack.__init__ = _init
    bt.STrack._ema_patched = True


def build_botsort():
    _patch_appearance_ema(APPEARANCE_EMA_ALPHA)
    tuning = dict(
        track_high_thresh=TRACK_HIGH_THRESH,
        track_low_thresh=TRACK_LOW_THRESH,
        new_track_thresh=NEW_TRACK_THRESH,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESH,
        proximity_thresh=PROXIMITY_THRESH,
        appearance_thresh=APPEARANCE_THRESH,
        cmc_method=CMC_METHOD,
    )
    return create_tracker(
        tracker_type="botsort",
        reid_weights=Path(REID_MODEL),
        device=torch.device(DEVICE),
        half=HALF_PRECISION,
        per_class=False,
        evolve_param_dict=tuning,
    )


# =============================================================
#  Geometry / color / zone helpers -- inlined
# =============================================================
def box_overlap_frac(bbox, polys) -> float:
    """Fraction of the person's BOX (grid-sampled) that falls inside any of `polys`.
    Used to decide if a person is 'in' an ENTER zone (>= ENTER_ZONE_FRAC)."""
    if not polys:
        return 0.0
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inside = total = 0
    for px in np.linspace(x1, x2, 6):
        for py in np.linspace(y1, y2, 6):
            total += 1
            if any(cv2.pointPolygonTest(poly, (px, py), False) >= 0 for poly in polys):
                inside += 1
    return inside / total if total else 0.0


def circle_overlap_frac(ca, cb, r) -> float:
    """Intersection area of two EQUAL circles (radius r, centers ca, cb) as a
    fraction of one circle's area. 1.0 = concentric, 0.0 = centers >= 2r apart.
    (r=200: ~0.5 at d=162, ~0.39 at d=200, 0 at d>=400.)"""
    d = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
    if d >= 2 * r:
        return 0.0
    if d <= 0:
        return 1.0
    lens = 2 * r * r * np.arccos(d / (2 * r)) - (d / 2) * np.sqrt(max(0.0, 4 * r * r - d * d))
    return float(lens / (np.pi * r * r))


def _feet_in_polys(xyxy: np.ndarray, polys) -> np.ndarray:
    """Boolean mask: True where a detection's FEET (bottom-center) fall in any poly."""
    if not polys or len(xyxy) == 0:
        return np.zeros((len(xyxy),), dtype=bool)
    feet = np.c_[(xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 3]]
    mask = np.zeros((len(xyxy),), dtype=bool)
    for i, (fx, fy) in enumerate(feet):
        mask[i] = any(cv2.pointPolygonTest(p, (float(fx), float(fy)), False) >= 0
                      for p in polys)
    return mask


def in_mirror_zone(xyxy: np.ndarray) -> np.ndarray:
    return _feet_in_polys(xyxy, _MIRROR_POLYS)


def seller_zone_frac(bbox) -> float:
    """Fraction of the person's BOX (6x6 grid) that falls inside any SELLER zone."""
    if not _SELLER_POLYS:
        return 0.0
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inside = total = 0
    for px in np.linspace(x1, x2, 6):
        for py in np.linspace(y1, y2, 6):
            total += 1
            if any(cv2.pointPolygonTest(poly, (px, py), False) >= 0 for poly in _SELLER_POLYS):
                inside += 1
    return inside / total if total else 0.0


def in_seller_zone(bbox) -> bool:
    """True if at least SELLER_ZONE_FRAC of the person's BOX is inside a SELLER zone
    (grid-sampled box-overlap, not just feet)."""
    return seller_zone_frac(bbox) >= SELLER_ZONE_FRAC


def beige_at_keypoints(frame: np.ndarray, kp_xy_p, kp_cf_p) -> bool:
    """SELLER cue: sample a small patch at each visible hip/knee keypoint and test
    it against the beige band. Returns True if >= BEIGE_KP_MIN of the leg keypoints
    sit on beige pants. A small patch (not one pixel) absorbs keypoint jitter; the
    band's low-saturation cap rejects most bare skin. Occluded legs (low-confidence
    keypoints) are skipped -> a seated staffer relies on the SELLER zone instead."""
    H, W = frame.shape[:2]
    n_beige = 0
    for idx in LOWER_KP_IDX:
        if kp_cf_p[idx] < KP_CONF:                 # keypoint not visible -> skip
            continue
        x, y = int(kp_xy_p[idx][0]), int(kp_xy_p[idx][1])
        x1, y1 = max(0, x - KP_PATCH_R), max(0, y - KP_PATCH_R)
        x2, y2 = min(W, x + KP_PATCH_R + 1), min(H, y + KP_PATCH_R + 1)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        inband = np.all((hsv >= UNIFORM_BAND_LO) & (hsv <= UNIFORM_BAND_HI), axis=2)
        if inband.mean() >= KP_PATCH_FRAC:
            n_beige += 1
    return n_beige >= BEIGE_KP_MIN


def white_at_keypoints(frame: np.ndarray, kp_xy_p, kp_cf_p) -> bool:
    """Second SELLER cue: sample a patch at each visible SHOULDER keypoint and test
    it for WHITE (low saturation, high value). Returns True if >= WHITE_KP_MIN of the
    shoulders are white -> the person is wearing the white shirt."""
    H, W = frame.shape[:2]
    n_white = 0
    for idx in UPPER_KP_IDX:
        if kp_cf_p[idx] < KP_CONF:                 # keypoint not visible -> skip
            continue
        x, y = int(kp_xy_p[idx][0]), int(kp_xy_p[idx][1])
        x1, y1 = max(0, x - KP_PATCH_R), max(0, y - KP_PATCH_R)
        x2, y2 = min(W, x + KP_PATCH_R + 1), min(H, y + KP_PATCH_R + 1)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        white = (hsv[:, :, 1] <= WHITE_S_MAX) & (hsv[:, :, 2] >= WHITE_V_MIN)
        if white.mean() >= WHITE_PATCH_FRAC:
            n_white += 1
    return n_white >= WHITE_KP_MIN


def torso_white_ratio(frame: np.ndarray, bbox) -> float:
    """Fraction of near-white pixels in the upper-torso (shirt) region of a box."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return 0.0
    ty1 = max(0, y1 + int(0.15 * h)); ty2 = max(0, y1 + int(0.45 * h))
    tx1 = max(0, x1 + int(0.20 * w)); tx2 = max(0, x2 - int(0.20 * w))
    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= WHITE_S_MAX) & (hsv[:, :, 2] >= WHITE_V_MIN)
    return float(white.mean())


def body_white_ratio(frame: np.ndarray, bbox) -> float:
    """Fraction of near-white pixels over (most of) the WHOLE person box. Used for
    the SELLER-ZONE white check -- robust when the person is bent over and hair
    covers the chest (unlike the upper-torso torso_white_ratio): it still catches
    the white shirt/pants elsewhere in the box. Edges trimmed 5% to skip background."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return 0.0
    bx1 = max(0, x1 + int(0.05 * w)); bx2 = max(0, x2 - int(0.05 * w))
    by1 = max(0, y1 + int(0.05 * h)); by2 = max(0, y2 - int(0.05 * h))
    crop = frame[by1:by2, bx1:bx2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= WHITE_S_MAX) & (hsv[:, :, 2] >= WHITE_V_MIN)
    return float(white.mean())


def beige_pants_ratio(frame: np.ndarray, bbox) -> float:
    """Fraction of beige/tan pixels in the LOWER body (pants) of a person box
    (broad generic beige gate; the staff-specific band is learned in UniformModel)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return 0.0
    ly1 = max(0, y1 + int(0.55 * h)); ly2 = max(0, y1 + int(0.88 * h))
    lx1 = max(0, x1 + int(0.25 * w)); lx2 = max(0, x2 - int(0.25 * w))
    crop = frame[ly1:ly2, lx1:lx2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    beige = ((hsv[:, :, 0] >= BEIGE_H_MIN) & (hsv[:, :, 0] <= BEIGE_H_MAX) &
             (hsv[:, :, 1] <= BEIGE_S_MAX) & (hsv[:, :, 2] >= BEIGE_V_MIN))
    return float(beige.mean())


def lower_body_hsv(frame: np.ndarray, bbox):
    """HSV pixels (N,3) of the central lower-body (pants) region of a person box."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return None
    ly1 = max(0, y1 + int(0.55 * h)); ly2 = max(0, y1 + int(0.88 * h))
    lx1 = max(0, x1 + int(0.25 * w)); lx2 = max(0, x2 - int(0.25 * w))
    crop = frame[ly1:ly2, lx1:lx2]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int16)


# =============================================================
#  UniformModel -- the SELLER color cue (baked band or re-learn)
# =============================================================
class UniformModel:
    def __init__(self):
        self.pix = []
        self._npix = 0
        self._bounds = None
        self._center = None
        self._fixed = False

    def set_fixed_band(self, lo, hi):
        """Use a pre-learned (baked-in) HSV band -> no images needed at runtime."""
        lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
        self._bounds = (lo, hi)
        self._center = (lo + hi) / 2.0
        self._fixed = True

    def _beige_pixels(self, frame, bbox):
        px = lower_body_hsv(frame, bbox)
        if px is None or len(px) == 0:
            return None
        h, s, v = px[:, 0], px[:, 1], px[:, 2]
        m = (h >= _BG_H_MIN) & (h <= _BG_H_MAX) & (s <= _BG_S_MAX) & (v >= _BG_V_MIN)
        keep = px[m]
        return keep if len(keep) else None

    def _collect(self, frame, bbox):
        if self._fixed or self._npix >= _MAX_PIX:
            return
        keep = self._beige_pixels(frame, bbox)
        if keep is not None:
            self.pix.append(keep)
            self._npix += len(keep)
            self._bounds = None

    def add_seed_sample(self, frame, bbox):
        if beige_pants_ratio(frame, bbox) >= SEED_BEIGE_MIN:
            self._collect(frame, bbox)

    def learn_from_dir(self, ref_dir: str):
        """Re-learn the band from curated seller crops (dev). Prints the band so
        you can copy it into UNIFORM_BAND_*."""
        paths = sorted(glob.glob(os.path.join(ref_dir, "*.png")) +
                       glob.glob(os.path.join(ref_dir, "*.jpg")))
        n = 0
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            box = (0, 0, w, h)
            if beige_pants_ratio(img, box) < SEED_BEIGE_MIN:
                continue
            before = self._npix
            self._collect(img, box)
            n += int(self._npix > before)
        print(f"[UNIFORM] learned beige from {n} crop(s) in '{ref_dir}' "
              f"({self._npix} pixels)")

    def _fit(self):
        allpx = np.concatenate(self.pix, axis=0)
        lo = np.percentile(allpx, COLOR_PCT_LO, axis=0) - COLOR_MARGIN
        hi = np.percentile(allpx, COLOR_PCT_HI, axis=0) + COLOR_MARGIN
        lo = np.maximum(lo, [0, 0, 0])
        hi = np.minimum(hi, [180, 255, 255])
        self._bounds = (lo, hi)
        self._center = np.median(allpx, axis=0)

    def bounds(self):
        if self._bounds is None and self.pix:
            self._fit()
        return self._bounds

    def color_ratio(self, frame, bbox) -> float:
        px = lower_body_hsv(frame, bbox)
        if px is None or len(px) == 0:
            return 0.0
        b = self.bounds()
        if b is None:
            return beige_pants_ratio(frame, bbox)
        lo, hi = b
        mask = np.all((px >= lo) & (px <= hi), axis=1)
        return float(mask.mean())

    def is_uniform(self, frame, bbox):
        """(is_seller, color_ratio, white_ratio) for one detection this frame."""
        color = self.color_ratio(frame, bbox)
        white = torso_white_ratio(frame, bbox)
        return (color >= COLOR_TAKE and white >= WHITE_MIN), color, white


# =============================================================
#  Drawing  (no ids)
# =============================================================
def draw_frame(frame, persons, entry_count, interaction_count=0, served=None,
               persons_kp=None, contacts=None):
    SELLER_COLOR = (0, 0, 255)
    CLIENT_COLOR = (0, 200, 0)
    served = served or set()
    contacts = contacts or set()
    # interaction circles: 200 px at each person's box CENTER (seller red / client
    # green); a CLIENT currently in contact is highlighted yellow + thick.
    for bbox, role, _tid in persons:
        cx = int((bbox[0] + bbox[2]) / 2); cy = int((bbox[1] + bbox[3]) / 2)
        if role == "CLIENT" and _tid is not None and _tid in contacts:
            cv2.circle(frame, (cx, cy), INTERACTION_RADIUS, (0, 255, 255), 3)
        else:
            cv2.circle(frame, (cx, cy), INTERACTION_RADIUS,
                       SELLER_COLOR if role == "SELLER" else CLIENT_COLOR, 1)
    # decision keypoints: LEG dots green=beige / red=not; UPPER dots white=white shirt
    # / blue=not. (These are exactly the pixels the SELLER test reads.)
    if persons_kp:
        H, W = frame.shape[:2]
        for kp_xy_p, kp_cf_p in persons_kp:
            for idx in LOWER_KP_IDX:                 # beige pants check
                if kp_cf_p[idx] < KP_CONF:
                    continue
                x, y = int(kp_xy_p[idx][0]), int(kp_xy_p[idx][1])
                patch = frame[max(0, y - KP_PATCH_R):min(H, y + KP_PATCH_R + 1),
                              max(0, x - KP_PATCH_R):min(W, x + KP_PATCH_R + 1)]
                beige = False
                if patch.size:
                    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                    beige = np.all((hsv >= UNIFORM_BAND_LO) & (hsv <= UNIFORM_BAND_HI),
                                   axis=2).mean() >= KP_PATCH_FRAC
                cv2.circle(frame, (x, y), 4, (0, 255, 0) if beige else (0, 0, 255), -1)
            for idx in UPPER_KP_IDX:                 # white shirt check
                if kp_cf_p[idx] < KP_CONF:
                    continue
                x, y = int(kp_xy_p[idx][0]), int(kp_xy_p[idx][1])
                patch = frame[max(0, y - KP_PATCH_R):min(H, y + KP_PATCH_R + 1),
                              max(0, x - KP_PATCH_R):min(W, x + KP_PATCH_R + 1)]
                white = False
                if patch.size:
                    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                    white = ((hsv[:, :, 1] <= WHITE_S_MAX) &
                             (hsv[:, :, 2] >= WHITE_V_MIN)).mean() >= WHITE_PATCH_FRAC
                cv2.circle(frame, (x, y), 4, (255, 255, 255) if white else (255, 0, 0), -1)
    for bbox, role, _tid in persons:
        x1, y1, x2, y2 = map(int, bbox)
        color = SELLER_COLOR if role == "SELLER" else CLIENT_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(role, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, role, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # served CLIENT -> "+1" on their box, stays until they leave (track gone)
        if role == "CLIENT" and _tid is not None and _tid in served:
            cv2.putText(frame, "+1", (x1 + 2, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # DEBUG: show the two SELLER-ZONE gate values -> zone-overlap% and whole-box white%
        # (drawn INSIDE the box near the top so it is always on-screen)
        if DEBUG_ZONE:
            zf = seller_zone_frac(bbox); wr = body_white_ratio(frame, bbox)
            ty = min(y1 + 20, frame.shape[0] - 4)
            cv2.putText(frame, f"Z={zf*100:.0f}% W={wr*100:.0f}%", (x1 + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, f"Z={zf*100:.0f}% W={wr*100:.0f}%", (x1 + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for poly in _MIRROR_POLYS:
        cv2.polylines(frame, [poly], True, (255, 0, 255), 2, cv2.LINE_AA)
    for poly in _SELLER_POLYS:
        cv2.polylines(frame, [poly], True, (0, 0, 255), 2, cv2.LINE_AA)
    for poly in _IGNORE_POLYS:
        cv2.polylines(frame, [poly], True, (128, 128, 128), 2, cv2.LINE_AA)
    for poly in _ENTER_POLYS_1:                                  # entry zone 1 (crossed first)
        cv2.polylines(frame, [poly], True, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "1", tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    for poly in _ENTER_POLYS_2:                                  # entry zone 2 (reached second)
        cv2.polylines(frame, [poly], True, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "2", tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
    cv2.putText(frame, f"Entries: {entry_count}  Interactions: {interaction_count}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


# =============================================================
#  Main
# =============================================================
def main():
    print("=" * 50)
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
    else:
        print("[GPU] running on CPU.")
    print("=" * 50)

    model = YOLO(MODEL_NAME)
    model.to(DEVICE)
    model.fuse()

    botsort = build_botsort()                 # only for client track ids
    print(f"[UNIFORM] beige band  LO={UNIFORM_BAND_LO.tolist()}  HI={UNIFORM_BAND_HI.tolist()}  "
          f"-> sampled at leg keypoints {LOWER_KP_IDX} (need {BEIGE_KP_MIN} of {len(LOWER_KP_IDX)})")

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    fps        = video_info.fps
    generator  = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    entry_count = 0                           # clients counted entering (zone 1 -> zone 2)
    zone_objs = []                            # id-FREE tracks: [cx, cy, last_zone, counted, last_frame]
    entry_by_tid = {}                         # tid -> ts it was counted entering (for REVERT)
    entry_seller_since = {}                   # tid -> ts its current continuous SELLER streak began

    seller_interactions = {}                  # client tid -> ts it reached MIN_SECS (counted once)
    contact_secs = {}                         # client tid -> accumulated contact seconds in current streak (grace-tolerant)
    contact_last_frame = {}                   # client tid -> processed-frame idx of its last in-contact frame

    stats = {}
    last_persons = []
    last_persons_kp = []
    last_contacts = set()
    processed_frame_idx = -1
    frame_ms_total = 0.0
    timed_frames = 0

    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame_idx, frame in enumerate(generator):
            if frame_idx % SKIP != 0:
                draw_frame(frame, last_persons, entry_count, len(seller_interactions),
                           served=set(seller_interactions), persons_kp=last_persons_kp,
                           contacts=last_contacts)
                sink.write_frame(frame)
                continue
            if MAX_FRAMES and processed_frame_idx + 1 >= MAX_FRAMES:
                break

            processed_frame_idx += 1
            timestamp = frame_idx / fps
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.perf_counter()

            # 1) YOLO
            results = model(frame, conf=DETECT_CONF, classes=[PERSON_CLASS_ID],
                            iou=0.4, imgsz=640, device=DEVICE,
                            half=HALF_PRECISION, verbose=False)
            detections = sv.Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
            )
            # pose KEYPOINTS, in the SAME order as detections above (per-person cue)
            kpts = results[0].keypoints
            if kpts is not None and kpts.conf is not None and len(detections.xyxy) > 0:
                kp_xy = kpts.xy.cpu().numpy()      # (N,17,2) pixel coords
                kp_cf = kpts.conf.cpu().numpy()    # (N,17) per-keypoint confidence
            else:
                kp_xy = np.zeros((len(detections.xyxy), 17, 2))
                kp_cf = np.zeros((len(detections.xyxy), 17))

            # 1b) drop mirror reflections (reflections are never real people, so drop
            # them for EVERYONE here). Apply the SAME mask to the keypoints so kp_xy[i]
            # / kp_cf[i] stay aligned with detections.xyxy[i]. IGNORE zones are handled
            # later CLIENT-ONLY, so a real SELLER standing in one is still kept.
            if _MIRROR_POLYS and len(detections.xyxy) > 0:
                keep = ~in_mirror_zone(detections.xyxy)
                detections = detections[keep]
                kp_xy = kp_xy[keep]
                kp_cf = kp_cf[keep]

            # 2) BoT-SORT on ALL detections -> per-detection track id
            n_det = len(detections.xyxy)
            if n_det > 0:
                dets = np.hstack([
                    detections.xyxy,
                    detections.confidence[:, None],
                    detections.class_id[:, None].astype(float),
                ])
            else:
                dets = np.empty((0, 6))
            tracks = botsort.update(dets, frame)
            tracker_ids = [None] * n_det
            if tracks is not None and len(tracks) > 0:
                for tr in tracks:
                    j = int(tr[7])                       # index into ALL dets
                    if 0 <= j < n_det:
                        tracker_ids[j] = int(tr[4])

            # 3) Role: decided FRESH every processed frame from the leg KEYPOINTS
            #    only -- NO tracking / vote-lock / seed. >= BEIGE_KP_MIN hip/knee
            #    keypoints on beige => SELLER, else CLIENT. The seller ZONE still
            #    forces SELLER (seated staffer, legs hidden). Skipped frames just
            #    reuse the last drawn boxes/labels (top of the loop) -- no logic.
            persons = []
            persons_kp = []
            for i in range(n_det):
                bbox = detections.xyxy[i]
                tid  = tracker_ids[i]

                # SELLER needs BOTH cues: beige pants (>=2/4 leg kpts) AND a white
                # shirt (>=2/4 upper kpts). If the pants aren't beige -> CLIENT right
                # away (white is never checked). The seller ZONE still forces SELLER,
                # but ONLY if the white-shirt check also passes.
                is_beige = beige_at_keypoints(frame, kp_xy[i], kp_cf[i])
                is_white = white_at_keypoints(frame, kp_xy[i], kp_cf[i]) if is_beige \
                    else False
                role = "SELLER" if (is_beige and is_white) else "CLIENT"
                # ZONE override: legs (beige) are hidden here, and shoulder keypoints
                # are unreliable (hair / bent over), so the white check uses the whole
                # upper-torso RECTANGLE instead of the 2 shoulder keypoints.
                if in_seller_zone(bbox) and body_white_ratio(frame, bbox) >= WHITE_MIN:
                    role = "SELLER"                 # per-frame zone override (no tracking)
                if DEBUG_CALIB and tid is not None:
                    st = stats.setdefault(tid, {"frames": 0, "seller": 0, "beige": 0, "white": 0})
                    st["frames"] += 1
                    st["seller"] += int(role == "SELLER")
                    st["beige"]  += int(is_beige)
                    st["white"]  += int(is_white)

                # REVERT: a track counted as an ENTERING client that becomes SELLER
                # (a real staffer misread at the doorway) -- undo the +1, but ONLY if
                # the SELLER role PERSISTS >= REVERT_MIN_SELLER_SECS. A shorter flip is
                # a momentary misread of a real client and must NOT cancel their entry.
                # The whole thing must still happen within ENTRY_REVERT_SECS (enforced
                # by the entry_by_tid expiry below). Fires once per counted entry.
                if tid is not None and tid in entry_by_tid:
                    if role == "SELLER":
                        entry_seller_since.setdefault(tid, timestamp)   # start/continue streak
                        held = timestamp - entry_seller_since[tid]
                        if held >= REVERT_MIN_SELLER_SECS and entry_count > 0:
                            entry_count -= 1
                            print(f"[CNT] f{processed_frame_idx} REVERT -1 (tid {tid} "
                                  f"held SELLER {held:.1f}s, "
                                  f"{timestamp - entry_by_tid[tid]:.1f}s after entry) "
                                  f"-> count={entry_count}")
                            del entry_by_tid[tid]
                            entry_seller_since.pop(tid, None)
                    else:
                        entry_seller_since.pop(tid, None)               # flipped back to CLIENT -> streak broken

                # IGNORE zones drop static false detections (e.g. the handbag on the
                # shelf) -- but CLIENT-ONLY, so a real SELLER standing there is kept.
                # In-zone test = BOX-OVERLAP FRACTION (like the entry/seller zones):
                # > IGNORE_ZONE_FRAC of the box inside an ignore zone -> drop it.
                if role == "CLIENT" and box_overlap_frac(bbox, _IGNORE_POLYS) > IGNORE_ZONE_FRAC:
                    continue
                persons.append((bbox, role, tid))
                persons_kp.append((kp_xy[i], kp_cf[i]))

            # 4a) ENTRY: ORDERED TWO-ZONE pass, id-FREE. We track each client by
            #     POSITION (box center, not BoT-SORT id) -- so a client with NO track
            #     id (occluded) is still followed across the two zones. We remember the
            #     last ENTER zone the person was inside; a 1->2 transition = ENTERING
            #     (+1). A 2->1 transition (leaving) or touching only one zone counts
            #     nothing. Direction is the visiting ORDER -- no line / inward math.
            for bbox, role, tid in persons:
                if role != "CLIENT":
                    continue
                f1 = box_overlap_frac(bbox, _ENTER_POLYS_1)
                f2 = box_overlap_frac(bbox, _ENTER_POLYS_2)
                cz = 0                                  # current zone: 0=neither, 1, or 2
                if f1 >= ENTER_ZONE_FRAC or f2 >= ENTER_ZONE_FRAC:
                    cz = 1 if f1 >= f2 else 2
                cx = (bbox[0] + bbox[2]) / 2.0          # box center (matching point)
                cy = (bbox[1] + bbox[3]) / 2.0
                pf = processed_frame_idx
                # link to the nearest recent track (by position, id-free)
                best = None; best_dist = ZONE_MATCH_DIST
                for o in zone_objs:
                    if pf - o[4] <= ZONE_TTL:
                        dd = np.hypot(cx - o[0], cy - o[1])
                        if dd < best_dist:
                            best_dist = dd; best = o
                if best is None:
                    # start a track only once the person is actually in a zone
                    if cz != 0:
                        zone_objs.append([cx, cy, cz, False, pf])  # [cx,cy,last_zone,counted,last_frame]
                else:
                    prev_zone = best[2]
                    if cz != 0 and cz != prev_zone:     # a zone transition happened
                        if prev_zone == 1 and cz == 2 and not best[3]:
                            entry_count += 1
                            best[3] = True              # counted -> don't recount this pass
                            if tid is not None:         # remember id so a later SELLER flip can revert it
                                entry_by_tid[tid] = timestamp
                            print(f"[CNT] f{pf} ENTER (1->2) at ({cx:.0f},{cy:.0f}) "
                                  f"-> count={entry_count}")
                        best[2] = cz                    # 2->1 (leaving) just updates, no count
                    best[0], best[1], best[4] = cx, cy, pf   # keep position/frame fresh

            # expire stale in-zone tracks (id-free entree counter)
            zone_objs = [o for o in zone_objs if processed_frame_idx - o[4] <= ZONE_TTL]
            # forget entrants past the revert window (so a reused id can't revert late)
            entry_by_tid = {t: ts for t, ts in entry_by_tid.items()
                            if timestamp - ts <= ENTRY_REVERT_SECS}
            entry_seller_since = {t: ts for t, ts in entry_seller_since.items()
                                  if t in entry_by_tid}

            # 4b) INTERACTION (PEC): a CLIENT within INTERACTION_DIST px of ANY seller
            #     accumulates contact time. The distance test is OMNIDIRECTIONAL (near
            #     any seller, any direction). The streak is GRACE-TOLERANT: a no-contact
            #     gap up to INTERACTION_GRACE_SECS (client not detected, occluded, or
            #     turned away) does NOT reset the timer -- the accumulated seconds are
            #     kept and continue on the next in-contact frame. Only a gap longer than
            #     the grace resets to zero. When the accumulated contact reaches
            #     INTERACTION_MIN_SECS it counts as ONE interaction; the client tid is
            #     remembered so we count it ONCE (id checked before every count) and a
            #     "+1" is drawn on its box thereafter.
            grace_frames = max(1, round(INTERACTION_GRACE_SECS * fps / SKIP))
            # circles are centered on each person's BOX CENTER
            seller_centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                              for b, r, _t in persons if r == "SELLER"]
            pf = processed_frame_idx
            contact_now = set()                                 # client tids in contact THIS frame
            for b, r, tid in persons:
                if r != "CLIENT" or tid is None:
                    continue
                if tid in seller_interactions:                  # already counted this client (id check)
                    continue
                cc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)      # client circle center
                close = any(
                    circle_overlap_frac(cc, sc, INTERACTION_RADIUS) >= INTERACTION_OVERLAP_MIN
                    for sc in seller_centers)                   # circles overlap >= 50% with ANY seller
                if close:
                    contact_now.add(tid)
                    last = contact_last_frame.get(tid)
                    if last is not None and pf - last <= grace_frames:  # within grace -> keep the streak
                        contact_secs[tid] = contact_secs.get(tid, 0.0) + SKIP / fps
                    else:                                       # first contact or grace exceeded -> fresh streak
                        contact_secs[tid] = SKIP / fps
                    contact_last_frame[tid] = pf
                    if contact_secs[tid] >= INTERACTION_MIN_SECS:
                        seller_interactions[tid] = timestamp    # threshold reached -> count once
                        contact_secs.pop(tid, None)
                        contact_last_frame.pop(tid, None)
                        print(f"[PEC] f{processed_frame_idx} INTERACTION +1 (client tid {tid} "
                              f"at t={timestamp:.1f}s) -> total={len(seller_interactions)}")
            # drop streaks whose no-contact gap now EXCEEDS the grace window -> the client
            # has really left; the timer resets to zero on their next contact
            for tid in list(contact_last_frame):
                if pf - contact_last_frame[tid] > grace_frames:
                    contact_secs.pop(tid, None)
                    contact_last_frame.pop(tid, None)

            # 5) Draw + write
            draw_frame(frame, persons, entry_count, len(seller_interactions),
                       served=set(seller_interactions), persons_kp=persons_kp,
                       contacts=contact_now)
            sink.write_frame(frame)
            last_persons = persons
            last_persons_kp = persons_kp
            last_contacts = contact_now

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_ms_total += (time.perf_counter() - _t0) * 1000.0
            timed_frames   += 1

    del model, botsort
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if DEBUG_CALIB:
        print(f"\n[CALIB] beige band  LO={UNIFORM_BAND_LO.tolist()}  HI={UNIFORM_BAND_HI.tolist()}"
              f"  (sampled at leg keypoints {LOWER_KP_IDX}, need {BEIGE_KP_MIN})")
        for tid, st in sorted(stats.items(), key=lambda kv: -kv[1]["frames"])[:20]:
            print(f"  tid {tid:>4}: frames={st['frames']:>4} "
                  f"seller_frames={st['seller']:>4} beige_frames={st['beige']:>4}")

    print("\n========== RESULT ==========")
    print(f"Processed frames : {timed_frames}")
    print(f"Clients entered  : {entry_count}")
    print(f"Interactions     : {len(seller_interactions)}")
    print(f"Output video     : {TARGET_VIDEO_PATH}")
    if timed_frames > 0:
        print(f"Full pipeline    : {frame_ms_total / timed_frames:.1f} ms/frame "
              f"({timed_frames / (frame_ms_total / 1000.0):.1f} FPS)")
    print("============================")


if __name__ == "__main__":
    main()