import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dataset_writer import FEATURE_KEYPOINTS, FEATURE_ANGLES, ERROR_LABELS_ORDER


def _pelvis_center(kp: Dict[str, List[float]]) -> Optional[Tuple[float, float]]:
    lh = kp.get("LEFT_HIP")
    rh = kp.get("RIGHT_HIP")
    if isinstance(lh, (list, tuple)) and len(lh) == 2 and isinstance(rh, (list, tuple)) and len(rh) == 2:
        return (float(lh[0] + rh[0]) / 2.0, float(lh[1] + rh[1]) / 2.0)
    return None


def _hip_distance(kp: Dict[str, List[float]]) -> float:
    lh = kp.get("LEFT_HIP")
    rh = kp.get("RIGHT_HIP")
    if (
        isinstance(lh, (list, tuple)) and len(lh) == 2 and
        isinstance(rh, (list, tuple)) and len(rh) == 2
    ):
        dx = float(lh[0]) - float(rh[0])
        dy = float(lh[1]) - float(rh[1])
        d = math.hypot(dx, dy)
        return max(d, 1e-6)
    return 1.0


def center_scale_keypoints(
    kp: Dict[str, Any],
    center: str = "pelvis",
    scale_mode: Optional[str] = "hip_distance",
) -> Dict[str, List[float]]:
    
    out: Dict[str, List[float]] = {}
    for name, pt in (kp or {}).items():
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            out[name] = [float(pt[0]), float(pt[1])]

    # Determine center
    cx, cy = 0.0, 0.0
    cpt = None
    if center == "pelvis":
        cpt = _pelvis_center(out)
    if cpt is not None:
        cx, cy = cpt

    # Determine scale
    scale = 1.0
    if scale_mode == "hip_distance":
        scale = _hip_distance(out)
    scale = max(scale, 1e-6)

    # Apply transform
    transformed: Dict[str, List[float]] = {}
    for name in FEATURE_KEYPOINTS:  # keep only the needed order subset
        pt = out.get(name)
        if pt is None:
            transformed[name] = [0.0, 0.0]
        else:
            x = (pt[0] - cx) / scale
            y = (pt[1] - cy) / scale
            transformed[name] = [float(x), float(y)]
    return transformed


def build_feature_vector(
    kp_centered: Dict[str, List[float]],
    angles: Dict[str, Any],
) -> List[float]:
    """Construct numeric feature vector [kp(x,y)*, angles*] in the canonical order."""
    vec: List[float] = []
    for name in FEATURE_KEYPOINTS:
        xy = kp_centered.get(name, [0.0, 0.0])
        x, y = 0.0, 0.0
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            try:
                x, y = float(xy[0]), float(xy[1])
            except Exception:
                x, y = 0.0, 0.0
        vec.extend([x, y])
    for a in FEATURE_ANGLES:
        try:
            v = float(angles.get(a, 0.0) or 0.0)
        except Exception:
            v = 0.0
        # Optional angle normalization (0..180 is typical) but keep raw degrees
        vec.append(v)
    return vec


def ema_smooth_sequence(
    seq: List[List[float]],
    feature_dim: int,
    smooth_keypoints_only: bool = True,
    alpha: float = 0.15,
) -> List[List[float]]:
    """Apply simple EMA smoothing along time.

    - If smooth_keypoints_only=True, smooth only first 2*len(FEATURE_KEYPOINTS) values per frame (x,y coords), keep angles intact.
    """
    if not seq:
        return seq
    T = len(seq)
    D = feature_dim
    if D <= 0:
        D = len(seq[0])
    arr = np.asarray(seq, dtype=np.float32)

    if smooth_keypoints_only:
        K = 2 * len(FEATURE_KEYPOINTS)
        end = min(K, arr.shape[1])
    else:
        end = arr.shape[1]

    # EMA per column up to end
    em = arr.copy()
    for j in range(end):
        prev = em[0, j]
        for t in range(1, T):
            prev = alpha * em[t, j] + (1.0 - alpha) * prev
            em[t, j] = prev
    return em.tolist()


def preprocess_record(
    record: Dict[str, Any],
    center: str = "pelvis",
    scale_mode: Optional[str] = "hip_distance",
    ema_alpha: Optional[float] = 0.15,
) -> Dict[str, Any]:
    """Return a new record with sequence rebuilt using pelvis-centered keypoints
    and optional EMA smoothing.
    """
    frames = record.get("frames") or []
    new_seq: List[List[float]] = []
    for fr in frames:
        kp = fr.get("keypoints_normalized") or {}
        ang = fr.get("angles") or {}
        kp_c = center_scale_keypoints(kp, center=center, scale_mode=scale_mode)
        vec = build_feature_vector(kp_c, ang)
        new_seq.append(vec)

    feature_dim = 2 * len(FEATURE_KEYPOINTS) + len(FEATURE_ANGLES)
    if ema_alpha is not None and 0.0 < ema_alpha < 1.0:
        new_seq = ema_smooth_sequence(new_seq, feature_dim=feature_dim, smooth_keypoints_only=True, alpha=ema_alpha)

    out = dict(record)
    out["sequence"] = new_seq
    out["seq_len"] = len(new_seq)
    out["feature_dim"] = feature_dim
    # Update feature_names since keypoints are now pelvis-centered (semantics changed subtly)
    feat_names: List[str] = []
    for n in FEATURE_KEYPOINTS:
        feat_names.extend([f"{n}.x_centered", f"{n}.y_centered"])
    feat_names.extend(FEATURE_ANGLES)
    out["feature_names"] = feat_names
    return out


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_dataset(
    records: List[Dict[str, Any]]
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]]:
    """Build X, y, feature_names, label_names from preprocessed records.

    Returns:
      X_list: list of np.ndarray, each shape [T, D]
      y_list: list of np.ndarray, each shape [C]
      feature_names: list[str]
      label_names: list[str]
    """
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    feature_names: List[str] = []
    label_names: List[str] = ERROR_LABELS_ORDER.copy()

    for rec in records:
        seq = rec.get("sequence") or []
        X_list.append(np.asarray(seq, dtype=np.float32))
        labels = rec.get("labels") or {}
        y = np.array([1 if bool(labels.get(k, False)) else 0 for k in label_names], dtype=np.int32)
        y_list.append(y)
        if not feature_names:
            feature_names = list(rec.get("feature_names") or [])

    return X_list, y_list, feature_names, label_names
