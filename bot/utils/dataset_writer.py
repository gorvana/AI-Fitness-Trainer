import os
import re
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# Minimal, Keras-friendly dataset writer
DATASET_DIR = os.path.join(os.getcwd(), "dataset")
SEQUENCES_JSONL = os.path.join(DATASET_DIR, "sequences.jsonl")

SCHEMA_VERSION = "1.0"


def _ensure_dataset_dir() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)


# ==========================
# Feature layout (per frame)
# ==========================

# We will use only normalized keypoints and angles (numbers only)
FEATURE_KEYPOINTS = [
    "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
    "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "NOSE",
]

FEATURE_ANGLES = [
    "LEFT_KNEE_ANGLE", "RIGHT_KNEE_ANGLE",
    "LEFT_HIP_ANGLE", "RIGHT_HIP_ANGLE",
    "LEFT_TORSO_ANGLE", "RIGHT_TORSO_ANGLE",
]

# Multi-label taxonomy (kept simple and explicit)
ERROR_LABELS_ORDER = [
    "knees_in",         # колени уходят внутрь
    "shallow_depth",    # недостаточная глубина
    "heels_off",        # пятки отрываются от пола
    "forward_lean",     # чрезмерный наклон корпуса вперёд
]


def _extract_frame_index(path: Optional[str]) -> int:
    if not path:
        return 10**9
    m = re.search(r"_frame_(\d+)\.", os.path.basename(path))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 10**9
    return 10**9


def _frame_feature_vector(frame: Dict[str, Any]) -> List[float]:
    """Build a numeric vector for one frame: [kp(x,y)*, angles*]. Missing -> 0.0"""
    kp_norm = frame.get("keypoints_normalized") or {}
    angles = frame.get("angles") or {}

    vec: List[float] = []
    for name in FEATURE_KEYPOINTS:
        pt = kp_norm.get(name)
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            try:
                x, y = float(pt[0]), float(pt[1])
            except Exception:
                x, y = 0.0, 0.0
        else:
            x, y = 0.0, 0.0
        vec.extend([x, y])

    for a in FEATURE_ANGLES:
        try:
            v = float(angles.get(a, 0.0) or 0.0)
        except Exception:
            v = 0.0
        vec.append(v)

    return vec


def _canonicalize_error_label(label: Optional[str]) -> Dict[str, Any]:
    """
    Convert free-form label (caption) into a multi-label dict + original string.
    Unrecognized -> all False.
    """
    label_raw = (label or "").strip().lower()

    # Basic RU/EN synonyms -> canonical keys
    synonyms = {
        "knees_in": "knees_in",
        "колени внутрь": "knees_in",
        "колени вовнутрь": "knees_in",
        "valgus": "knees_in",

        "shallow_depth": "shallow_depth",
        "недостаточная глубина": "shallow_depth",
        "мелко": "shallow_depth",
        "маленькая глубина": "shallow_depth",

        "heels_off": "heels_off",
        "пятки отрываются": "heels_off",
        "оторваны пятки": "heels_off",

        "forward_lean": "forward_lean",
        "наклон вперед": "forward_lean",
        "наклон вперёд": "forward_lean",
    }

    canonical = synonyms.get(label_raw)
    labels = {k: False for k in ERROR_LABELS_ORDER}
    if canonical in labels:
        labels[canonical] = True
    else:
        return None
    
    return {"label": label_raw or None, "labels": labels}


def write_sequence_record(
    summary: Dict[str, Any],
    video_path: Optional[str] = None,
    error_label: Optional[str] = None,
) -> str:
    """
    Append one training sample into sequences.jsonl.

    Stored format (Keras-friendly):
    - sequence: list[list[float]] without padding (pad later in dataloader)
    - feature_names: order of features in each frame vector
    - frames: optional lightweight raw per-frame dicts (keypoints_normalized, angles, frame_index)
    - labels: multi-label dict in ERROR_LABELS_ORDER, plus original label string for traceability
    """
    _ensure_dataset_dir()

    # Determine video_id
    video_filename = os.path.basename(video_path) if video_path else os.path.basename(
        str(summary.get("min_knee_frame_path", "unknown"))
    )
    video_id = os.path.splitext(video_filename)[0]

    # Sort frames by index to preserve time order
    results = summary.get("results") or []
    sorted_frames = sorted(
        [r for r in results if isinstance(r, dict)],
        key=lambda r: _extract_frame_index(r.get("image_path")),
    )

    # Build feature names (once)
    feature_names: List[str] = []
    for name in FEATURE_KEYPOINTS:
        feature_names.extend([f"{name}.x", f"{name}.y"])
    feature_names.extend(FEATURE_ANGLES)

    # Build numeric sequence and compact per-frame dicts for traceability
    sequence: List[List[float]] = []
    frames_out: List[Dict[str, Any]] = []
    for fr in sorted_frames:
        sequence.append(_frame_feature_vector(fr))
        frames_out.append({
            "frame_index": _extract_frame_index(fr.get("image_path")),
            "keypoints_normalized": fr.get("keypoints_normalized") or {},
            "angles": fr.get("angles") or {},
        })

    canon = _canonicalize_error_label(error_label)

    created_at = datetime.now(timezone.utc).isoformat()
    record = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "video_id": video_id,
        "video_path": video_path,
        "seq_len": len(sequence),
        "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "sequence": sequence,           # shape [T, D], no padding here
        "frames": frames_out,           # per-frame dicts for optional inspection
        "labels": canon["labels"],     # multi-label one-hot in ERROR_LABELS_ORDER
        "label": canon["label"],       # original raw label (string)
        "label_names": ERROR_LABELS_ORDER,
    }

    with open(SEQUENCES_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return SEQUENCES_JSONL
