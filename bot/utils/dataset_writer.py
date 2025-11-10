import os
import json
from typing import Any, Dict, List, Tuple, Optional


DATASET_DIR = os.path.join(os.getcwd(), "dataset")
VIDEOS_JSONL = os.path.join(DATASET_DIR, "videos.jsonl")
FRAMES_JSONL = os.path.join(DATASET_DIR, "frames.jsonl")


def _ensure_dataset_dir() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)


def _to_list_point(value: Any) -> List[float]:
    # Convert (x, y) to a JSON-serializable list of floats
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [float(value[0]), float(value[1])]
    return []


def _clamp01(v: float) -> float:
    try:
        v = float(v)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _normalize_keypoints_normalized(kp_norm: Dict[str, Any]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for name, pt in kp_norm.items():
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            out[name] = [_clamp01(pt[0]), _clamp01(pt[1])]
    return out


def _sanitize_keypoints_pixels(kp_px: Dict[str, Any]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for name, pt in kp_px.items():
        out[name] = _to_list_point(pt)
    return out


def append_video_and_frames(summary: Dict[str, Any], video_path: Optional[str] = None) -> Tuple[str, str]:
    """
    Append processed summary to dataset JSONL files.

    - videos.jsonl: one record per input video (aggregate summary)
    - frames.jsonl: one record per processed frame (detailed keypoints/angles)

    Returns: (videos_jsonl_path, frames_jsonl_path)
    """
    _ensure_dataset_dir()

    # Identify the video source
    video_filename = None
    if video_path:
        video_filename = os.path.basename(video_path)
    if not video_filename:
        # fall back to the frame path if available
        video_filename = os.path.basename(
            str(summary.get("min_knee_frame_path", "unknown"))
        )

    video_id = os.path.splitext(video_filename)[0]

    # Aggregate video-level record
    video_record = {
        "video_id": video_id,
        "video_path": video_path,
        "frames_count": int(summary.get("frames_count", 0) or 0),
        "processed_count": int(summary.get("processed_count", 0) or 0),
        "min_knee_angle": float(summary.get("min_knee_angle", 0.0) or 0.0),
        "min_knee_frame_path": summary.get("min_knee_frame_path"),
        "min_knee_annotated_path": summary.get("min_knee_annotated_path"),
    }

    # Write/append the video record (append mode ensures old data remains)
    with open(VIDEOS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(video_record, ensure_ascii=False) + "\n")

    # Frame-level detailed records
    results = summary.get("results", []) or []
    if isinstance(results, list):
        with open(FRAMES_JSONL, "a", encoding="utf-8") as f:
            for r in results:
                if not isinstance(r, dict):
                    continue
                frame_path = r.get("image_path")
                size = r.get("size") or r.get("frame_size")
                width, height = None, None
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    width, height = int(size[0]), int(size[1])

                keypoints_px = _sanitize_keypoints_pixels(
                    r.get("keypoints_pixels", {}) or {}
                )
                keypoints_norm = _normalize_keypoints_normalized(
                    r.get("keypoints_normalized", {}) or {}
                )
                angles = r.get("angles", {}) or {}
                # ensure angles numeric
                angles_clean = {}
                for k, v in angles.items():
                    try:
                        angles_clean[k] = float(v)
                    except Exception:
                        continue

                frame_record = {
                    "video_id": video_id,
                    "frame_path": frame_path,
                    "frame_width": width,
                    "frame_height": height,
                    "keypoints_pixels": keypoints_px,
                    "keypoints_normalized": keypoints_norm,
                    "angles": angles_clean,
                }
                f.write(json.dumps(frame_record, ensure_ascii=False) + "\n")

    return VIDEOS_JSONL, FRAMES_JSONL
