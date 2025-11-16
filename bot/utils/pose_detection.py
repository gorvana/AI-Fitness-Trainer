import mediapipe as mp
import cv2
import logging
from typing import List, Optional, Dict
import math

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_PATH = "frames/1.jpg"

    
def landmark_to_pixel(landmark, image_width:int, image_height:int)->tuple[int, int]:
    x_px = min(int(landmark.x * image_width), image_width - 1)
    y_px = min(int(landmark.y * image_height), image_height - 1)
    return x_px, y_px

def angle_three_points(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> float:
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    mag_a = math.hypot(ax, ay)
    mag_c = math.hypot(cx, cy)
    if mag_a == 0 or mag_c == 0:
        return 0.0
    dot = ax * cx + ay * cy
    cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(cos_angle))

def draw_squat_overlay(
        image,
        keypoints: Dict[str, tuple[int, int]],
        angles: Dict[str, float]
)-> None:
    circle_color = (0, 255, 0)
    line_color = (255, 0, 0)
    text_color = (0, 0, 255)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    for name, (x,y) in keypoints.items():
        cv2.circle(image, (x,y), 6, circle_color, -1)
        cv2.putText(image, name, (x+6, y-6), font, 0.45, text_color, 1, cv2.LINE_AA)
    
    def line(a: str, b: str):
        if a in keypoints and b in keypoints:   
            cv2.line(image, keypoints[a], keypoints[b], line_color, thickness)

    line("LEFT_SHOULDER", "LEFT_HIP")
    line("LEFT_HIP", "LEFT_KNEE")
    line("LEFT_KNEE", "LEFT_ANKLE")

    # Правая цепочка
    line("RIGHT_SHOULDER", "RIGHT_HIP")
    line("RIGHT_HIP", "RIGHT_KNEE")
    line("RIGHT_KNEE", "RIGHT_ANKLE")

    # Подписи углов
    def put_angle(key_angle: str, point_name: str, dy: int = 20):
        if key_angle in angles and point_name in keypoints:
            x, y = keypoints[point_name]
            cv2.putText(
                image,
                f"{int(angles[key_angle])}",
                (x + 8, y + dy),
                font,
                0.5,
                text_color,
                1,
                cv2.LINE_AA
            )

    put_angle("LEFT_KNEE_ANGLE", "LEFT_KNEE")
    put_angle("RIGHT_KNEE_ANGLE", "RIGHT_KNEE")
    put_angle("LEFT_HIP_ANGLE", "LEFT_HIP", dy=-24)
    put_angle("RIGHT_HIP_ANGLE", "RIGHT_HIP", dy=-24)


def process_single_frame(image_path, show: bool = False) -> Optional[Dict]: 
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Image at path {image_path} could not be loaded.")
            return None

        image_height, image_width, _ = image.shape
        original_image = image.copy()

        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            logger.error("No pose landmarks detected.")
            cv2.imshow('Original Image', original_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return None

        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        needed = [
            "NOSE",
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE",
            "LEFT_ANKLE", "RIGHT_ANKLE"
        ]

        lm = results.pose_landmarks.landmark
        keypoints_pixels: Dict[str, tuple[int, int]] = {}
        keypoints_normalized: Dict[str, tuple[float, float]] = {}

        for name in needed:
            try:
                landmark = lm[getattr(mp_pose.PoseLandmark, name).value]
                keypoints_pixels[name] = landmark_to_pixel(landmark, image_width, image_height)
                keypoints_normalized[name] = (landmark.x, landmark.y)
            except Exception:
                continue

        angles: Dict[str, float] = {}

        def safe_angle(a_name, b_name, c_name, key: str):
            if a_name in keypoints_pixels and b_name in keypoints_pixels and c_name in keypoints_pixels:
                angles[key] = angle_three_points(
                    keypoints_pixels[a_name],
                    keypoints_pixels[b_name],
                    keypoints_pixels[c_name]
                )

        # Колено: hip - knee - ankle
        safe_angle("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "LEFT_KNEE_ANGLE")
        safe_angle("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_KNEE_ANGLE")

        # Тазобедренный: shoulder - hip - knee
        safe_angle("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE", "LEFT_HIP_ANGLE")
        safe_angle("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_HIP_ANGLE")

        # Наклон корпуса (грубый): NOSE - SHOULDER - HIP (можно использовать для оценки прогиба спины)
        safe_angle("NOSE", "LEFT_SHOULDER", "LEFT_HIP", "LEFT_TORSO_ANGLE")
        safe_angle("NOSE", "RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_TORSO_ANGLE")


        if show:
            draw_squat_overlay(annotated_image, keypoints_pixels, angles)
            cv2.imshow("Original Image", original_image)
            cv2.imshow("Annotated Image", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        payload = {
            "image_path": image_path,
            "size": (image_width, image_height),
            "keypoints_pixels": keypoints_pixels,
            "keypoints_normalized": keypoints_normalized,
            "angles": angles
        }

        logger.info("Кадр обработан успешно.")
        return payload

def process_frames_batch(image_paths: list[str], show: bool = False) -> list[Optional[Dict]]:
    results: List[Dict] = []
    for frame in image_paths:
        result = process_single_frame(frame, show=show)
        if result is not None:
            results.append(result)
    return results

if __name__ == "__main__":
    data = process_single_frame(DEFAULT_IMAGE_PATH, show=True)
    if data:
        print(data)
    else:
        print("Поза не распознана.")