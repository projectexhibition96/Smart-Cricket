"""
Pose + shot classification + form feedback (extracted from live webcam script).
Used by Flask /predict_frame with images from the browser.
"""
import json
import os
from collections import deque
from typing import Any, Dict, Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# Same mapping as live_feedback.py — adjust keys to match your model's class labels.
DEFAULT_SHOT_TO_STANCE = {
    "cover_drive": "side_on_stance",
    "straight_drive": "side_on_stance",
    "pull_shot": "open_stance",
    "hook_shot": "open_stance",
    "sweep_shot": "open_stance",
}

# Server-side smoothing (last N raw predictions)
SMOOTH_WINDOW = int(os.getenv("POSE_SMOOTH_WINDOW", "10"))
_prediction_history: Dict[str, deque] = {}


def _session_deque(session_id: str) -> deque:
    if session_id not in _prediction_history:
        _prediction_history[session_id] = deque(maxlen=SMOOTH_WINDOW)
    return _prediction_history[session_id]


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return 0
    cosine = np.dot(ba, bc) / denom
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return int(np.degrees(np.arccos(cosine)))


def get_feedback(joint_name, user, ideal, tolerance):
    diff = user - ideal
    if abs(diff) <= tolerance:
        return f"{joint_name}: OK"
    if joint_name == "Elbow":
        return "Elbow: Raise" if diff < 0 else "Elbow: Lower"
    if joint_name == "Knee":
        return "Knee: Straight" if diff < 0 else "Knee: Bend"
    return f"{joint_name}: adjust"


def is_correct(user, ideal, tolerance):
    return abs(user - ideal) <= tolerance


class PoseLivePipeline:
    def __init__(self):
        self.model = None
        self.dataset: Dict[str, Any] = {}
        self.shot_to_stance = dict(DEFAULT_SHOT_TO_STANCE)
        self.pose = None

    def ensure_loaded(self):
        if self.model is not None:
            return
        model_path = os.getenv("SHOT_MODEL_PATH", "shot_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Shot model not found: {model_path}. Place shot_model.pkl or set SHOT_MODEL_PATH."
            )
        self.model = joblib.load(model_path)

        angles_path = os.getenv("IDEAL_ANGLES_PATH", "ideal_batting_angles.json")
        if os.path.exists(angles_path):
            with open(angles_path, encoding="utf-8") as f:
                self.dataset = json.load(f)
        else:
            self.dataset = {}

        stance_map_path = os.getenv("SHOT_TO_STANCE_PATH", "")
        if stance_map_path and os.path.exists(stance_map_path):
            with open(stance_map_path, encoding="utf-8") as f:
                self.shot_to_stance = json.load(f)

        # Browser frames arrive as independent images and may be sent concurrently.
        # Using video-mode tracking across requests can trigger MediaPipe timestamp/state errors.
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=int(os.getenv("MEDIAPIPE_MODEL_COMPLEXITY", "1")),
            min_detection_confidence=float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.5")),
            min_tracking_confidence=float(os.getenv("MIN_TRACKING_CONFIDENCE", "0.5")),
        )

    def predict_from_image_bytes(self, image_bytes: bytes, session_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Returns (payload_dict, error_message).
        """
        self.ensure_loaded()
        assert self.pose is not None and self.model is not None

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, "Could not decode image"

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return {
                "pose_detected": False,
                "shot": None,
                "confidence": None,
                "stance": None,
                "elbow_feedback": None,
                "knee_feedback": None,
                "total_score": None,
                "elbow_angle": None,
                "knee_angle": None,
            }, None

        lm_list = results.pose_landmarks.landmark

        keypoints = []
        for lm in lm_list:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        keypoints_np = np.array(keypoints, dtype=np.float32).reshape(1, -1)

        try:
            raw_pred = self.model.predict(keypoints_np)[0]
            predicted_raw = str(raw_pred)
        except Exception as e:
            return None, f"Model predict failed: {e}"

        dq = _session_deque(session_id or "default")
        dq.append(predicted_raw)
        predicted_shot = str(max(set(dq), key=dq.count))

        confidence = None
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(keypoints_np)
                confidence = float(np.max(probs))
            except Exception:
                pass

        selected_stance = self.shot_to_stance.get(predicted_shot, "side_on_stance")
        ideal = self.dataset.get(selected_stance) if self.dataset else None

        shoulder = [
            lm_list[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
            lm_list[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
        ]
        elbow = [
            lm_list[mp_pose.PoseLandmark.LEFT_ELBOW].x,
            lm_list[mp_pose.PoseLandmark.LEFT_ELBOW].y,
        ]
        wrist = [
            lm_list[mp_pose.PoseLandmark.LEFT_WRIST].x,
            lm_list[mp_pose.PoseLandmark.LEFT_WRIST].y,
        ]
        hip = [lm_list[mp_pose.PoseLandmark.LEFT_HIP].x, lm_list[mp_pose.PoseLandmark.LEFT_HIP].y]
        knee = [lm_list[mp_pose.PoseLandmark.LEFT_KNEE].x, lm_list[mp_pose.PoseLandmark.LEFT_KNEE].y]
        ankle = [lm_list[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm_list[mp_pose.PoseLandmark.LEFT_ANKLE].y]

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        knee_angle = calculate_angle(hip, knee, ankle)

        payload: Dict[str, Any] = {
            "pose_detected": True,
            "shot": predicted_shot,
            "raw_shot": predicted_raw,
            "confidence": confidence,
            "stance": selected_stance,
            "elbow_angle": elbow_angle,
            "knee_angle": knee_angle,
        }

        if ideal and isinstance(ideal, dict):
            elbow_feedback = get_feedback("Elbow", elbow_angle, ideal["elbow_mean"], ideal["tolerance"])
            knee_feedback = get_feedback("Knee", knee_angle, ideal["knee_mean"], ideal["tolerance"])
            elbow_correct = is_correct(elbow_angle, ideal["elbow_mean"], ideal["tolerance"])
            knee_correct = is_correct(knee_angle, ideal["knee_mean"], ideal["tolerance"])
            elbow_score = max(0, 100 - abs(elbow_angle - ideal["elbow_mean"]))
            knee_score = max(0, 100 - abs(knee_angle - ideal["knee_mean"]))
            total_score = int((elbow_score + knee_score) / 2)
            payload.update(
                {
                    "elbow_feedback": elbow_feedback,
                    "knee_feedback": knee_feedback,
                    "elbow_correct": elbow_correct,
                    "knee_correct": knee_correct,
                    "total_score": total_score,
                }
            )
        else:
            payload.update(
                {
                    "elbow_feedback": None,
                    "knee_feedback": None,
                    "total_score": None,
                }
            )

        return payload, None
