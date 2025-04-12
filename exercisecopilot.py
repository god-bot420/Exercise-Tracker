import cv2
import numpy as np
import mediapipe as mp
import json
from abc import ABC, abstractmethod

# Load configuration
with open("config.json", "r") as config_file:
    CONFIG = json.load(config_file)


class ExerciseTracker(ABC):
    def __init__(self, exercise_name):
        self.exercise_name = exercise_name
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Font settings
        font_settings = CONFIG["font_settings"]
        self.font = eval(font_settings["font"])
        self.font_scale = font_settings["font_scale"]
        self.font_thickness = font_settings["font_thickness"]

        # Angle thresholds
        thresholds = CONFIG["angle_thresholds"][exercise_name]
        self.down_angle = thresholds["down_angle"]
        self.up_angle = thresholds["up_angle"]

        # Counter variables
        self.left_count = 0
        self.right_count = 0
        self.left_stage = "down"
        self.right_stage = "down"
        self.stable_frames_required = 3
        self.left_stable_frames = 0
        self.right_stable_frames = 0

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
            return None
        ba, bc = a - b, c - b
        if np.all(ba == 0) or np.all(bc == 0):
            return None
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    @abstractmethod
    def detect_movement(self, landmarks, side):
        """Detect movement for a specific side."""
        pass

    def count_reps(self, landmarks):
        """Count repetitions for both sides."""
        right_angle, right_position = self.detect_movement(landmarks, "right")
        left_angle, left_position = self.detect_movement(landmarks, "left")

        # Process right side
        self._update_stage(right_position, "right")

        # Process left side
        self._update_stage(left_position, "left")

        return right_angle, left_angle

    def _update_stage(self, position, side):
        """Update stage and count reps for a specific side."""
        if side == "right":
            stage, stable_frames, count = self.right_stage, self.right_stable_frames, self.right_count
        else:
            stage, stable_frames, count = self.left_stage, self.left_stable_frames, self.left_count

        if position == "down" and stage == "up":
            stable_frames += 1
            if stable_frames >= self.stable_frames_required:
                count += 1
                stage = "down"
                stable_frames = 0
        elif position == "up" and stage == "down":
            stable_frames += 1
            if stable_frames >= self.stable_frames_required:
                stage = "up"
                stable_frames = 0
        else:
            stable_frames = 0

        if side == "right":
            self.right_stage, self.right_stable_frames, self.right_count = stage, stable_frames, count
        else:
            self.left_stage, self.left_stable_frames, self.left_count = stage, stable_frames, count

    def process_frame(self, frame):
        """Process a single frame of video."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            right_angle, left_angle = self.count_reps(results.pose_landmarks.landmark)
            self._display_angles(frame, right_angle, left_angle)

        self._display_counters(frame)
        return frame

    def _display_angles(self, frame, right_angle, left_angle):
        """Display angles on the frame."""
        if right_angle:
            cv2.putText(frame, f"Right angle: {right_angle:.1f}°", (10, 190),
                        self.font, self.font_scale, CONFIG["colors"]["text_color"], self.font_thickness, cv2.LINE_AA)
        if left_angle:
            cv2.putText(frame, f"Left angle: {left_angle:.1f}°", (10, 220),
                        self.font, self.font_scale, CONFIG["colors"]["text_color"], self.font_thickness, cv2.LINE_AA)

    def _display_counters(self, frame):
        """Display rep counters and stages."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (300, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"{self.exercise_name.upper()} COUNTER", (10, 30),
                    self.font, self.font_scale, CONFIG["colors"]["text_color"], self.font_thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Right Count: {self.right_count} reps", (10, 70),
                    self.font, self.font_scale, CONFIG["colors"]["text_color"], self.font_thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Left Count: {self.left_count} reps", (10, 100),
                    self.font, self.font_scale, CONFIG["colors"]["text_color"], self.font_thickness, cv2.LINE_AA)

    def reset(self):
        """Reset counters and stages."""
        self.left_count = 0
        self.right_count = 0
        self.left_stage = "down"
        self.right_stage = "down"
        self.left_stable_frames = 0
        self.right_stable_frames = 0

    def release(self):
        """Release resources."""
        self.pose.close()


class CurlCounter(ExerciseTracker):
    def __init__(self):
        super().__init__("bicep_curl")

    def detect_movement(self, landmarks, side):
        """Detect bicep curl movement for a specific side."""
        if side == "right":
            shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        else:
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        angle = self.calculate_angle(shoulder, elbow, wrist)
        if angle > self.down_angle:
            return angle, "down"
        elif angle < self.up_angle:
            return angle, "up"
        else:
            return angle, None


class FitnessApp:
    def __init__(self):
        self.trackers = {
            "Bicep Curls": CurlCounter(),
            # Add other trackers here (e.g., Squats, Situps)
        }
        self.active_tracker_name = "Bicep Curls"
        self.active_tracker = self.trackers[self.active_tracker_name]

    def process_frame(self, frame):
        return self.active_tracker.process_frame(frame)

    def reset(self):
        self.active_tracker.reset()

    def release(self):
        for tracker in self.trackers.values():
            tracker.release()


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    app = FitnessApp()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = app.process_frame(frame)
        cv2.imshow("Fitness Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            app.reset()

    app.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()