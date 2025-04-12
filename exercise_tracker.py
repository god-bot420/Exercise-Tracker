import cv2
import numpy as np
import mediapipe as mp
from abc import ABC, abstractmethod


class ExerciseTracker(ABC):
    def __init__(self, exercise_name, down_angle, up_angle):
        self.exercise_name = exercise_name
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.down_angle = down_angle
        self.up_angle = up_angle
        self.left_count = 0
        self.right_count = 0
        self.left_stage = "down"
        self.right_stage = "down"

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    @abstractmethod
    def detect_movement(self, landmarks, side):
        pass

    def count_reps(self, landmarks):
        right_angle, right_position = self.detect_movement(landmarks, "right")
        left_angle, left_position = self.detect_movement(landmarks, "left")
        if right_position == "down" and self.right_stage == "up":
            self.right_count += 1
            self.right_stage = "down"
        elif right_position == "up" and self.right_stage == "down":
            self.right_stage = "up"
        if left_position == "down" and self.left_stage == "up":
            self.left_count += 1
            self.left_stage = "down"
        elif left_position == "up" and self.left_stage == "down":
            self.left_stage = "up"
        return right_angle, left_angle

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame,results.pose_landmarks,self.mp_pose.POSE_CONNECTIONS)
            right_angle, left_angle = self.count_reps(results.pose_landmarks.landmark)
            self.display_info(frame, right_angle, left_angle)
        return frame

    def display_info(self, frame, right_angle, left_angle):
        cv2.putText(frame, f"{self.exercise_name.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Right Count: {self.right_count}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Left Count: {self.left_count}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if right_angle:
            cv2.putText(frame, f"Right Angle: {int(right_angle)}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if left_angle:
            cv2.putText(frame, f"Left Angle: {int(left_angle)}",
                        (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    def reset(self):
        self.left_count = 0
        self.right_count = 0
        self.left_stage = "down"
        self.right_stage = "down"

    def release(self):
        self.pose.close()