from exercise_tracker import ExerciseTracker


class CurlCounter(ExerciseTracker):
    def __init__(self):
        super().__init__("Bicep Curls", down_angle=150, up_angle=70)

    def detect_movement(self, landmarks, side):
        if side == "right":
            shoulder = [landmarks[11].x, landmarks[11].y]
            elbow = [landmarks[13].x, landmarks[13].y]
            wrist = [landmarks[15].x, landmarks[15].y]
        else:
            shoulder = [landmarks[12].x, landmarks[12].y]
            elbow = [landmarks[14].x, landmarks[14].y]
            wrist = [landmarks[16].x, landmarks[16].y]

        angle = self.calculate_angle(shoulder, elbow, wrist)
        if angle > self.down_angle:
            return angle, "down"
        elif angle < self.up_angle:
            return angle, "up"
        else:
            return angle, None