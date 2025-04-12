from exercise_tracker import ExerciseTracker


class SquatCounter(ExerciseTracker):
    def __init__(self):
        super().__init__("Squats", down_angle=110, up_angle=160)

    def detect_movement(self, landmarks, side):
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]
        ankle = [landmarks[27].x, landmarks[27].y]

        angle = self.calculate_angle(hip, knee, ankle)
        if angle > self.up_angle:
            return angle, "up"
        elif angle < self.down_angle:
            return angle, "down"
        else:
            return angle, None