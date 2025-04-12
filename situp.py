from exercise_tracker import ExerciseTracker


class SitupCounter(ExerciseTracker):
    def __init__(self):
        super().__init__("Situps", down_angle=160, up_angle=70)

    def detect_movement(self, landmarks, side):
        """Detect situp movement."""
        shoulder = [landmarks[11].x, landmarks[11].y]
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]

        angle = self.calculate_angle(shoulder, hip, knee)
        if angle > self.down_angle:
            return angle, "down"
        elif angle < self.up_angle:
            return angle, "up"
        else:
            return angle, None