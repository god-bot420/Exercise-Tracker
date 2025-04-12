import cv2
from curl import CurlCounter
from situp import SitupCounter
from squat import SquatCounter


class FitnessApp:
    def __init__(self):
        self.trackers = {
            "Bicep Curls": CurlCounter(),
            "Situps": SitupCounter(),
            "Squats": SquatCounter()
        }
        self.active_tracker_name = "Bicep Curls"
        self.active_tracker = self.trackers[self.active_tracker_name]
        self.dropdown_open = False  # Tracks whether the dropdown is open

    def switch_mode(self, mode):
        if mode in self.trackers:
            self.active_tracker_name = mode
            self.active_tracker = self.trackers[mode]

    def process_frame(self, frame):
        return self.active_tracker.process_frame(frame)

    def reset(self):
        self.active_tracker.reset()

    def release(self):
        for tracker in self.trackers.values():
            tracker.release()


def draw_dropdown(frame, modes, active_mode, dropdown_open):
    """Draw the dropdown menu on the frame."""
    # Dropdown button (top-right corner)
    frame_height, frame_width, _ = frame.shape
    dropdown_x_start = frame_width - 210
    dropdown_x_end = frame_width - 10
    dropdown_y_start = 10
    dropdown_y_end = 50

    # Draw dropdown button
    cv2.rectangle(frame, (dropdown_x_start, dropdown_y_start), (dropdown_x_end, dropdown_y_end), (0, 0, 0), -1)
    cv2.putText(frame, f"Mode: {active_mode}", (dropdown_x_start + 10, dropdown_y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # If dropdown is open, display all modes
    if dropdown_open:
        for i, mode in enumerate(modes):
            y_start = dropdown_y_end + i * 40
            y_end = y_start + 40
            cv2.rectangle(frame, (dropdown_x_start, y_start), (dropdown_x_end, y_end), (50, 50, 50), -1)
            cv2.putText(frame, mode, (dropdown_x_start + 10, y_start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def check_dropdown_click(x, y, modes, dropdown_open, frame_width):
    """Check if a dropdown item was clicked."""
    dropdown_x_start = frame_width - 210
    dropdown_x_end = frame_width - 10
    dropdown_y_start = 10
    dropdown_y_end = 50

    if dropdown_x_start <= x <= dropdown_x_end:  # Within dropdown width
        if dropdown_y_start <= y <= dropdown_y_end:  # Dropdown button
            return "toggle"
        if dropdown_open:
            for i, mode in enumerate(modes):
                y_start = dropdown_y_end + i * 40
                y_end = y_start + 40
                if y_start <= y <= y_end:
                    return mode
    return None


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    app = FitnessApp()
    modes = list(app.trackers.keys())
    dropdown_open = False
    window_closed = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal dropdown_open
        if event == cv2.EVENT_LBUTTONDOWN:
            frame_width = param["frame_width"]
            result = check_dropdown_click(x, y, modes, dropdown_open, frame_width)
            if result == "toggle":
                dropdown_open = not dropdown_open
            elif result in modes:
                app.switch_mode(result)
                dropdown_open = False

    cv2.namedWindow("Fitness Tracker")
    cv2.setMouseCallback("Fitness Tracker", mouse_callback, param={"frame_width": 640})

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Update frame width for dropdown click detection
        frame_height, frame_width, _ = frame.shape
        cv2.setMouseCallback("Fitness Tracker", mouse_callback, param={"frame_width": frame_width})

        # Draw dropdown menu
        draw_dropdown(frame, modes, app.active_tracker_name, dropdown_open)

        # Process frame
        frame = app.process_frame(frame)
        cv2.imshow("Fitness Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            app.reset()

        # Check if the window is closed
        if cv2.getWindowProperty("Fitness Tracker", cv2.WND_PROP_VISIBLE) < 1:
            window_closed = True
            break

    app.release()
    cap.release()
    cv2.destroyAllWindows()

    if window_closed:
        print("Window closed by user.")


if __name__ == "__main__":
    main()