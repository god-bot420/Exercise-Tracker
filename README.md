# Fitness Tracker Application

The **Fitness Tracker Application** is a Python-based program that uses a webcam to track and count exercises such as bicep curls, sit-ups, and squats. It features a dropdown menu for switching between exercise modes and provides real-time feedback on the screen.

## Features

- **Real-Time Exercise Tracking**: Tracks exercises using computer vision.
- **Multiple Exercise Modes**: Supports bicep curls, sit-ups, and squats.
- **Dropdown Menu**: Allows users to switch between exercise modes via an on-screen dropdown menu.
- **Reset Functionality**: Resets the current exercise counter.
- **Webcam Integration**: Uses the webcam to process video frames.
- **User-Friendly Interface**: Displays exercise mode and count on the screen.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.7 or higher
- OpenCV (`cv2`) library
- Custom modules for exercise tracking:
  - `curl.py` (implements `CurlCounter`)
  - `situp.py` (implements `SitupCounter`)
  - `squat.py` (implements `SquatCounter`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fitness-tracker.git
   cd fitness-tracker
