
# Hand Tracker And Gesture Controller

This project allows you to control your computer's cursor and perform mouse actions using hand gestures detected by your webcam.

This README will guide you through setting up a virtual environment, installing all the necessary dependencies, calibrating the system to generate the `calibration_data.json` file, and running the tracker application to start controlling your computer with hand gestures.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Set Up a Virtual Environment](#2-set-up-a-virtual-environment)
  - [3. Activate the Virtual Environment](#3-activate-the-virtual-environment)
  - [4. Install Dependencies](#4-install-dependencies)
- [Calibration](#calibration)
  - [Run the Calibration Script](#run-the-calibration-script)
- [Running the Tracker Application](#running-the-tracker-application)
- [Video Tutorial](#video-tutorial)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.7 or higher**
- **pip** (Python package manager)
- **Virtualenv** (optional but recommended)
- **Webcam** (built-in or external)

## Setup Instructions

### 1. Clone the Repository

First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/yourusername/handTracker.git
```

Alternatively, you can download the ZIP file from the repository and extract it.

### 2. Set Up a Virtual Environment

It's recommended to use a virtual environment to manage your project's dependencies without affecting your global Python installation.

Navigate to the project directory:

```bash
cd handTracker
```

Create a virtual environment named `venv`:

- **On Windows:**

  ```bash
  python -m venv venv
  ```

- **On macOS/Linux:**

  ```bash
  python3 -m venv venv
  ```

### 3. Activate the Virtual Environment

Activate the virtual environment you just created.

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **On macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

You should now see `(venv)` at the beginning of your command prompt, indicating that the virtual environment is active.

### 4. Install Dependencies

Install all the necessary Python packages using `pip` and the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Calibration

Before using the hand gesture controller, you need to calibrate the system to recognize your hand gestures accurately.

### Run the Calibration Script

Run the calibration script to generate the `calibration_data.json` file:

```bash
python calibrate.py
```

**Calibration Steps:**

1. **Screen Dimensions Calibration:**

   - Enter your screen's width and height in pixels when prompted.
   - If unsure, the script will use your default screen size.

2. **Click Distance Calibration:**

   - For each finger (index, middle, ring, pinky), you will calibrate the click distance.
   - **Instructions:**
     - Read the instructions displayed on the screen.
     - Touch your thumb and the specified finger together as if clicking.
     - Press the **spacebar** to start.
     - Press **'s'** to start collecting data and **'e'** to end data collection.
     - Repeat as necessary for accurate calibration.

3. **Click Duration Calibration:**

   - For each finger, you will calibrate the duration you hold the click gesture.
   - **Instructions:**
     - Hold your thumb and the specified finger together for the duration you prefer for clicks.
     - Press the **spacebar** to start.
     - Press **'s'** to start collecting data and **'e'** to end data collection.

4. **Movement Mapping Calibration:**

   - This step maps your hand movements to your screen coordinates.
   - **Instructions:**
     - Move your ring finger to each corner of the screen when prompted.
     - Press the **spacebar** to start.

5. **Smoothing Factor Calibration:**

   - Determines the smoothness of cursor movement.
   - **Instructions:**
     - Follow the moving circle with your ring finger.
     - Press the **spacebar** to start.
     - The calibration will run automatically for about 10 seconds.

6. **Scroll Sensitivity Calibration:**

   - Calibrates the sensitivity of the scrolling gesture.
   - **Instructions:**
     - Perform scrolling gestures by moving your thumb and ring finger together when prompted.
     - Press the **spacebar** to start.
     - Perform the gesture five times as instructed.

After completing all the steps, the calibration data will be saved to `calibration_data.json` in the project directory.

---

## Running the Tracker Application

With the calibration complete, you can now run the tracker application to start controlling your computer with hand gestures.

```bash
python tracker.py
```

**Usage Instructions:**

- **Cursor Movement:**

  - Extend your **ring finger** and move it to control the cursor.
  - Ensure other fingers are not in any clicking gestures.

- **Left Click:**

  - Touch your **thumb and index finger** together briefly.
  - Hold longer for a double-click (if implemented).

- **Right Click:**

  - Touch your **thumb and middle finger** together briefly.

- **Scrolling:**

  - Touch your **thumb and ring finger** together.
  - Move your ring finger up and down to scroll.

- **Click and Drag:**

  - Touch your **thumb and pinky finger** together to initiate drag.
  - Move your ring finger to drag the cursor.
  - Separate your thumb and pinky to release.

- **Exit the Application:**

  - Press **'q'** or close the window to exit the application.

---

## Video Tutorial

For a detailed step-by-step guide on setting up and using the Hand Gesture Controller, please refer to the video tutorial:

[Hand Gesture Controller Video Tutorial]([https://www.example.com/your-video-link](https://www.linkedin.com/feed/update/urn:li:activity:7256885068663513091/))

---

## Troubleshooting

- **Webcam Not Detected:**

  - Ensure your webcam is connected and not being used by another application.
  - Check the `VideoCapture` index in the code (`cv2.VideoCapture(0)`). Try changing `0` to `1` if you have multiple cameras.

- **Dependencies Not Installing:**

  - Make sure you're in the virtual environment when installing dependencies.
  - Update `pip` using `pip install --upgrade pip`.

- **Calibration Issues:**

  - Ensure you're following the on-screen instructions carefully.
  - Maintain consistent lighting and minimize background distractions during calibration.

- **Application Not Responding:**

  - Check the console for any error messages.
  - Ensure all dependencies are correctly installed.
  - Try rerunning the calibration script.

---

## License

This project is licensed under the Apache-2.0 license - see the [LICENSE](LICENSE) file for details.

---

**Enjoy controlling your computer with gestures! If you have any questions or need further assistance, feel free to reach out.**
