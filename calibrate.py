import cv2
import mediapipe as mp
import time
import json
import numpy as np
import pyautogui
import sys

class HandGestureCalibrator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.parameters = {}
        self.capture = cv2.VideoCapture(0)

        # Check if the webcam is opened correctly
        if not self.capture.isOpened():
            raise IOError("Cannot open webcam")

        self.min_detection_confidence = 0.7
        self.min_tracking_confidence = 0.7

    def calibrate(self):
        try:
            with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            ) as hands:
                # Step 1: Calibrate screen dimensions
                self.calibrate_screen_dimensions()
                # Step 2: Calibrate click distances and durations
                self.calibrate_click_parameters(hands)
                # Step 3: Calibrate movement mapping and amplification
                self.calibrate_movement_mapping(hands)
                # Step 4: Calibrate smoothing factor
                self.calibrate_smoothing_factor(hands)
                # Step 5: Calibrate scroll sensitivity
                self.calibrate_scroll_sensitivity(hands)

                self.save_parameters()
        except Exception as e:
            print("An error occurred during calibration: ", str(e))
        finally:
            self.capture.release()
            cv2.destroyAllWindows()

    def show_pre_calibration_screen(self, title, instructions, window_name='Calibration'):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape

            # Blur the frame
            blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)

            # Split instructions into multiple lines if necessary
            instruction_lines = instructions.split('\n')
            y_start = 60  # Starting Y position for the first line
            y_gap = 30    # Gap between lines

            # Display title
            cv2.putText(
                blurred_frame,
                title,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )

            # Display instructions line by line
            for i, line in enumerate(instruction_lines):
                y_position = y_start + i * y_gap
                cv2.putText(
                    blurred_frame,
                    line,
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            cv2.putText(
                blurred_frame,
                "Press SPACEBAR to start or 'q' to quit.",
                (10, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow(window_name, blurred_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Spacebar pressed
                break
            elif key == ord('q'):
                sys.exit(0)
        cv2.destroyWindow(window_name)

    def calibrate_screen_dimensions(self):
        print("\nCalibration Step 1: Screen Dimensions")
        try:
            screen_width = int(input("Enter your screen width in pixels (e.g., 1920): "))
            screen_height = int(input("Enter your screen height in pixels (e.g., 1080): "))
            self.parameters['screen_width'] = screen_width
            self.parameters['screen_height'] = screen_height
            print(f"Screen dimensions set to {screen_width}x{screen_height}")
        except ValueError:
            print(f"Invalid input for screen dimensions. Using default screen size: {pyautogui.size()}")
            self.parameters['screen_width'] = pyautogui.size().width
            self.parameters['screen_height'] = pyautogui.size().height

    def calibrate_click_parameters(self, hands):
        # Combine left and right click distance and duration calibration
        self.calibrate_click_distance(hands, finger="INDEX")
        self.calibrate_click_distance(hands, finger="MIDDLE")
        self.calibrate_click_distance(hands, finger="RING")
        self.calibrate_click_distance(hands, finger="PINKY")
        self.calibrate_click_duration(hands, finger="INDEX")
        self.calibrate_click_duration(hands, finger="MIDDLE")
        self.calibrate_click_duration(hands, finger="RING")
        self.calibrate_click_duration(hands, finger="PINKY")

    def calibrate_click_distance(self, hands, finger="INDEX"):
        finger_name = finger.lower()
        print(f"\nCalibration Step: {finger_name.capitalize()} Click Distance")
        # Show pre-calibration screen with clearer instructions
        self.show_pre_calibration_screen(
            f"{finger_name.capitalize()} Click Distance Calibration",
            f"Please touch your thumb and {finger_name} finger together\n"
            f"as if you are performing a click gesture.",
        )

        distances = []
        collecting_data = False
        min_distance_threshold = 50  # Minimum distance in pixels to start collecting data

        if finger == "INDEX":
            finger_tip_id = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
        elif finger == "MIDDLE":
            finger_tip_id = mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP
        elif finger == "RING":
            finger_tip_id = mp.solutions.hands.HandLandmark.RING_FINGER_TIP
        else:
            finger_tip_id = mp.solutions.hands.HandLandmark.PINKY_TIP

        window_name = f'Calibration - {finger_name.capitalize()} Click Distance'

        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                frame_height, frame_width, _ = frame.shape
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                instruction_text = "Press 's' to start collecting data or 'e' to finish."
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0].landmark
                    thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
                    finger_tip = landmarks[finger_tip_id]

                    thumb_coords = (int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height))
                    finger_coords = (int(finger_tip.x * frame_width), int(finger_tip.y * frame_height))
                    distance = np.hypot(thumb_coords[0] - finger_coords[0], thumb_coords[1] - finger_coords[1])

                    # Draw landmarks and line
                    cv2.circle(frame, thumb_coords, 5, (0, 255, 0), -1)
                    cv2.circle(frame, finger_coords, 5, (0, 255, 0), -1)
                    cv2.line(frame, thumb_coords, finger_coords, (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        f"Distance: {int(distance)}",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

                    if collecting_data and distance < min_distance_threshold:
                        distances.append(distance)

                # Display instructions
                cv2.putText(
                    frame,
                    f"Touch your thumb and {finger_name} finger together.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    instruction_text,
                    (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    print("Starting data collection...")
                    collecting_data = True
                elif key == ord('e'):
                    print("Ending data collection.")
                    break
                elif key == ord('q'):
                    sys.exit(0)
        except Exception as e:
            print(f"An error occurred during {finger_name} click distance calibration: ", str(e))
        finally:
            cv2.destroyWindow(window_name)

        if distances:
            distance_value = float(np.percentile(distances, 50))  # Median distance
            if finger == "INDEX":
                self.parameters['left_click_distance'] = distance_value
            elif finger == "MIDDLE":
                self.parameters['right_click_distance'] = distance_value
            elif finger == "RING":
                self.parameters['scroll_click_distance'] = distance_value
            else:
                self.parameters['drag_click_distance'] = distance_value

            print(f"Calibrated {finger_name} click distance: {distance_value}")
        else:
            print(f"No data collected for {finger_name} click distance. Using default value 20.")
            if finger == "INDEX":
                self.parameters['left_click_distance'] = 20.0
            elif finger == "MIDDLE":
                self.parameters['right_click_distance'] = 20.0
            elif finger == "RING":
                self.parameters['scroll_click_distance'] = 20.0
            else:
                self.parameters['drag_click_distance'] = 20.0

    def calibrate_click_duration(self, hands, finger="INDEX"):
        finger_name = finger.lower()
        print(f"\nCalibration Step: {finger_name.capitalize()} Click Duration")
        # Show pre-calibration screen with clearer instructions
        self.show_pre_calibration_screen(
            f"{finger_name.capitalize()} Click Duration Calibration",
            f"Hold your thumb and {finger_name} finger together\n"
            f"for the duration you wish to use for clicks.",
        )

        collecting_data = False
        start_time = None
        durations = []
        min_distance_threshold = 50  # Threshold to detect fingers are together

        if finger == "INDEX":
            finger_tip_id = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
        elif finger == "MIDDLE":
            finger_tip_id = mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP
        elif finger == "RING":
            finger_tip_id = mp.solutions.hands.HandLandmark.RING_FINGER_TIP
        else:
            finger_tip_id = mp.solutions.hands.HandLandmark.PINKY_TIP

        window_name = f'Calibration - {finger_name.capitalize()} Click Duration'

        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                frame_height, frame_width, _ = frame.shape
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                instruction_text = "Press 's' to start collecting data or 'e' to finish."
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0].landmark
                    thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
                    finger_tip = landmarks[finger_tip_id]

                    thumb_coords = (int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height))
                    finger_coords = (int(finger_tip.x * frame_width), int(finger_tip.y * frame_height))
                    distance = np.hypot(thumb_coords[0] - finger_coords[0], thumb_coords[1] - finger_coords[1])

                    # Draw landmarks and line
                    cv2.circle(frame, thumb_coords, 5, (0, 255, 0), -1)
                    cv2.circle(frame, finger_coords, 5, (0, 255, 0), -1)
                    cv2.line(frame, thumb_coords, finger_coords, (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        f"Distance: {int(distance)}",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

                    if collecting_data:
                        if distance < min_distance_threshold:
                            if start_time is None:
                                start_time = time.time()
                        else:
                            if start_time is not None:
                                duration = time.time() - start_time
                                durations.append(duration)
                                start_time = None

                # Display instructions
                cv2.putText(
                    frame,
                    f"Hold thumb and {finger_name} finger together for click duration.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    instruction_text,
                    (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    print("Starting data collection...")
                    collecting_data = True
                elif key == ord('e'):
                    print("Ending data collection.")
                    if start_time is not None:
                        duration = time.time() - start_time
                        durations.append(duration)
                    break
                elif key == ord('q'):
                    sys.exit(0)
        except Exception as e:
            print(f"An error occurred during {finger_name} click duration calibration: ", str(e))
        finally:
            cv2.destroyWindow(window_name)

        if durations:
            duration_value = float(np.percentile(durations, 50))  # Median duration
            if finger == "INDEX":
                self.parameters['left_click_duration'] = duration_value
            elif finger == "MIDDLE":
                self.parameters['right_click_duration'] = duration_value
            elif finger == "RING":
                self.parameters['scroll_click_duration'] = duration_value
            else:
                self.parameters['drag_click_duration'] = duration_value
            print(f"Calibrated {finger_name} click duration: {duration_value} seconds")
        else:
            print(f"No data collected for {finger_name} click duration. Using default value 1.0 seconds.")
            if finger == "INDEX":
                self.parameters['left_click_duration'] = 1.0
            elif finger == "MIDDLE":
                self.parameters['right_click_duration'] = 1.0
            elif finger == "RING":
                self.parameters['scroll_click_duration'] = 1.0
            else:
                self.parameters['drag_click_duration'] = 1.0

    def calibrate_movement_mapping(self, hands):
        print("\nCalibration Step: Movement Mapping and Amplification")
        # Show pre-calibration screen with clearer instructions
        self.show_pre_calibration_screen(
            "Movement Mapping Calibration",
            "Please move your ring finger to each corner\n"
            "of the screen when prompted.",
        )

        calibration_points = {
            'Top-Left': None,
            'Top-Right': None,
            'Bottom-Left': None,
            'Bottom-Right': None
        }
        point_names = list(calibration_points.keys())
        current_point_index = 0

        finger_tip_id = mp.solutions.hands.HandLandmark.RING_FINGER_TIP
        window_name = 'Calibration - Movement Mapping'

        try:
            while current_point_index < len(point_names):
                point_name = point_names[current_point_index]
                ret, frame = self.capture.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                frame_height, frame_width, _ = frame.shape

                # Draw guide circles
                if point_name == 'Top-Left':
                    cv2.circle(frame, (50, 50), 20, (0, 255, 0), 2)
                elif point_name == 'Top-Right':
                    cv2.circle(frame, (frame_width - 50, 50), 20, (0, 255, 0), 2)
                elif point_name == 'Bottom-Left':
                    cv2.circle(frame, (50, frame_height - 50), 20, (0, 255, 0), 2)
                elif point_name == 'Bottom-Right':
                    cv2.circle(frame, (frame_width - 50, frame_height - 50), 20, (0, 255, 0), 2)

                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0].landmark
                    finger_tip = landmarks[finger_tip_id]
                    finger_coords = (int(finger_tip.x * frame_width), int(finger_tip.y * frame_height))

                    # Draw finger tip
                    cv2.circle(frame, finger_coords, 5, (0, 255, 255), -1)

                    # Check if the finger tip is near the guide circle
                    guide_coords = None
                    if point_name == 'Top-Left':
                        guide_coords = (50, 50)
                    elif point_name == 'Top-Right':
                        guide_coords = (frame_width - 50, 50)
                    elif point_name == 'Bottom-Left':
                        guide_coords = (50, frame_height - 50)
                    elif point_name == 'Bottom-Right':
                        guide_coords = (frame_width - 50, frame_height - 50)

                    distance = np.hypot(finger_coords[0] - guide_coords[0], finger_coords[1] - guide_coords[1])
                    if distance < 30:
                        calibration_points[point_name] = finger_tip
                        current_point_index += 1
                        print(f"Captured {point_name} point.")
                        time.sleep(1)  # Pause briefly before moving to the next point

                # Display instructions
                cv2.putText(
                    frame,
                    f"Move ring finger to the {point_name} corner.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    sys.exit(0)
        except Exception as e:
            print("An error occurred during movement mapping calibration: ", str(e))
        finally:
            cv2.destroyWindow(window_name)

        # Calculate mapping parameters
        if None not in calibration_points.values():
            # Extract coordinates
            top_left = calibration_points['Top-Left']
            top_right = calibration_points['Top-Right']
            bottom_left = calibration_points['Bottom-Left']
            bottom_right = calibration_points['Bottom-Right']

            # Convert normalized coordinates to actual pixel values
            tl_x, tl_y = top_left.x, top_left.y
            tr_x, tr_y = top_right.x, top_right.y
            bl_x, bl_y = bottom_left.x, bottom_left.y
            br_x, br_y = bottom_right.x, bottom_right.y

            # Calculate min and max for X and Y
            min_x = min(tl_x, bl_x)
            max_x = max(tr_x, br_x)
            min_y = min(tl_y, tr_y)
            max_y = max(bl_y, br_y)

            # Save mapping parameters
            self.parameters['min_hand_x'] = min_x
            self.parameters['max_hand_x'] = max_x
            self.parameters['min_hand_y'] = min_y
            self.parameters['max_hand_y'] = max_y

            # Calculate movement amplification factors
            hand_range_x = max_x - min_x
            hand_range_y = max_y - min_y

            if hand_range_x != 0 and hand_range_y != 0:
                self.parameters['movement_amplification_x'] = self.parameters['screen_width'] / hand_range_x
                self.parameters['movement_amplification_y'] = self.parameters['screen_height'] / hand_range_y
                print("Movement mapping calibration complete.")
            else:
                print("Invalid hand movement range detected. Using default amplification factors.")
                self.parameters['movement_amplification_x'] = 5.0
                self.parameters['movement_amplification_y'] = 5.0
        else:
            print("Failed to capture all calibration points. Using default amplification factors.")
            self.parameters['movement_amplification_x'] = 5.0
            self.parameters['movement_amplification_y'] = 5.0

    def calibrate_smoothing_factor(self, hands):
        print("\nCalibration Step: Smoothing Factor")
        # Show pre-calibration screen with clearer instructions
        self.show_pre_calibration_screen(
            "Smoothing Factor Calibration",
            "Follow the moving circle with your ring finger\n"
            "to help determine cursor movement smoothness.",
        )

        positions_x = []
        positions_y = []
        finger_tip_id = mp.solutions.hands.HandLandmark.RING_FINGER_TIP
        window_name = 'Calibration - Smoothing Factor'

        try:
            start_time = time.time()
            duration = 10  # Collect data for 10 seconds
            moving_circle_radius = 20
            angle = 0
            angular_speed = 2 * np.pi / duration  # Complete one circle in 'duration' seconds

            while True:
                ret, frame = self.capture.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                frame_height, frame_width, _ = frame.shape

                # Calculate moving circle position
                center_x = frame_width // 2 + int((frame_width // 3) * np.cos(angle))
                center_y = frame_height // 2 + int((frame_height // 3) * np.sin(angle))
                angle += angular_speed / 30  # Assuming 30 FPS

                # Draw moving circle
                cv2.circle(frame, (center_x, center_y), moving_circle_radius, (0, 255, 0), 2)

                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0].landmark
                    finger_tip = landmarks[finger_tip_id]

                    finger_coords = (int(finger_tip.x * frame_width), int(finger_tip.y * frame_height))

                    # Draw landmarks
                    cv2.circle(frame, finger_coords, 5, (0, 0, 255), -1)

                    # Collect data
                    positions_x.append(finger_coords[0])
                    positions_y.append(finger_coords[1])

                # Display instructions
                cv2.putText(
                    frame,
                    "Follow the moving circle with your ring finger.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.imshow(window_name, frame)

                if (time.time() - start_time) >= duration:
                    print("Data collection complete.")
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    sys.exit(0)

        except Exception as e:
            print("An error occurred during smoothing factor calibration: ", str(e))
        finally:
            cv2.destroyWindow(window_name)

        if positions_x and positions_y:
            # Calculate frame-to-frame differences
            diffs_x = np.diff(positions_x)
            diffs_y = np.diff(positions_y)
            # Calculate standard deviation of differences
            std_dev_x = np.std(diffs_x)
            std_dev_y = np.std(diffs_y)
            avg_std_dev = (std_dev_x + std_dev_y) / 2

            # Determine smoothing factor inversely proportional to variability
            # Map avg_std_dev to a smoothing factor between 0.1 and 0.9
            max_std_dev = 15.0  # Maximum expected standard deviation
            smoothing_factor = 1.0 - min(max(avg_std_dev / max_std_dev, 0.1), 0.9)
            smoothing_factor = round(smoothing_factor, 2)

            self.parameters['smoothing_factor'] = smoothing_factor
            print(f"Calibrated smoothing factor: {self.parameters['smoothing_factor']}")
        else:
            print("No data collected for smoothing factor. Using default value 0.4.")
            self.parameters['smoothing_factor'] = 0.4

    def calibrate_scroll_sensitivity(self, hands):
        print("\nCalibration Step: Scroll Sensitivity")
        # Show pre-calibration screen with clearer instructions
        self.show_pre_calibration_screen(
            "Scroll Sensitivity Calibration",
            "Perform scrolling gestures (move thumb and ring finger together)\n"
            "when prompted to determine scroll speed.",
        )

        movements = []
        last_position = None
        finger_tip_id = mp.solutions.hands.HandLandmark.RING_FINGER_TIP
        thumb_tip_id = mp.solutions.hands.HandLandmark.THUMB_TIP

        gesture_detected = False
        gesture_start_time = None
        gesture_count = 0
        max_gestures = 5  # Number of scrolling gestures to perform
        min_distance_threshold = self.parameters.get('scroll_click_distance', 20.0)  # Use calibrated distance

        window_name = 'Calibration - Scroll Sensitivity'

        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                frame_height, frame_width, _ = frame.shape

                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0].landmark
                    thumb_tip = landmarks[thumb_tip_id]
                    finger_tip = landmarks[finger_tip_id]

                    thumb_coords = (int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height))
                    finger_coords = (int(finger_tip.x * frame_width), int(finger_tip.y * frame_height))
                    distance = np.hypot(thumb_coords[0] - finger_coords[0], thumb_coords[1] - finger_coords[1])

                    # Draw landmarks and line
                    cv2.circle(frame, thumb_coords, 5, (0, 255, 0), -1)
                    cv2.circle(frame, finger_coords, 5, (0, 255, 0), -1)
                    cv2.line(frame, thumb_coords, finger_coords, (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        f"Distance: {int(distance)}",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

                    if distance < min_distance_threshold:
                        if not gesture_detected:
                            gesture_detected = True
                            gesture_start_time = time.time()
                            last_position = finger_coords
                            print(f"Scrolling gesture {gesture_count + 1} started.")
                        else:
                            movement = last_position[1] - finger_coords[1]
                            movements.append(movement)
                            last_position = finger_coords
                    else:
                        if gesture_detected:
                            gesture_detected = False
                            gesture_count += 1
                            print(f"Scrolling gesture {gesture_count} ended.")
                            if gesture_count >= max_gestures:
                                print("Data collection complete.")
                                break

                # Display instructions
                cv2.putText(
                    frame,
                    f"Perform scrolling gesture ({gesture_count + 1}/{max_gestures}).",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    sys.exit(0)

        except Exception as e:
            print("An error occurred during scroll sensitivity calibration: ", str(e))
        finally:
            cv2.destroyWindow(window_name)

        if movements:
            avg_movement = np.mean([abs(m) for m in movements if m != 0])
            if avg_movement != 0:
                # Determine scroll sensitivity inversely proportional to average movement
                self.parameters['scroll_sensitivity'] = round(20 / avg_movement, 2)
            else:
                self.parameters['scroll_sensitivity'] = 1.0  # Default value
            print(f"Calibrated scroll sensitivity: {self.parameters['scroll_sensitivity']}")
        else:
            print("No data collected for scroll sensitivity. Using default value 1.")
            self.parameters['scroll_sensitivity'] = 1.0

    def save_parameters(self):
        try:
            with open('calibration_data.json', 'w') as f:
                json.dump(self.parameters, f, indent=4)
            print("\nCalibration complete. Parameters saved to 'calibration_data.json'.")
            print("Calibration Data:")
            print(json.dumps(self.parameters, indent=4))
        except Exception as e:
            print("An error occurred while saving calibration data: ", str(e))
            print("Calibration data was not saved.")

# Usage
if __name__ == "__main__":
    calibrator = HandGestureCalibrator()
    calibrator.calibrate()
