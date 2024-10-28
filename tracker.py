import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import json

# Ensure PyAutoGUI matches screen resolution
print(f"Screen size: {pyautogui.size()}")
pyautogui.FAILSAFE = False

class HandGestureController:
    def __init__(self):
        # Load calibration parameters
        try:
            with open('calibration_data.json', 'r') as f:
                params = json.load(f)
                self.left_click_distance = params.get('left_click_distance', 20.0)
                self.right_click_distance = params.get('right_click_distance', 20.0)
                self.scroll_click_distance = params.get('scroll_click_distance', 20.0)
                self.drag_click_distance = params.get('drag_click_distance', 20.0)
                self.scroll_sensitivity = params.get('scroll_sensitivity', 1.0)
                self.screen_width = params.get('screen_width', pyautogui.size().width)
                self.screen_height = params.get('screen_height', pyautogui.size().height)
                self.smoothing_factor = params.get('smoothing_factor', 0.4)
                self.double_click_threshold = params.get('left_click_duration', 1.0)  # Duration for double click

                # Load movement mapping parameters
                self.min_hand_x = params.get('min_hand_x', 0.0)
                self.max_hand_x = params.get('max_hand_x', 1.0)
                self.min_hand_y = params.get('min_hand_y', 0.0)
                self.max_hand_y = params.get('max_hand_y', 1.0)
        except Exception as e:
            print("An error occurred while loading calibration data: ", str(e))
            print("Using default parameters.")
            self.left_click_distance = 20.0
            self.right_click_distance = 20.0
            self.scroll_click_distance = 20.0
            self.drag_click_distance = 20.0
            self.scroll_sensitivity = 1.0
            self.screen_width = pyautogui.size().width
            self.screen_height = pyautogui.size().height
            self.smoothing_factor = 0.4
            self.double_click_threshold = 1.0
            self.min_hand_x = 0.0
            self.max_hand_x = 1.0
            self.min_hand_y = 0.0
            self.max_hand_y = 1.0

        # Initialize other variables
        self.mp_hands = mp.solutions.hands
        self.left_click_in_progress = False
        self.left_click_start_time = None
        self.right_click_in_progress = False
        self.scrolling = False
        self.drag_in_progress = False
        self.cursor_x, self.cursor_y = pyautogui.position()
        self.last_position = None

    def process_frame(self, frame, landmarker):
        thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip = self.get_finger_tips(frame, landmarker)
        if thumb_tip and index_tip and middle_tip and ring_tip and pinky_tip:
            self.handle_gestures(thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip, frame)
            self.draw_hand_landmarks(frame, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip)

    def get_finger_tips(self, frame, landmarker):
        results = landmarker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
            return thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip
        return None, None, None, None, None

    def handle_gestures(self, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip, frame):
        frame_height, frame_width = frame.shape[:2]
        thumb_coords = (thumb_tip.x * frame_width, thumb_tip.y * frame_height)
        index_coords = (index_tip.x * frame_width, index_tip.y * frame_height)
        middle_coords = (middle_tip.x * frame_width, middle_tip.y * frame_height)
        ring_coords = (ring_tip.x * frame_width, ring_tip.y * frame_height)
        pinky_coords = (pinky_tip.x * frame_width, pinky_tip.y * frame_height)

        # Calculate distances between thumb and other fingertips
        index_distance = np.hypot(thumb_coords[0] - index_coords[0],
                                  thumb_coords[1] - index_coords[1])
        middle_distance = np.hypot(thumb_coords[0] - middle_coords[0],
                                   thumb_coords[1] - middle_coords[1])
        ring_distance = np.hypot(thumb_coords[0] - ring_coords[0],
                                 thumb_coords[1] - ring_coords[1])
        pinky_distance = np.hypot(thumb_coords[0] - pinky_coords[0],
                                  thumb_coords[1] - pinky_coords[1])

        # Dragging Logic (Thumb and Pinky Finger)
        if pinky_distance < self.drag_click_distance + 10:
            if not self.drag_in_progress:
                self.drag_in_progress = True
                print("Drag started.")
                pyautogui.mouseDown()
        else:
            if self.drag_in_progress:
                self.drag_in_progress = False
                print("Drag ended.")
                pyautogui.mouseUp()

        # Left Click Logic (Thumb and Index Finger)
        if index_distance < self.left_click_distance and \
           middle_distance > self.right_click_distance and \
           ring_distance > self.scroll_click_distance and \
           pinky_distance > self.drag_click_distance and \
           not self.drag_in_progress:
            if not self.left_click_in_progress:
                self.left_click_in_progress = True
                self.left_click_start_time = time.time()
                print("Left click gesture detected.")
        else:
            if self.left_click_in_progress:
                click_duration = time.time() - self.left_click_start_time
                if click_duration >= self.double_click_threshold:
                    print("Double click performed.")
                    pyautogui.doubleClick()
                else:
                    print("Left click performed.")
                    pyautogui.click()
                self.left_click_in_progress = False
                self.left_click_start_time = None

        # Right Click Logic (Thumb and Middle Finger)
        if middle_distance < self.right_click_distance and \
           index_distance > self.left_click_distance and \
           ring_distance > self.scroll_click_distance and \
           pinky_distance > self.drag_click_distance and \
           not self.drag_in_progress:
            if not self.right_click_in_progress:
                self.right_click_in_progress = True
                print("Right click gesture detected.")
        else:
            if self.right_click_in_progress:
                print("Right click performed.")
                pyautogui.rightClick()
                self.right_click_in_progress = False

        # Scrolling Logic (Thumb and Ring Finger)
        if ring_distance < self.scroll_click_distance and \
           pinky_distance > self.drag_click_distance and \
           not self.drag_in_progress:
            if not self.scrolling:
                self.scrolling = True
                self.last_position = ring_coords
                print("Scrolling activated.")
            else:
                movement = self.last_position[1] - ring_coords[1]
                pyautogui.scroll(int(movement * self.scroll_sensitivity))
                self.last_position = ring_coords
        else:
            if self.scrolling:
                print("Scrolling deactivated.")
                self.scrolling = False
                self.last_position = None

        # Cursor Movement (Controlled by Ring Finger)
        if ring_distance > self.scroll_click_distance and \
           (self.drag_in_progress or \
            (index_distance > (self.left_click_distance * 3) and \
             middle_distance > (self.right_click_distance * 3) and \
             pinky_distance > (self.drag_click_distance * 3))):
            self.move_cursor(ring_tip)
        else:
            # Optional: You can add code here to indicate that cursor movement is paused
            pass

    def move_cursor(self, finger_tip):
        # Normalize hand coordinates
        normalized_x = (finger_tip.x - self.min_hand_x) / (self.max_hand_x - self.min_hand_x)
        normalized_y = (finger_tip.y - self.min_hand_y) / (self.max_hand_y - self.min_hand_y)

        # Invert x-axis to correct for frame flipping
        normalized_x = 1.0 - normalized_x

        # Map to screen coordinates
        cursor_x = normalized_x * self.screen_width
        cursor_y = normalized_y * self.screen_height

        # Apply smoothing
        self.cursor_x = self.cursor_x * (1 - self.smoothing_factor) + cursor_x * self.smoothing_factor
        self.cursor_y = self.cursor_y * (1 - self.smoothing_factor) + cursor_y * self.smoothing_factor

        # Ensure the cursor stays within screen bounds
        self.cursor_x = np.clip(self.cursor_x, 0, self.screen_width - 1)
        self.cursor_y = np.clip(self.cursor_y, 0, self.screen_height - 1)

        pyautogui.moveTo(self.cursor_x, self.cursor_y)

    def draw_hand_landmarks(self, frame, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip):
        frame_height, frame_width = frame.shape[:2]
        thumb_coords = (int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height))
        index_coords = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))
        middle_coords = (int(middle_tip.x * frame_width), int(middle_tip.y * frame_height))
        ring_coords = (int(ring_tip.x * frame_width), int(ring_tip.y * frame_height))
        pinky_coords = (int(pinky_tip.x * frame_width), int(pinky_tip.y * frame_height))

        # Draw circles on tips
        cv2.circle(frame, thumb_coords, 8, (0, 255, 0), -1)    # Green
        cv2.circle(frame, index_coords, 8, (0, 255, 0), -1)    # Green
        cv2.circle(frame, middle_coords, 8, (0, 255, 255), -1) # Yellow
        cv2.circle(frame, ring_coords, 8, (0, 0, 255), -1)     # Red
        cv2.circle(frame, pinky_coords, 8, (255, 0, 255), -1)  # Magenta

        # Draw lines between thumb and other fingers
        cv2.line(frame, thumb_coords, index_coords, (255, 0, 0), 2)    # Blue line for index
        cv2.line(frame, thumb_coords, middle_coords, (0, 255, 255), 2) # Yellow line for middle
        cv2.line(frame, thumb_coords, ring_coords, (0, 0, 255), 2)     # Red line for ring
        cv2.line(frame, thumb_coords, pinky_coords, (255, 0, 255), 2)  # Magenta line for pinky

        # Display distances
        index_distance = int(np.hypot(thumb_coords[0] - index_coords[0],
                                      thumb_coords[1] - index_coords[1]))
        middle_distance = int(np.hypot(thumb_coords[0] - middle_coords[0],
                                       thumb_coords[1] - middle_coords[1]))
        ring_distance = int(np.hypot(thumb_coords[0] - ring_coords[0],
                                     thumb_coords[1] - ring_coords[1]))
        pinky_distance = int(np.hypot(thumb_coords[0] - pinky_coords[0],
                                      thumb_coords[1] - pinky_coords[1]))

        cv2.putText(frame, f"Index Dist: {index_distance}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Middle Dist: {middle_distance}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Ring Dist: {ring_distance}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pinky Dist: {pinky_distance}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Usage
if __name__ == "__main__":
    controller = HandGestureController()

    cap = cv2.VideoCapture(0)
    try:
        with controller.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                controller.process_frame(frame, landmarker)

                cv2.imshow('Hand Gesture Controller', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print("An error occurred during hand gesture control:")
        print(str(e))
    finally:
        cap.release()
        cv2.destroyAllWindows()
