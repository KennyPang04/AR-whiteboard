import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
whiteboard = np.ones((500, 500, 3), dtype=np.uint8) * 255

# Store the last position of the index finger
last_x, last_y = None, None

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Loop through landmarks to find the index finger (ID 8)
                for id, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _ = img.shape
                    # Adjust the x-coordinate to invert it
                    cx, cy = int(w - (landmark.x * w)), int(landmark.y * h)
                    
                    # Check if the landmark is the tip of the index finger
                    if id == 8:
                        # If there's a previous point, draw a line to make drawing smoother
                        if last_x is not None and last_y is not None:
                            cv2.line(whiteboard, (last_x, last_y), (cx, cy), (0, 0, 0), 3)
                        # Update the last position
                        last_x, last_y = cx, cy
                        
                        # Draw a circle at the current position for better visualization
                        cv2.circle(whiteboard, (cx, cy), 3, (0, 0, 0), -1)

        else:
            # Reset the last position if no hand is detected
            last_x, last_y = None, None

        # Display the camera feed and the whiteboard
        cv2.imshow("Hand Tracking", img)
        cv2.imshow("Whiteboard", whiteboard)

        # Check for key press to clear the board
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('c'):  # Press 'c' to clear the board
            whiteboard = np.ones((500, 500, 3), dtype=np.uint8) * 255  # Reinitialize the whiteboard

cap.release()
cv2.destroyAllWindows()
