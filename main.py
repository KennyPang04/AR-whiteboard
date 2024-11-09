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
x_coords, y_coords = [], []  # To store recent positions for smoothing
smooth_factor = 5  # Number of points to average for smoothing

# Interpolation function to create smoother lines between points
def interpolate_line(x0, y0, x1, y1, num_points=10):
    x_vals = np.linspace(x0, x1, num_points)
    y_vals = np.linspace(y0, y1, num_points)
    return list(zip(x_vals, y_vals))

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
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

                # Get the landmarks of the index finger tip and the thumb finger tip
                index_finger_tip = hand_landmarks.landmark[8]
                thumb_finger_tip = hand_landmarks.landmark[4]

                # Convert normalized coordinates to pixel coordinates
                h, w, _ = img.shape
                index_x, index_y = int(w - (index_finger_tip.x * w)), int(index_finger_tip.y * h)
                thumb_x, thumb_y = int(w - (thumb_finger_tip.x * w)), int(thumb_finger_tip.y * h)

                EPSILON = 30  # Threshold for detecting if the fingers are touching
                distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

                if distance <= EPSILON:
                    # Add the current position to the list for smoothing
                    x_coords.append(index_x)
                    y_coords.append(index_y)

                    # Smooth the path by averaging recent positions
                    if len(x_coords) > smooth_factor:
                        x_coords.pop(0)
                        y_coords.pop(0)
                    avg_x = int(np.mean(x_coords))
                    avg_y = int(np.mean(y_coords))

                    # Draw line if there is a previous point
                    if last_x is not None and last_y is not None:
                        points = interpolate_line(last_x, last_y, avg_x, avg_y, num_points=10)
                        for i in range(1, len(points)):
                            cv2.line(whiteboard, (int(points[i-1][0]), int(points[i-1][1])), 
                                     (int(points[i][0]), int(points[i][1])), (0, 0, 0), 3)

                    # Update last position
                    last_x, last_y = avg_x, avg_y

                    # Draw a small circle at the fingertip position for visualization
                    cv2.circle(whiteboard, (avg_x, avg_y), 3, (0, 0, 0), -1)

        else:
            # Reset if no hand is detected
            last_x, last_y = None, None
            x_coords, y_coords = [], []

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