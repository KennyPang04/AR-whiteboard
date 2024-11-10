import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
whiteboard = np.ones((500, 500, 3), dtype=np.uint8) * 255

def draw_pattern(whiteboard):
    height, width, _ = whiteboard.shape
    top_left_x, top_left_y = 0, 0
    top_right_x, top_right_y = width - 25, 0  # Position for the right-side box
    box_size = 25  # Size of the pattern (e.g., 50x50 pixels)
    # Display Color
    whiteboard[top_right_y:top_right_y + box_size, top_right_x:top_right_x + box_size] = COLOR
    
    # Display Eraser
    whiteboard[top_left_y:top_left_y + box_size, top_left_x:top_left_x + box_size] = Colors["WHITE"]
    if ERASER:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2 
        text_x = top_left_x + 5
        text_y = top_left_y + 20
        cv2.putText(whiteboard, 'E', (text_x, text_y), font, font_scale, Colors["BLACK"], thickness)
    

# Interpolation function to create smoother lines between points
def interpolate_line(x0, y0, x1, y1, num_points=10):
    x_vals = np.linspace(x0, x1, num_points)
    y_vals = np.linspace(y0, y1, num_points)
    return list(zip(x_vals, y_vals))

def distance(landmark1, landmark2, img):
    h, w, _ = img.shape
    l1x, l1y = int(w - (landmark1.x * w)), int(landmark1.y * h)
    l2x, l2y = int(w - (landmark2.x * w)), int(landmark2.y * h)
    return np.sqrt((l1x - l2x) ** 2 + (l1y - l2y) ** 2)

def get_next_color(current_color):
    current_index = Colors_list.index(current_color)
    # Loop to find the next non-white color
    while True:
        current_index = (current_index + 1) % len(Colors_list)
        next_color_name = Colors_list[current_index]
        if next_color_name != "WHITE":
            return next_color_name

Colors = {
    "BLACK": (0, 0, 0),
    "WHITE": (255, 255, 255),
    "RED": (0, 0, 255),
    "ORANGE": (0, 127, 255),
    "YELLOW": (0, 255, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
}

Colors_list = ["BLACK",
    "WHITE",
    "RED",
    "ORANGE",
    "YELLOW",
    "GREEN",
    "BLUE"
]

#initialize Varibles
COLOR = Colors["BLACK"]
color_word = "BLACK"
THICKNESS = 3
last_peace_sign_time = 0
last_thumb_up_time = 0
lockout_time = 2
ERASER = False
last_x, last_y = None, None
x_coords, y_coords = [], []  # To store recent positions for smoothing
smooth_factor = 5  # Number of points to average for smoothing
draw_pattern(whiteboard)

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

                # Extract landmarks for fingertips and wrist
                wrist =  hand_landmarks.landmark[0]
                thumb_tip = hand_landmarks.landmark[4]
                thumb_ip = hand_landmarks.landmark[3]
                index_tip = hand_landmarks.landmark[8]
                index_knuckle = hand_landmarks.landmark[6]
                middle_tip = hand_landmarks.landmark[12]
                middle_knuckle = hand_landmarks.landmark[10]
                ring_tip = hand_landmarks.landmark[16]
                ring_knuckle = hand_landmarks.landmark[14]
                pinky_tip = hand_landmarks.landmark[20]
                pinky_knuckle = hand_landmarks.landmark[18]

                if (thumb_tip.y < thumb_ip.y and            
                        pinky_tip.y < pinky_knuckle.y and    
                        index_tip.y > index_knuckle.y and    
                        middle_tip.y > middle_knuckle.y and  
                        ring_tip.y > ring_knuckle.y):
                    current_time = time.time()
                    if current_time - last_thumb_up_time >= lockout_time and not ERASER:
                        print("Changing Color")
                        next_color = get_next_color(color_word)
                        COLOR = Colors[next_color]
                        color_word = next_color
                        last_thumb_up_time = current_time
                        draw_pattern(whiteboard)

                # Detecting Peace Sign to enter Eraser 
                if (index_tip.y < index_knuckle.y and middle_tip.y < middle_knuckle.y and
                        ring_tip.y > ring_knuckle.y and pinky_tip.y > pinky_knuckle.y):
                    current_time = time.time()
                    if current_time - last_peace_sign_time >= lockout_time and not ERASER:
                        print("ENTERING eraser mode")
                        COLOR, THICKNESS, last_peace_sign_time,ERASER,color_word = Colors["WHITE"],20,current_time,True, "WHITE"
                    if current_time - last_peace_sign_time >= lockout_time and ERASER:
                        print("LEAVING eraser mode")
                        COLOR, THICKNESS, last_peace_sign_time,ERASER,color_word = Colors["BLACK"],3,current_time,False, "BLACK"
                    draw_pattern(whiteboard)

                # Drawing condition (index and thumb touching)
                EPSILON = 23  # Threshold for detecting if the fingers are touching
                if distance(index_tip, thumb_tip, img) <= EPSILON:
                    index_x, index_y = int(img.shape[1] - (index_tip.x * img.shape[1])), int(index_tip.y * img.shape[0])
                    # Add the current position to the list for smoothing
                    x_coords.append(index_x)
                    y_coords.append(index_y)
                    # Smooth the path by averaging recent positions
                    if len(x_coords) > smooth_factor:
                        x_coords.pop(0)
                        y_coords.pop(0)
                    avg_x,avg_y = int(np.mean(x_coords)),int(np.mean(y_coords))
                    # Draw line to smoothen drawing
                    if last_x is not None and last_y is not None:
                        points = interpolate_line(last_x, last_y, avg_x, avg_y, num_points=10)
                        for i in range(1, len(points)):
                            cv2.line(whiteboard, (int(points[i-1][0]), int(points[i-1][1])), 
                                     (int(points[i][0]), int(points[i][1])), COLOR, THICKNESS)
                    # Update last position
                    last_x, last_y = avg_x, avg_y
                    # Drawing
                    cv2.circle(whiteboard, (avg_x, avg_y), THICKNESS, COLOR, -1)
                else:
                    # Reset the drawing points if no hand is detected or fingers are not touching
                    last_x, last_y = None, None
                    x_coords, y_coords = [], []
        else:
            # If no hand is detected, reset the last position
            last_x, last_y = None, None
            x_coords, y_coords = [], []
        
        # Display the camera feed and the whiteboard
        cv2.imshow("Hand Tracking", img)
        cv2.imshow("Whiteboard", whiteboard)

        # QUit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
        # Clear
        elif key == ord('c'):
            whiteboard = np.ones((500, 500, 3), dtype=np.uint8) * 255  # Reinitialize the whiteboard

cap.release()
cv2.destroyAllWindows()
