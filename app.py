import cv2
import mediapipe as mp
import numpy as np
import time
from flask import Flask, Response, render_template
from flask_socketio import SocketIO
import threading

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize MediaPipe hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
whiteboard = np.ones((500, 500, 3), dtype=np.uint8) * 255

# Store the last position of the index finger
last_x, last_y = None, None
x_coords, y_coords = [], []  
smooth_factor = 5  
def interpolate_line(x0, y0, x1, y1, num_points=10):
    x_vals = np.linspace(x0, x1, num_points)
    y_vals = np.linspace(y0, y1, num_points)
    return list(zip(x_vals, y_vals))

def distance(landmark1, landmark2, img):
    h, w, _ = img.shape
    l1x, l1y = int(w - (landmark1.x * w)), int(landmark1.y * h)
    l2x, l2y = int(w - (landmark2.x * w)), int(landmark2.y * h)
    return np.sqrt((l1x - l2x) ** 2 + (l1y - l2y) ** 2)



last_peace_sign_time = 0
lockout_time = 1
ERASER = False

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_coordinates_stream():
    global last_x, last_y, x_coords, y_coords, whiteboard, THICKNESS, ERASER, last_peace_sign_time
    
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

            # Initialize coordinates in case no valid position is found
            index_x, index_y = None, None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract landmarks for fingertips and wrist
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    index_knuckle = hand_landmarks.landmark[6]
                    middle_tip = hand_landmarks.landmark[12]
                    middle_knuckle = hand_landmarks.landmark[10]
                    ring_tip = hand_landmarks.landmark[16]
                    ring_knuckle = hand_landmarks.landmark[14]
                    pinky_tip = hand_landmarks.landmark[20]
                    pinky_knuckle = hand_landmarks.landmark[18]

                    # Detect peace sign gesture
                    current_time = time.time()
                    if (index_tip.y < index_knuckle.y and middle_tip.y < middle_knuckle.y and
                            ring_tip.y > ring_knuckle.y and pinky_tip.y > pinky_knuckle.y):
                        if current_time - last_peace_sign_time >= lockout_time and not ERASER:
                            print("ENTERING eraser mode")
                            index_x = "w"
                            index_y = "w"
                            last_peace_sign_time = current_time
                            ERASER = True
                            yield f"data: {index_x},{index_y}\n\n"
                        if current_time - last_peace_sign_time >= lockout_time and ERASER:
                            print("LEAVING eraser mode")
                            index_x = "l"
                            index_y = "l"
                            last_peace_sign_time = current_time
                            ERASER = False
                            yield f"data: {index_x},{index_y}\n\n"
                    EPSILON = 23  
                    # closed fist for delete
                    if (index_tip.y > index_knuckle.y and middle_tip.y > middle_knuckle.y and ring_tip.y > ring_knuckle.y and pinky_tip.y > pinky_knuckle.y):
                        print("ENTERING delete whole canvas")
                        last_x, last_y = None, None
                        x_coords, y_coords = [], []
                        index_x = "b"
                        index_y = "b"

                    elif distance(index_tip, thumb_tip, img) <= EPSILON: #Drawing condition (index and thumb touching)
                        index_x, index_y = int(img.shape[1] - (index_tip.x * img.shape[1])), int(index_tip.y * img.shape[0])
                        x_coords.append(index_x)
                        y_coords.append(index_y)
                        if len(x_coords) > smooth_factor: # Smooth the path by averaging recent positions
                            x_coords.pop(0)
                            y_coords.pop(0)
                        avg_x = int(np.mean(x_coords))
                        avg_y = int(np.mean(y_coords))
                        index_x = avg_x
                        index_y = avg_y
                        if last_x is not None and last_y is not None:
                            points = interpolate_line(last_x, last_y, avg_x, avg_y, num_points=10)
                        else:
                            last_x, last_y = None, None
                            x_coords, y_coords = [], []
                            index_x = "a"
                            index_y = "a"
                        last_x, last_y = avg_x, avg_y
                    else:
                        # Reset the drawing points if no hand is detected or fingers are not touching
                        last_x, last_y = None, None
                        x_coords, y_coords = [], []
                        index_x = "a"
                        index_y = "a"
            # Yield the coordinates only if they are valid
            if index_x is not None and index_y is not None:
                yield f"data: {index_x},{index_y}\n\n"
        cap.release()
        cv2.destroyAllWindows()

@app.route('/coordinates')
def coordinates():
    return Response(generate_coordinates_stream(), content_type='text/event-stream')

def start_streams():
    video_thread = threading.Thread(target=generate_frames)
    coordinates_thread = threading.Thread(target=generate_coordinates_stream)
    video_thread.daemon = True  
    coordinates_thread.daemon = True
    video_thread.start()
    coordinates_thread.start()

if __name__ == '__main__':
    start_streams()
    socketio.run(app, debug=True, host='0.0.0.0', port=8080)