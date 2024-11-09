from flask import Flask, Response, render_template, send_from_directory, redirect, make_response, jsonify
from flask_socketio import SocketIO
import threading
import cv2
import mediapipe as mp
import numpy as np 

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

def generate_coordinates_stream():
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

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
                    index_tip = hand_landmarks.landmark[8]
                    thumb_tip = hand_landmarks.landmark[4]

                    h, w, _ = img.shape
                    index_x, index_y = int(w - (index_tip.x * w)), int(index_tip.y * h)

                    # Yield coordinates to Flask response
                    yield f'data: {index_x},{index_y}\n\n'
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

@app.route('/coordinates')
def coordinates():
    return Response(generate_coordinates_stream(), content_type='text/event-stream')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8080)
