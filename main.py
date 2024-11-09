import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
whiteboard = np.ones((500, 500, 3), dtype=np.uint8) * 255

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
                for id, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    
                    print(f"ID: {id}, X: {cx}, Y: {cy}")

                    if(id == 8):
                        cv2.circle(whiteboard, (cx, cy), 3, (0,0,0), -1)
                        pass 

        cv2.imshow("Hand Tracking", img)
        cv2.imshow("Whiteboard", whiteboard)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()