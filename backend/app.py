from flask import Flask, render_template, Response, request
import cv2
import pickle
import mediapipe as mp
import numpy as np

app = Flask(__name__, template_folder='../templates', static_folder='../static')

model_dict = pickle.load(open('./backend/model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
labels_dict = {i: chr(65 + i) for i in range(26)}

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux, x_, y_ = [], [], []
            for lm in results.multi_hand_landmarks[0].landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in results.multi_hand_landmarks[0].landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            fontScale = 3  # Adjust for size
            thickness = 5  # Adjust for boldness

            cv2.putText(frame, predicted_character, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 0, 0), thickness)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
