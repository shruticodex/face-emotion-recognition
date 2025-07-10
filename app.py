from flask import Flask, render_template, Response, jsonify
from keras.models import load_model
import cv2
import numpy as np
from collections import deque

app = Flask(__name__)

# Load model and emotion labels
model = load_model("model/emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Webcam
camera = cv2.VideoCapture(0)

# Emotion tracking
emotion_history = deque(maxlen=10)  # last 10 emotions
latest_emotion = 'Analyzing'

def preprocess_face(gray_frame, face_coords):
    x, y, w, h = face_coords
    roi_gray = gray_frame[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi = roi_gray.astype("float") / 255.0
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=-1)
    return roi

def generate_frames():
    global latest_emotion
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = preprocess_face(gray, (x, y, w, h))
                prediction = model.predict(roi, verbose=0)
                label = emotion_labels[np.argmax(prediction)]
                latest_emotion = label
                emotion_history.append(label)

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (36, 255, 12), 2)

            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_history')
def emotion_api():
    return jsonify(list(emotion_history))

if __name__ == '__main__':
    app.run(debug=True)
