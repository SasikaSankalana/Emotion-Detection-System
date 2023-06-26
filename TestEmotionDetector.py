import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


def load_emotion_model():
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights("model/emotion_model.h5")
    return emotion_model


def detect_emotions(emotion_model):
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break

        face_detector = cv2.CascadeClassifier(
            'haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        num_of_faces = face_detector.detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_of_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray, (48, 48)), -1), 0)

            emotion_prediction = emotion_model.predict(cropped_img)
            max_index = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[max_index], (x+5, y-20),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


emotion_model = load_emotion_model()
detect_emotions(emotion_model)
