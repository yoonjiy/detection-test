from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
from my_model_3 import cnn
from keras.preprocessing import image
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS

model = cnn.emotion_recognition((48, 48, 1))
emotion_classifier = model.load_weights('my_model_3/emotion_weights_30.hdf5')
face_detection = cv2.CascadeClassifier('my_model_3/haarcascade_frontalface_default.xml')
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


def index(request):
    context = {}
    return render(request, "index.html", context)


class EmotionDetect(object):
    def __init__(self):
        self.vs = VideoStream(src=0).start()
        # start the FPS throughput estimator
        self.fps = FPS().start()

    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self):
        cap_image = self.vs.read()
        cap_image = cv2.flip(cap_image, 1)
        cap_img_gray = cv2.cvtColor(cap_image, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(cap_img_gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(cap_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = cap_img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)

            predictions = model.predict(img_pixels)
            emotion_label = np.argmax(predictions)

            emotion_prediction = label_dict[emotion_label]

            cv2.putText(cap_image, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0),
                                1)

            cap_image = cv2.resize(cap_image, (1000, 700))

        self.fps.update()
        ret, jpeg = cv2.imencode('.jpg', cap_image)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def detect(request):
    return StreamingHttpResponse(gen(EmotionDetect()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')