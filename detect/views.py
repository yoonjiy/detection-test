import time

from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
from rest_framework.response import Response

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

    def get_frame(self, start, emotion_cnt):
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

            elapsed = time.time() - start

            if (elapsed > 5):
                count_result(emotion_cnt, emotion_prediction, elapsed)

        self.fps.update()
        ret, jpeg = cv2.imencode('.jpg', cap_image)
        return jpeg.tobytes()


def gen(camera):
    emotion_cnt = {'Angry': 0, 'Happiness': 0, 'Sad': 0, 'Neutral': 0}
    start = time.time()
    while True:
        frame = camera.get_frame(start, emotion_cnt)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def show_result(cnt):
    dict = sorted(cnt.items(), key=lambda item: item[1], reverse=True)
    print(dict[0])
    # response 반환하기


def count_result(cnt, emotion, elapsed):
    if elapsed >= 15:
        show_result(cnt)
    elif emotion == 'Angry':
        cnt['Angry'] += 1
    elif emotion == 'Happiness':
        cnt['Happiness'] += 1
    elif emotion == 'Sad':
        cnt['Sad'] += 1
    else:
        cnt['Neutral'] += 1


def detect(request):
    return StreamingHttpResponse(gen(EmotionDetect()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')