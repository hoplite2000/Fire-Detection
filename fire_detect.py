from alert import mailer
from tensorflow.keras.models import load_model
import config
from imutils import paths
from datetime import datetime
from cloud_upload import *
import numpy as np
import imutils
import cv2
import time
import os
import multiprocessing

class Process(multiprocessing.Process):
    def __init__(self, path, msg):
        super(Process, self).__init__()
        self.path = path
        self.msg = msg

    def run(self):
        cloud_upload(self.path)
        if(self.msg != ""):
            print(self.msg)

class Mail(multiprocessing.Process):
    def __init__(self):
        super(Mail, self).__init__()

    def run(self):
        mailer()

print("[INFO] loading model...")
model = load_model(config.Model_path)
print("[INFO] model loaded...")

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
fire = 0

vs = cv2.VideoCapture("./inputs/fire.mp4")
#vs = cv2.VideoCapture(0)
start_time = time.time()
output_video  = None

print("[INFO] Starting the Stream...")

while True:
    (frame_exists, frame) = vs.read()
    if not frame_exists:
        break
    else:
        output = frame.copy()
        frame = cv2.resize(frame, (128, 128))
        frame = frame.astype("float32") / 255.0
        
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        j = np.argmax(preds)
        label = config.Classes[j]

        output = imutils.resize(output, width=700)
        if label == 'Non-Fire':
            if(fire >0):
                fire = fire-1
            cv2.putText(output, "Non-Fire", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, COLOR_GREEN, 2)
        else:
            fire = fire+1
            cv2.putText(output, "WARNING! Fire!", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, COLOR_RED, 2)

        if(fire == 50):
            mail = Mail()
            mail.start()

        now = datetime.now()
        date_text = now.strftime("%a %d-%m-%Y %H:%M:%S")
        cv2.putText(output, date_text, (10, output.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_BLUE, 2)

        cv2.imshow("Cam", output)
        key = cv2.waitKey(1) & 0xFF

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        if  output_video is None:
            video_file_name = str(datetime.now().strftime("%c")) + ".avi"
            output_video = cv2.VideoWriter(video_file_name, fourcc, 25, (output.shape[1], output.shape[0]), True)

        if time.time() - start_time > 3600:
            pvideo_file_name = str(datetime.now().strftime("%c")) + ".avi"
            p = Process(pvideo_file_name, "")
            p.start()
            start_time = time.time()
            video_file_name = str(datetime.now().strftime("%c")) + ".avi"
            output_video = cv2.VideoWriter(video_file_name, fourcc, 25,(output.shape[1], output.shape[0]), True)

        if output_video is not None:
            output_video.write(output)

        if key == 27:
            break

pe = Process(video_file_name, "[INFO] Uploaded all files to cloud...")
pe.start()

vs.release()
cv2.destroyAllWindows()



