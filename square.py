import cv2
import mediapipe as mp
import time
import os
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
eye_casecade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')

name="4"
cap=cv2.VideoCapture("videos/"+name+".mp4")
pTime=0

mpFaceDetection = mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils

faceDetection = mpFaceDetection.FaceDetection() #페이스디텍션 시작
imgNum=0
if not os.path.exists("./nowtrying"):
    os.makedirs("./nowtrying")
if not os.path.exists("./nowtrying/"+name):
    os.makedirs("./nowtrying/"+name)
while True:
    success,img=cap.read()

    imgRGB = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2RGB)

    results = faceDetection.process(imgRGB)

    if results.detections: #가능하면
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img,detection)
            # print(id,detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data. relative_bounding_box
            ih, iw, ic = img.shape
            x=int(bboxC.xmin * iw)
            y=int(bboxC.ymin * ih)
            w=int(bboxC.width * iw)
            h=int(bboxC.height * ih)
            cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
            # 이미지를 저장
            cv2.imwrite("./nowtrying/"+name+"/" + str(imgNum) + ".png", cropped)
            imgNum += 1
            #cv2.rectangle(img, bbox, (250,123, 0), 2)
    cTime = time.time()

    try:
        fps=1/(cTime-pTime)
    except:
        fps=0
    pTime=cTime
    cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,
                3,(255,255,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(2)