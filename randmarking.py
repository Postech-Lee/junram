import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

title='IU'
image_path = '.\\'+title+'.jpg'
image = cv2.imread(image_path)

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.detections:
        copy_image = image.copy()
        for detection in results.detections:
            mp_drawing.draw_detection(copy_image, detection,
                                      bbox_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1))

        cv2.imwrite("./landmarked/" + title + ".png", copy_image)
