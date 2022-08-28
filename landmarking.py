import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
def landmarking(path):
    i = 1
    for filename in os.listdir(path):
        image = cv2.imread(path)
        print('landmark '+str(i)+'.jpg')
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.detections:
                copy_image = image.copy()
                for detection in results.detections:
                    mp_drawing.draw_detection(copy_image, detection,
                                              bbox_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1))
                cv2.imwrite(path+'landmark'+str(i)+ ".jpg", copy_image)
        i += 1

changeName('/Users/aaron/Desktop/testFolder/')