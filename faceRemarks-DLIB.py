from imutils import face_utils
import dlib
import cv2
video_capture = cv2.VideoCapture('rtsp://admin:junaid123@169.254.91.58:554/Streaming/Channels/102')
# video_capture = cv2.VideoCapture(0)
p = "resources/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

while True:
    ret, frame = video_capture.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow("Remarks", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()

