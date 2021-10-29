import cv2
import dlib

video_capture = cv2.VideoCapture('rtsp://admin:junaid123@169.254.91.58:554/Streaming/Channels/102')
# video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hogFaceDetector = dlib.get_frontal_face_detector()
        faces = hogFaceDetector(gray, 1)

        for (i, rect) in enumerate(faces):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        cv2.imshow("Image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
