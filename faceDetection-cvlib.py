import cvlib as cv
import cv2

video_capture = cv2.VideoCapture('rtsp://admin:junaid123@169.254.91.58:554/Streaming/Channels/102')
# video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()
    if ret:
        face, confidence = cv.detect_face(frame)

        for idx, f in enumerate(face):
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, "", (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0), 2)

        cv2.imshow("faceDetection-cvlib", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
video_capture.release()
cv2.destroyAllWindows()   