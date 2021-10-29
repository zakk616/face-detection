import cv2

video_capture = cv2.VideoCapture('rtsp://admin:junaid123@169.254.91.58:554/Streaming/Channels/102')
# video_capture = cv2.VideoCapture(0)
haar_cascade_face = cv2.CascadeClassifier('resources/haarcascades/haarcascade_frontalface_alt2.xml')

while True:
    ret, frame = video_capture.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces_rects = haar_cascade_face.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
        for (x,y,w,h) in faces_rects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


        cv2.imshow('Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()