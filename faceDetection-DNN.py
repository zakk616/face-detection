from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

net = cv2.dnn.readNetFromCaffe("resources/dnn/deploy.prototxt.txt", "resources/dnn/res10_300x300_ssd_iter_140000.caffemodel")
video_capture = VideoStream('rtsp://admin:junaid123@169.254.91.58:554/Streaming/Channels/102').start()
# video_capture = VideoStream(src=0).start()

while True:
	frame = video_capture.read()
	frame = imutils.resize(frame, width=400)

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	net.setInput(blob)
	detections = net.forward()


	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence < 0.5:
			continue

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
video_capture.stop()