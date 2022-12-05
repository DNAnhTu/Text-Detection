# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import time
import cv2

def decode_predictions(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		for x in range(0, numCols):
			if scoresData[x] < 0.5:
				continue
			
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	return (rects, confidences)

(W, H) = (None, None)
(newW, newH) = (320,320)
(rW, rH) = (None, None)

layerNames = [
	"feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

print("starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

fps = FPS().start()

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    orig = frame.copy()

    (H, W) = frame.shape[:2]

    rW = W / float(newW)
    rH = H / float(newH)

    frame = cv2.resize(frame, (newW, newH))
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # cv2.putText(orig, pytesseract.image_to_string(orig[startY:endY, startX:endX]), (startX, startY), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
        
    cv2.imshow("Text Detection", orig)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        break

    fps.update()