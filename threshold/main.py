import numpy as np
import cv2 as cv

cap = cv.VideoCapture("../datasets/fast-rc-plane.mp4")

while cap.isOpened():
	ret, frame = cap.read()
	
	if not ret:
		print("Can't receive frame (stream end?). Exiting...")
		break
	frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	ret, thresh  = cv.threshold(frame, 50, 255, cv.THRESH_BINARY_INV)

	contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	frame = cv.addWeighted(frame, 0.5, thresh, 0.5, 0)
	
	frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
	
	if contours:
		largest_contour = max(contours, key=cv.contourArea)

		x, y, w, h = cv.boundingRect(largest_contour)

		rect_size = (100, 100)
		cv.rectangle(frame, (x - int(rect_size[0]/2), y - int(rect_size[1]/2)), 
			            (x + int(rect_size[0]/2), y + int(rect_size[1]/2)), (0, 255, 0), 2)

	cv.imshow("frame", frame)
	if cv.waitKey(1) == ord("q"):
		break

cap.release()
cv.destroyAllWindows()
