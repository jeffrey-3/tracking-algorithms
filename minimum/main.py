import numpy as np
import cv2 as cv

cap = cv.VideoCapture("../datasets/fast-rc-plane.mp4")

while cap.isOpened():
	ret, frame = cap.read()
	
	if not ret:
		print("Can't receive frame (stream end?). Exiting...")
		break
	frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	
	(min_val, max_val, min_loc, max_loc) = cv.minMaxLoc(frame)

	x = min_loc[0]
	y = min_loc[1]
	
	rect_size = (100, 100)
	cv.rectangle(frame, (x - int(rect_size[0]/2), y - int(rect_size[1]/2)), 
			            (x + int(rect_size[0]/2), y + int(rect_size[1]/2)), (0, 255, 0), 2)

	cv.imshow("frame", frame)
	if cv.waitKey(1) == ord("q"):
		break

cap.release()
cv.destroyAllWindows()

