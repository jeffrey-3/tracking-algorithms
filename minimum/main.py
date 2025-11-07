import numpy as np
import cv2 as cv

cap = cv.VideoCapture("../datasets/fast-rc-plane.mp4")

x = 0
y = 0

window_size = 100

while cap.isOpened():
	ret, frame = cap.read()
	
	if not ret:
		print("Can't receive frame (stream end?). Exiting...")
		break
	
	frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	coarse = False
	window = None
	if (x == 0 and y == 0) or min(x, y) < int(window_size / 2) or x + int(window_size / 2) > frame.shape[1] or y + int(window_size / 2) > frame.shape[0]:
		coarse = True

	if coarse:
		window = frame.copy()
	else:
		window = frame[y - int(window_size / 2):y + int(window_size / 2),
			       x - int(window_size / 2):x + int(window_size / 2)].copy()

	window = cv.GaussianBlur(window, (3, 3), 0)
	
	(min_val, max_val, min_loc, max_loc) = cv.minMaxLoc(window)

	if coarse:
		x = min_loc[0]
		y = min_loc[1]
	else:
		x += min_loc[0] - int(window_size / 2)
		y += min_loc[1] - int(window_size / 2)

	cv.rectangle(frame, (x - int(window_size/2), y - int(window_size/2)), 
		            (x + int(window_size/2), y + int(window_size/2)), (0, 255, 0), 2)

	print(coarse)

	if not coarse:
		cv.imshow("window", cv.resize(window, (300, 300)))
	
	cv.imshow("frame", cv.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2))))
	
	if cv.waitKey(1) == ord("q"):
		break

cap.release()
cv.destroyAllWindows()

