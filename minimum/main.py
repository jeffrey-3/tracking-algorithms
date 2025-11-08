import numpy as np
import cv2 as cv

cap = cv.VideoCapture("../datasets/fast-rc-plane.mp4")

x = 200
y = 200

window_size = 100

def check_window_outside_frame(x, y, window_size, frame_width, frame_height):
	return min(x, y) < int(window_size / 2) or x + int(window_size / 2) > frame_width or y + int(window_size / 2) > frame_height

def get_mouse_coords(event, mouse_x, mouse_y, flags, param):
	global x, y
	if event == cv.EVENT_LBUTTONDOWN:  # Check if left mouse button was clicked
		x, y = mouse_x, mouse_y
		print((x, y))

cv.namedWindow("frame")
cv.setMouseCallback("frame", get_mouse_coords)

while cap.isOpened():
	ret, frame = cap.read()
	
	if not ret:
		print("Can't receive frame (stream end?). Exiting...")
		break
	
	frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	coarse = check_window_outside_frame(x, y, window_size, frame.shape[1], frame.shape[0])
	
	window = None
	if coarse:
		window = frame.copy()
	else:
		window = frame[y - int(window_size / 2):y + int(window_size / 2),
			       x - int(window_size / 2):x + int(window_size / 2)].copy()

	# window = cv.GaussianBlur(window, (3, 3), 0)
	
	(min_val, max_val, min_loc, max_loc) = cv.minMaxLoc(window)
	
	# Filter out non-bright false tracks such as clouds
	# if min_val > 100:
	#	coarse = True

	if coarse:
		x = min_loc[0]
		y = min_loc[1]
	else:
		x += min_loc[0] - int(window_size / 2)
		y += min_loc[1] - int(window_size / 2)

		cv.rectangle(frame, (x - int(window_size/2), y - int(window_size/2)), 
		            	    (x + int(window_size/2), y + int(window_size/2)), (0, 255, 0), 2)

	cv.imshow("frame", frame)
	
	if cv.waitKey(10) == ord("q"):
		break

cap.release()
cv.destroyAllWindows()

