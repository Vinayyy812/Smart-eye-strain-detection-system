import cv2

# SETTING UP OF CAMERA TO 1 YOU
# CAN EVEN CHOOSE 0 IN PLACE OF 1
cap = cv2.VideoCapture(0)

# MAIN LOOP IT WILL RUN ALL THE UNLESS 
# AND UNTIL THE PROGRAM IS BEING KILLED 
# BY THE USER
while True:
	null, frame = cap.read()
	gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.imshow("Camera Detection Code", frame)
	
    #TO QUIT THE VIDEO FRAME CLICK ESCAPE BUTTON WHOSE KEY VALUE IS 27
	key = cv2.waitKey(9)
	if key == 27:
		break
cap.release()
cv2.destroyAllWindows()