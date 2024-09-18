import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream
import imutils
import threading
import time
import datetime

# Initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames
outputFrame = None
lock = threading.Lock()

# Set up the video capture (0 is for default camera)
cap = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

# Keep track of time
timediff = datetime.datetime.now()

counter = 0
totalBlinks = 0

# FACE DETECTION OR MAPPING THE FACE TO GET THE Eye AND EYES DETECTED
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Constants for eye aspect ratio threshold and consecutive frames
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3

# Variables for blink counting
counter = 0
totalBlinks = 0

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# Main loop - will run until the program is terminated
def detect_blinks():
    global cap, counter, totalBlinks

    # Grab the indexes of the facial landmarks for the left and right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    while True:
        if fileStream and not cap.more():
            break

        # Read a frame from the video stream
        frame = cap.read()

        # Resize the frame for faster processing
        frame = imutils.resize(frame, width=800)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        rects = detector(gray, 0)

        # Loop over the face detections
        for rect in rects:
            # Determine the facial landmarks for the face region
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Extract the left and right eye coordinates
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # Compute the eye aspect ratio (EAR) for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Average the EAR together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # Compute the convex hull for the left and right eye, then visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            # Draw the convex hull for the left and right eyes on the frame
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Check if the EAR is below the blink threshold
            if ear < EYE_AR_THRESH:
                counter += 1
            else:
                # If the eyes were below the threshold for a sufficient number of frames
                if counter >= EYE_AR_CONSEC_FRAMES:
                    totalBlinks += 1
                    # Reset the counter
                    counter = 0

            # Display the blink count on the frame
            cv2.putText(frame, "Blinks: {}".format(totalBlinks), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the video stream with the drawn convex hulls and blink count
        cv2.imshow('Video Stream', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close windows
    cap.stop()
    cv2.destroyAllWindows()

# Run the blink detection in the main thread
detect_blinks()
