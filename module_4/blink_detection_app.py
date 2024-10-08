import cv2
import dlib
import numpy as np
import time
import datetime
from scipy.spatial import distance as dist
from imutils import face_utils, resize
from imutils.video import VideoStream
from plyer import notification
import streamlit as st

# Set up dlib's face detector and the shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Constants for EAR and blinking
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
AVERAGE_BLINKS_PER_MINUTE = 12  # Average number of blinks per minute

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to send an alert notification
def send_alert():
    notification.notify(
        title="Eye Strain Alert",
        message="Your blink rate is below the average! Kindly blink more to avoid eye strain.",
        timeout=10
    )

# Initialize or reset session state variables
if 'totalBlinks' not in st.session_state:
    st.session_state.totalBlinks = 0
if 'blinksInLastMinute' not in st.session_state:
    st.session_state.blinksInLastMinute = 0
if 'startTime' not in st.session_state:
    st.session_state.startTime = datetime.datetime.now()
if 'running' not in st.session_state:
    st.session_state.running = False

# Function to run blink detection
def run_blink_detection():
    stop_btn = st.button('Stop Blink Detection')
    cap = VideoStream(src=0).start()  # Start the video stream
    time.sleep(1.0)  # Warm-up the camera
    counter = 0

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Create a Streamlit placeholder for the video stream
    video_placeholder = st.empty()

    while st.session_state.running:  # Continue only if the app is running
        frame = cap.read()
        frame = resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                counter += 1
            else:
                if counter >= EYE_AR_CONSEC_FRAMES:
                    st.session_state.totalBlinks += 1
                    st.session_state.blinksInLastMinute += 1
                    counter = 0

            cv2.putText(frame, "Blinks: {}".format(st.session_state.blinksInLastMinute), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            

        # Convert the frame to RGB (because Streamlit works with RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update the video stream placeholder with the new frame
        video_placeholder.image(frame, channels="RGB")

        # Check if a minute has passed and send alert if necessary
        currentTime = datetime.datetime.now()
        elapsedTime = (currentTime - st.session_state.startTime).total_seconds()

        if elapsedTime >= 60:
            if st.session_state.blinksInLastMinute < AVERAGE_BLINKS_PER_MINUTE:
                send_alert()
            st.session_state.blinksInLastMinute = 0  # Reset blink count
            st.session_state.startTime = currentTime

        # Check if the Stop button is pressed
        if stop_btn:
            st.session_state.running = False  # Set running to False
            st.experimental_set_query_params(running="False")  # Update the UI
            cap.stop()
            cv2.destroyAllWindows()
            break

    # After the loop ends (either due to stop button or other conditions), show the Start button again
    if not st.session_state.running:
        start_btn = st.button('Start Blink Detection')
        if start_btn:
            st.session_state.running = True  # Set running to True
            st.experimental_set_query_params(running="True")  # Update the UI
            run_blink_detection()  # Restart the detection
    

# Streamlit UI Elements
st.title("Real-Time Eye-Strain Detection System")
st.write("""
    This application detects your blink rate using your webcam to alert you if your blink rate is low, preventing eye strain.
    
    **Benefits**:
    - Helps reduce eye strain during extended screen time.
    - Provides real-time blink count and alerts.
""")

# Button logic for Start and Stop button replacement
if not st.session_state.running:
    start_btn = st.button('Start Blink Detection')
    if start_btn:
        st.session_state.running = True  # Set running to True
        st.experimental_set_query_params(running="True")  # Update the UI
        # print("start_btn:", start_btn)  # Add print statement here
        # Start detection immediately after pressing the button
        run_blink_detection()
    
    

# Real-time Blink Dashboard
st.sidebar.header("Blink Dashboard")
st.sidebar.write(f"Total Blinks Detected: {st.session_state.totalBlinks}")
st.sidebar.write(f"Blink Rate (Last Minute): {st.session_state.blinksInLastMinute}")