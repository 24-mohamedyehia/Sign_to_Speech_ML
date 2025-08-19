# Import necessary libraries
import pickle
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import os
import json

# Load pre-trained model from file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize hand tracking and drawing utilities from mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hand tracking using mediapipe
hands = mp_hands.Hands(static_image_mode=True,
                        min_detection_confidence=0.3,
                        max_num_hands=2,
                        )


# Function to read JSON file
def read_dict(file_path):
    path = os.path.expanduser(file_path)
    with open(path, "r") as f:
        dict_json = json.load(f)
    return dict_json

# Load sign mapping from JSON file
sign_mapping = read_dict('./sign_to_prediction_index_map.json')
sign_mapping = {int(key): value for key, value in sign_mapping.items()}

import time

# Initialize video capture object
cap = cv2.VideoCapture(0)
# Set the frame rate to 30 fps
cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize variables to store previous prediction and time
previous_prediction = None
last_prediction_time = time.time()

# Continuously capture video from default camera
while True:
    # Initialize empty lists to store hand landmark coordinates
    data_aux = []
    x_ = []
    y_ = []

    # Read a frame from the camera
    ret, frame = cap.read()
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, 'To Exit press (Q)', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(242, 218, 7), 2,cv2.LINE_AA)

    # Get the dimensions of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB color space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks using mediapipe
    results = hands.process(frame_rgb)
    
    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        # Get the coordinates of each hand landmark
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                # Append the coordinates to the respective lists
                x_.append(x)
                y_.append(y)
            
            # Calculate the normalized coordinates of each landmark
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Calculate the minimum and maximum values of the coordinates
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict the character label using the pre-trained model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = sign_mapping[int(prediction[0])]

        # If the prediction changes and it has been at least 3 seconds since the last prediction
        if predicted_character != previous_prediction and time.time() - last_prediction_time >= 3:
            # Convert predicted text to speech
            tts = gTTS(predicted_character, lang='en')
            tts.save('predicted_audio.mp3')
            os.system('start predicted_audio.mp3')  # Play the audio file
            previous_prediction = predicted_character
            last_prediction_time = time.time()

    # Display the processed frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #if u pressed q close the window and break      
        break 

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
