import pickle
import cv2
import mediapipe as mp
import numpy as np
import json
import os

# Load pre-trained model from file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize hand tracking and drawing utilities from mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def read_dict(file_path):
    path = os.path.expanduser(file_path)   # Expand the tilde character in the file path to the user's home directory
    with open(path, "r") as f:     # Open the file at the given file path in read mode
        dict_json = json.load(f)   # Load the JSON data from the file object into a Python dictionary
    return dict_json               # Return the resulting dictionary object


# Initialize hand tracking using mediapipe
hands = mp_hands.Hands(static_image_mode=True,
                        min_detection_confidence=0.3,
                        max_num_hands=2,
                        )

sign_path = './sign_to_prediction_index_map.json'
sign_mapping = read_dict(sign_path)    

sign_mapping = {int(key): value for key, value in sign_mapping.items()}  # to convert the keys in dict from (str) into (int)


# Initialize video capture object
cap = cv2.VideoCapture(0)
# Set the frame rate to 30 fps
cap.set(cv2.CAP_PROP_FPS, 30)

# Continuously capture video from default camera
while True:
    # Initialize empty lists to store hand landmark coordinates
    data_aux = []
    x_ = []
    y_ = []

    # Read a frame from the camera
    ret, frame = cap.read()
    # Flip the frame horizontally
    frame = cv2.flip(frame ,1)
    cv2.putText(frame, 'To Exit press (Q)', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(242, 218, 7), 2,cv2.LINE_AA)

    # Get the dimensions of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB color space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks using mediapipe
    results = hands.process(frame_rgb)
    
    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        # Draw the landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

        # Draw a rectangle around the hand gesture and display the predicted character label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Display the processed frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #if u pressed q close the window and break       
        break   


cap.release() # Release the video capture object and close all windows
cv2.destroyAllWindows()