import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
model = load_model('action.h5')

actions = np.array(['hello', 'thanks', 'iloveyou', 'Indian', 'Bye', 'Hearing', 'Man', 'Woman', 'Namaste', 'Love', 'Yes', 'No', 'sign', 'Good', 'Sorry'])

# Set up MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

# Create a Streamlit component for video display
st.title("Sign Language Detection")

# Create a video capture object using OpenCV
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.write("Failed to open webcam.")
else:
    sequence = []
    sentence = []
    threshold = 0.8

    # Create placeholders for dynamic updates
    stframe = st.empty()
    prediction_placeholder = st.empty()

    # Set up the MediaPipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break

            # Flip the frame horizontally for natural selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB for MediaPipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Extract keypoints and make a prediction
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep the last 30 keypoint sequences

            if len(sequence) == 30:
                prediction_input = np.expand_dims(sequence, axis=0)

                try:
                    # Make a prediction and handle possible errors
                    res = model.predict(prediction_input)[0]

                    # Determine the action with the highest probability
                    action_index = np.argmax(res)
                    action_probability = res[action_index]

                    # Check if the action probability is above the threshold
                    if action_probability > threshold:
                        action = actions[action_index]
                        if len(sentence) > 0:
                            if action != sentence[-1]:
                                sentence.append(action)
                        else:
                            sentence.append(action)
                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    # Update the prediction placeholder with the current action
                    prediction_placeholder.write(f"Predicted action: {sentence[-1] if sentence else 'None'}")

                except Exception as e:
                    st.write(f"Error during prediction: {e}")

            # Draw landmarks on the frame
            draw_styled_landmarks(frame, results)
            cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Convert frame to PIL Image for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)

            # Display the image in the Streamlit app
            stframe.image(pil_img)

    cap.release()
    st.write("Webcam released.")