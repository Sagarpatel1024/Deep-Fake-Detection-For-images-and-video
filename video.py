import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from statistics import mean

# Load the pre-trained model
model = load_model('model.h5')  # Ensure this path is correct

def load_and_preprocess_image(frame):
    img = cv2.resize(frame, (299, 299))  # Adjust target size to match the model's input
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_video(video_path, model, threshold=0.4, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file at {video_path}")
        return None, None

    predictions = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break

        # Process every `frame_skip` frames
        if frame_count % frame_skip == 0:
            img_array = load_and_preprocess_image(frame)
            prediction = model.predict(img_array)
            predictions.append(prediction[0][0])

        frame_count += 1

    cap.release()

    # Check if predictions are available
    if predictions:
        fake_confidence = mean(predictions)
        final_prediction = "Fake" if fake_confidence >= threshold else "Real"
        print(f"Final Video Prediction: {final_prediction}")
        print(f"Average Fake Confidence: {fake_confidence:.2f}")
        return final_prediction, fake_confidence
    else:
        print("No frames processed. Please check the video file.")
        return None, None

# Run on a sample video
video_path = r'D:\RAGUI\uploads\real.mp4'  # Replace with your video path
predict_video(video_path, model, threshold=0.5, frame_skip=10)
