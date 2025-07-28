import os
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from moviepy.editor import VideoFileClip

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model for deepfake detection
model = load_model('model.h5')

# Directory for storing uploaded files
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index')

# Image upload route for prediction
@app.route('/upload', methods=['POST'])
def upload_image_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load and preprocess the image
        img_array = load_and_preprocess_image(file_path)

        # Make prediction
        prediction = model.predict(img_array)
        prediction_probability = round(float(prediction[0][0]) * 100, 2)

        # Determine predicted label
        predicted_label = "Fake" if prediction[0][0] >= 0.5 else "Real"

        return render_template(
            'index.html',
            predicted_label=predicted_label,
            prediction_probability=prediction_probability,
            uploaded_image=url_for('static', filename=f'uploads/{filename}')
        )
    return 'No file uploaded', 400

# Video prediction route
@app.route('/video_predict')
def video_predict():
    return render_template('video_predict.html')

# Video upload route for prediction
@app.route('/upload_video', methods=['POST'])
def upload_video_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # Extract a frame to show as a thumbnail (first frame of the video)
        thumbnail_filename = f"{os.path.splitext(filename)[0]}_thumbnail.jpg"
        thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], thumbnail_filename)
        extract_first_frame(video_path, thumbnail_path)

        # Dummy prediction for video (replace with your actual model prediction logic)
        prediction_label = "Fake"  # Placeholder
        prediction_probability = 85.5  # Placeholder

        return render_template(
            'video_predict.html',
            uploaded_video=url_for('static', filename=f'uploads/{filename}'),
            thumbnail_image=url_for('static', filename=f'uploads/{thumbnail_filename}'),
            predicted_label=prediction_label,
            prediction_probability=prediction_probability
        )

    return 'No file uploaded', 400

def load_and_preprocess_image(img_path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(img_path, target_size=(299, 299))  # Adjust as per model input size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def extract_first_frame(video_path, thumbnail_path):
    clip = VideoFileClip(video_path)
    frame = clip.get_frame(0)  # Extract the first frame
    from PIL import Image
    image = Image.fromarray(frame)
    image.save(thumbnail_path)

if __name__ == '__main__':
    app.run(debug=True)
