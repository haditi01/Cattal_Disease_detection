from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
model_path = os.path.join('C:\\', 'Users', 'aditi', 'OneDrive', 'Documents', 'desktop data', 'design project 1', 'VScode_project', 'cattle_disease_model.h5')
model = load_model(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_disease(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    # Assuming the classes are ordered as ['FMD', 'IBK', 'LSD', 'NOR', 'Unknown']
    disease_labels = ['FMD', 'IBK', 'LSD', 'NOR']
    threshold = 0.8  # Adjust the threshold as needed
    max_confidence = np.max(prediction)
    if max_confidence < threshold:
        return 'Unknown', max_confidence
    else:
        predicted_class_index = np.argmax(prediction)
        predicted_class = disease_labels[predicted_class_index]
        confidence = prediction[0][predicted_class_index]
        return predicted_class, confidence

def is_cow(image_path):
    # Add code to determine if the image contains a cow
    # For simplicity, let's assume a placeholder function that always returns True
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        if not is_cow(file_path):
            return render_template('not_cow.html', filename=filename)
        
        predicted_class, confidence = predict_disease(file_path)
        return render_template('result.html', filename=filename, prediction=predicted_class, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
