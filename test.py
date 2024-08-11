from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import secrets

app = Flask(__name__)
app.config['SECRET_KEY']='sedsmnck'

# Load your trained model
model = load_model('C:\\Users\\Nikunj\\Desktop\\Space PROj\\models\\fine_tuned_flood_detection_model.h5')

# Define allowed extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess the uploaded image to the required format for the model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

@app.route('/', methods=['POST', 'GET'])
def index_page():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join("static/upload", filename)
            file.save(filepath)

            # Preprocess the image and make a prediction
            processed_image = preprocess_image(filepath)
            prediction = model.predict(processed_image)

            # Decode the prediction to a human-readable format
            prediction_class = np.argmax(prediction, axis=1)[0]
            labels = ['Flooding', 'No Flooding']
            result = labels[prediction_class]

            return render_template('result.html', prediction=result, image_filename=filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
