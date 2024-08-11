from flask import (Flask, render_template, request, redirect, session)
from werkzeug.utils import secure_filename
import os 
import numpy as np 
import h5py  
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests

# Define the endpoint URL
# url = 'http://127.0.0.1:5000/predict'

# Example data to send (replace with actual input data)
data = {'input': [[1.0, 2.0, 3.0, 4.0]]}

# Send a POST request to the Flask server
# response = requests.post(url, json=data)

# Print the prediction result
# print(response.json())

app = Flask(__name__)
app.config['SECRET_KEY']='sedsmnck'

model = load_model('C:\\Users\\Nikunj\\Desktop\\Space PROj\\models\\fine_tuned_flood_detection_model.h5')

ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess the uploaded image to the required format for the model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

@app.route('/', methods=['POST', 'GET'])
def index_page ():
    if request.method == 'POST':
        image = ''
        file = request.files['file']
        print(file)
        filename =  secure_filename(file.filename)

        if file and allowed_file(file.filename):
            session['image'] = file.filename
            file.save(os.path.join("static/upload", filename))

            processed_image =    preprocess_image(os.path.join("static/upload", filename))
            prediction = model.predict(processed_image)
            prediction_class = np.argmax(prediction, axis=1)[0]
            labels = ['Flooding', 'No Flooding']
            result = labels[prediction_class]

            print(preprocess_image)
            print (result)
            return render_template('index.html', res=result)
        else:
            return redirect('/')

    return render_template('index.html')



# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the data from the POST request
#     data = request.get_json(force=True)
    
#     # Preprocess the data to match model input shape
#     # Assuming input is already in correct shape, otherwise reshape/scale here
#     input_data = np.array(data['input'])
    
#     # Make prediction
#     prediction = model.predict(input_data)
    
#     # Convert the prediction to a human-readable format, if necessary
#     output = prediction.tolist()
    
#     # Return the result as a JSON response
#     return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True, port=5000)