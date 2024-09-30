import logging
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import requests  # Import requests for sending HTTP requests
from flask_cors import CORS
import mimetypes
import traceback
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Set maximum upload size to 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained CNN model once at the start
model = load_model('sketch_to_color_model.h5 (2).keras')

def is_image_file(filename):
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type and mime_type.startswith('image')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.debug("Received request for prediction.")
        # Ensure an image file was sent in the request
        if 'image' not in request.files:
            logging.error('No image file provided.')
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '':
            logging.error('No selected file.')
            return jsonify({'error': 'No selected file'}), 400

        if not is_image_file(file.filename):
            logging.error('Uploaded file is not an image.')
            return jsonify({'error': 'Uploaded file is not an image'}), 400

        # Open the image file and preprocess it
        with Image.open(file) as img:
            img = img.convert('RGB')  # Ensure it's RGB
            img = img.resize((256, 256))  # Resize image for the model
            image = np.array(img) / 255.0  # Normalize the image
            image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make the prediction
        prediction = model.predict(image)

        # Convert the prediction back to an image
        predicted_image = (prediction[0] * 255).astype(np.uint8)
        output_image = Image.fromarray(predicted_image)

        # Save the output image to a temporary file
        temp_output_path = 'output/' +file.filename
        output_image.save(temp_output_path)

        # Send the image to the server running on localhost:4000
        with open(temp_output_path, 'rb') as img_file:
            response = requests.post('http://localhost:4000/save_image', files={'image': img_file})
            if response.status_code != 200:
                logging.error('Failed to save image on localhost:4000: %s', response.text)
                return jsonify({'error': 'Failed to save image on server'}), 500

        logging.debug('Image successfully saved on localhost:4000')
        return jsonify({'message': 'Image successfully saved on localhost:4000', 'filename':temp_output_path}), 200

    except Exception as e:
        logging.error("Error processing the image: %s", str(e))
        logging.debug(traceback.format_exc())  # Print the traceback for debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
