import logging
import base64
import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import requests

from flask_cors import CORS
import traceback
from dotenv import load_dotenv
load_dotenv() 
# Disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)
CORS(app)

# Set maximum upload size to 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained CNN model once at the start
model = load_model('sketch_to_color_model.h5 (2).keras')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.debug("Received request for prediction.")
        
        # Decode Base64 image if sent via JSON
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Extract image URL from the JSON structure
        image_data = data['image']['_streams'][1]  # Adjust according to your structure
        a = image_data.split('uploads/')[1]
        p = a
        a = a[len(a)-(len(a)-a.find('.'))+1:].upper()
        if a == 'JPG':
            a = 'JPEG'
            p=p.replace('JPG','JPEG')
        
        # Fetch the image from the local server
        response = requests.get(image_data)
        img = Image.open(BytesIO(response.content)).convert('RGB')  # Ensure it's in RGB format
        # Preprocess the image for the model
        img = img.resize((256, 256))  # Resize image for the model
        image = np.array(img) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make the prediction
        prediction = model.predict(image)

        # Convert the prediction back to an image
        predicted_image = (prediction[0] * 255).astype(np.uint8)
        output_image = Image.fromarray(predicted_image)

        # Save the predicted image in memory (without writing to disk)
        image_io = BytesIO()
        output_image.save(image_io, format=a)  # Save as JPEG or other format
        image_io.seek(0)  # Move to the beginning of the BytesIO buffer

        # Send the image to the Node.js server
        files = {
            'images': (f'{p}', image_io, f'image/{a.lower()}'),
            'name': (None, data['user']),
            'filename': (None, f'{p}'),
        }

        logging.debug(f"Sending image to Node.js server for user: {data['user']}")
        response = requests.post(f'https://backend-9oaz.onrender.com/vendor/sktvendor/{data["user"]}', files=files)

        # Check if the response from Node.js server is successful
        if response.status_code != 200:
            logging.error(f"Failed to upload image to Node.js server, status code: {response.status_code}")
            return jsonify({'error': 'Failed to upload image to Node.js server'}), 500

        logging.debug('Image successfully processed and sent to Node.js server')
        return jsonify({'message': 'Image processed successfully', 'node_response': response.json()}), 200
    
    except Exception as e:
        logging.error("Error processing the image: %s", str(e))
        logging.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
