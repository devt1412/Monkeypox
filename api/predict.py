from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the model (adjust the path as necessary)
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.keras')
model = load_model(model_path)

# Define the class names (adjust these to match your classes)
class_names = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    try:
        img = load_img(image_file, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# This is for local testing
if __name__ == '__main__':
    app.run(debug=True)

