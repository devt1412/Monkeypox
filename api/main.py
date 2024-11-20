from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify your allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
model = tf.keras.models.load_model('models/final_model.h5')
classes = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']

def preprocess_image(image_bytes):
    # Read the image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize
    img = img.resize((224, 224))
    
    # Convert to array and preprocess
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

@app.get("/")
def read_root():
    return {"message": "Skin Disease Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the file
    image_bytes = await file.read()
    
    # Preprocess the image
    img_array = preprocess_image(image_bytes)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            class_name: float(prob) 
            for class_name, prob in zip(classes, prediction[0])
        }
    }
