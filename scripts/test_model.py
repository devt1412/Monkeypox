import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Load the saved model
model = tf.keras.models.load_model('models/final_model.h5')

# Function to preprocess and predict on a single image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to the same size as the training images
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to match the training preprocessing
    return img_array

def predict_image(model, img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return prediction

# Test with a single image (manual testing)
def test_single_image(img_path):
    prediction = predict_image(model, img_path)

    # Decode prediction
    classes = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']
    predicted_class = classes[np.argmax(prediction)]
    print(f"Prediction for {img_path}: {predicted_class}")


def test_batch_images():
    test_dir = 'data/test'

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',  # Change to categorical for multi-class classification
        shuffle=False
    )

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":

    # Option 1: Test with a single image
    img_path = 'data/val/healthy/healthyv44.png'
    test_single_image(img_path)

    # Option 2: Test with a batch of images from test dataset
    # test_batch_images()
