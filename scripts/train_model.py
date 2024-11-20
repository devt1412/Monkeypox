# For operating system operations (e.g., setting environment variables)
import os

# For numerical operations and array manipulations
import numpy as np

# Main library for building and training neural networks
import tensorflow as tf

# For creating plots and visualizations
import matplotlib.pyplot as plt

# For creating confusion matrix and its display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# For computing class weights to handle imbalanced datasets
from sklearn.utils.class_weight import compute_class_weight

# For data augmentation and preprocessing for image data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# For creating Keras models
from tensorflow.keras.models import Model

# Various layer types for building the neural network
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

# Callbacks for model training (checkpointing, early stopping, learning rate adjustment)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Adam optimizer for model compilation
from tensorflow.keras.optimizers import Adam

# Pre-trained MobileNetV2 model for transfer learning
from tensorflow.keras.applications import MobileNetV2

# L2 regularization to prevent overfitting
from tensorflow.keras.regularizers import l2

# Set seed for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Define constants
img_height, img_width = 224, 224
batch_size = 32
epochs = 30  # Reduced total epochs
num_classes = 4

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data preprocessing
val_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare the data
train_dir = 'data/train'
val_dir = 'data/val'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weight_dict = dict(enumerate(class_weights))

# Build the model using transfer learning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Initially freeze the base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks - save in .keras format first
checkpoint = ModelCheckpoint(
    'models/best_model.keras', 
    save_best_only=True, 
    monitor='val_accuracy', 
    mode='max'
)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# First phase: train only the top layers
history_1 = model.fit(
    train_generator,
    epochs=epochs // 2,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    class_weight=class_weight_dict
)

# Second phase: fine-tune the last few layers
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Freeze all but the last 10 layers
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_2 = model.fit(
    train_generator,
    epochs=epochs // 2,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    class_weight=class_weight_dict
)

# Combine histories
history = {}
for k in history_1.history.keys():
    history[k] = history_1.history[k] + history_2.history[k]

# After training, save final model in both formats
model.save('models/final_model.keras')  # Save in .keras format first
# Convert to h5 format
tf.keras.models.save_model(model, 'models/final_model.h5', save_format='h5')

# Plot training history function
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# Generate and plot confusion matrix function
def plot_confusion_matrix(model, generator):
    y_pred = []
    y_true = []

    for i in range(len(generator)):
        x, y = generator[i]
        y_pred.extend(np.argmax(model.predict(x), axis=1))
        y_true.extend(np.argmax(y, axis=1))
        if i == generator.samples // generator.batch_size - 1:
            break

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=generator.class_indices.keys())
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

# Generate plots
plot_training_history(history)
plot_confusion_matrix(model, val_generator)

print("Training completed. Model saved. Plots generated.")
