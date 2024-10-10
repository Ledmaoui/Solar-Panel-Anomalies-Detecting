import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Parameters
batch_size = 32
img_size = (224, 224)
epochs = 30  # Increased number of epochs
train_dir = 'data/Faulty_solar_panel/'  # Dataset directory path
val_dir = 'data/Faulty_solar_panel/'    # Same directory used for validation

# Data augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_dataset = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Get the class names directly after loading the dataset
class_names = train_dataset.class_indices  # Note: This gives you a dict of {class_name: index}
num_classes = len(class_names)

# Load the pre-trained VGG16 model without the top layer (include_top=False)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze the last few layers of VGG16 for fine-tuning
for layer in base_model.layers[-4:]:  # You can adjust this number depending on your need
    layer.trainable = True

# Build the custom classification head with more layers and Batch Normalization
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
output = Dense(num_classes, activation='softmax')(x)  # Output layer with softmax for multi-class classification

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_solar_panel_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Load the best model saved during training
best_model = tf.keras.models.load_model('best_solar_panel_model.keras')

# Evaluate the best model on the validation dataset
val_loss, val_acc = best_model.evaluate(val_dataset)
print(f'Best Model Validation Accuracy: {val_acc * 100:.2f}%')

# Function to predict fault type in a new image
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = best_model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_names_inv = {v: k for k, v in class_names.items()}  # Reverse the dictionary to get class names
    return class_names_inv[class_idx]

# Example usage to predict a new image
image_path = 'images.jpeg'  # Update this path with your image
predicted_class = predict_image(image_path)
print(f'The predicted class for the image is: {predicted_class}')
