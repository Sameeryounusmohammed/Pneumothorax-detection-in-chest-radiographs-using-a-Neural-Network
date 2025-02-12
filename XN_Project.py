# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:20:12 2023
@author: 16172
"""

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from vit_keras import vit

# Function to load and process images
def load_and_process_image(file_path, mask=False):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1 if mask else 3)
    image = tf.image.resize(image, [512, 512])
    if not mask:
        image = tf.cast(image, tf.float32) / 255.0
    return image

# Function to create TensorFlow datasets
def load_dataset(image_paths, mask_paths, batch_size=32):
    image_ds = tf.data.Dataset.from_tensor_slices(image_paths).map(load_and_process_image)
    mask_ds = tf.data.Dataset.from_tensor_slices(mask_paths).map(lambda x: load_and_process_image(x, mask=True))
    ds = tf.data.Dataset.zip((image_ds, mask_ds))
    ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return ds

# Paths to your directories
mask_folder_path = "C:/Users/16172/Downloads/archive/input/train/images/512/mask"
dicom_folder_path = "C:/Users/16172/Downloads/archive/input/train/images/512/dicom"

# Get paths of all mask and DICOM images in PNG format
mask_image_paths = [os.path.join(mask_folder_path, f) for f in os.listdir(mask_folder_path) if f.endswith('.png')]
dicom_png_image_paths = [os.path.join(dicom_folder_path, f) for f in os.listdir(dicom_folder_path) if f.endswith('.png')]

# Split the dataset into training, validation, and test sets
train_paths, val_test_paths, train_mask_paths, val_test_mask_paths = train_test_split(dicom_png_image_paths, mask_image_paths, test_size=0.3, random_state=42)
val_paths, test_paths, val_mask_paths, test_mask_paths = train_test_split(val_test_paths, val_test_mask_paths, test_size=0.5, random_state=42)

# Create TensorFlow datasets for efficient loading
train_dataset = load_dataset(train_paths, train_mask_paths)
val_dataset = load_dataset(val_paths, val_mask_paths)
test_dataset = load_dataset(test_paths, test_mask_paths)

# Load the base ViT model
vit_model = vit.vit_b16(
    image_size=512,
    activation='softmax',
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

# Define the custom head for our dataset
input_shape = (512, 512, 3)
inputs = Input(shape=input_shape)

# Use the ViT model as a layer
x = vit_model(inputs)

# Add custom layers on top of the ViT model

x = Dense(512, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

# Define the model
model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save('pneumothorax_vit_model.h5')
