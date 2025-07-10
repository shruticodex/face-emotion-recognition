# train_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
train_dir = 'data/train'
val_dir = 'data/test'

# Image Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, zoom_range=0.2, shear_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(48, 48), color_mode='grayscale', batch_size=64, class_mode='categorical')
val_data = val_datagen.flow_from_directory(val_dir, target_size=(48, 48), color_mode='grayscale', batch_size=64, class_mode='categorical')

# CNN Model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(7, activation='softmax')  # 7 emotion classes
])

# Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=30, validation_data=val_data)

# Save the model
os.makedirs('model', exist_ok=True)
model.save('model/emotion_model.h5')
