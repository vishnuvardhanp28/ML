import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Configuration
DATA_DIR = 'gtsrb'
NUM_CLASSES = 43
IMG_HEIGHT = 30
IMG_WIDTH = 30
CHANNELS = 3
EPOCHS = 15
BATCH_SIZE = 32

def load_data(data_dir):
    data = []
    labels = []
    
    # The CS50 dataset has the structure: gtsrb/0, gtsrb/1, etc.
    # So we don't need to append 'Train'
    train_path = data_dir
    
    if not os.path.exists(train_path):
        print(f"Error: Dataset not found at {train_path}")
        return None, None

    print("Loading data...")
    classes = NUM_CLASSES
    for i in range(classes):
        path = os.path.join(train_path, str(i))
        images = os.listdir(path)
        for a in images:
            try:
                image = Image.open(path + '\\'+ a)
                image = image.resize((IMG_HEIGHT, IMG_WIDTH))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except Exception as e:
                print(f"Error loading image: {e}")
        print(f"Loaded class {i}")

    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def build_model(input_shape):
    model = Sequential()
    
    # First Conv Layer
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    
    # Second Conv Layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    
    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    return model

def main():
    # 1. Load Data
    data, labels = load_data(DATA_DIR)
    
    if data is None:
        return

    print(f"Data loaded. Shape: {data.shape}, Labels: {labels.shape}")

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # 3. One-hot encoding
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    # 4. Build Model
    model = build_model((IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    model.summary()

    # 5. Compile Model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 6. Train Model
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test))

    # 7. Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/traffic_classifier.h5')
    print("Model saved to models/traffic_classifier.h5")

    # 8. Plot Accuracy
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    print("Accuracy plot saved as accuracy_plot.png")

if __name__ == '__main__':
    main()
