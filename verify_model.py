import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Configuration
MODEL_PATH = 'models/traffic_classifier.h5'
IMG_PATH = 'gtsrb/14/00000_00014.ppm' # Stop sign

# Classes dictionary
classes = { 
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)', 
    2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 
    4:'Speed limit (70km/h)', 
    5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 
    7:'Speed limit (100km/h)', 
    8:'Speed limit (120km/h)', 
    9:'No passing', 
    10:'No passing veh over 3.5 tons', 
    11:'Right-of-way at intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve left', 
    20:'Dangerous curve right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice/snow',
    31:'Wild animals crossing', 
    32:'End speed + passing limits', 
    33:'Turn right ahead', 
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left', 
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End no passing veh > 3.5 tons' 
}

def predict_traffic_sign(img_path):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(30, 30))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        model = load_model(MODEL_PATH)
        pred = model.predict(img_array)
        class_index = np.argmax(pred, axis=1)[0]
        return classes[class_index]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error predicting"

if __name__ == '__main__':
    print(f"Verifying model with image: {IMG_PATH}")
    prediction = predict_traffic_sign(IMG_PATH)
    print(f"Prediction: {prediction}")
    
    if prediction == 'Stop':
        print("SUCCESS: Model correctly predicted 'Stop'!")
    else:
        print(f"FAILURE: Model predicted '{prediction}' instead of 'Stop'.")
