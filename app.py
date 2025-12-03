import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
MODEL_PATH = 'models/traffic_classifier.h5'
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Model
# We load it globally so we don't reload it on every request
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have trained the model first using train.py")
    model = None

# Classes dictionary to map prediction index to label
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_traffic_sign(img_path):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(30, 30))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # img_array = img_array / 255.0 # If we normalized during training, we must do it here too. 
        # Note: In train.py I didn't explicitly normalize by dividing by 255 in the load_data function 
        # (it just returns np.array(image)), but it's good practice. 
        # Let's assume the model learns from raw pixel values or we should normalize both places.
        # For now, let's keep it consistent with train.py (raw values).
        
        pred = model.predict(img_array)
        class_index = np.argmax(pred, axis=1)[0]
        return classes[class_index]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error predicting"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict
            if model:
                prediction = predict_traffic_sign(filepath)
            else:
                prediction = "Model not loaded"
                
            return render_template('index.html', prediction=prediction, image_path=filepath)
    
    return render_template('index.html', prediction=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=False, port=8080)
