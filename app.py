from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load both models
try:
    fresh_rot_model = load_model("fresh_rot.h5")
    ripe_unripe_model = load_model("ripe_unripe.h5")
    print("✅ Models loaded successfully!")
except Exception as e:
    print("❌ Error loading models:", str(e))
    fresh_rot_model, ripe_unripe_model = None, None

# Preprocessing function
def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))  # Resize image
        img_array = img_to_array(img)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize pixel values
        return img_array
    except Exception as e:
        print("❌ Error preprocessing image:", str(e))
        return None

# Prediction function
def predict_image(image_path):
    try:
        if fresh_rot_model is None or ripe_unripe_model is None:
            return {"error": "Models not loaded"}

        img_array = preprocess_image(image_path)
        if img_array is None:
            return {"error": "Invalid Image"}

        # Fresh vs. Rot Prediction
        fresh_rot_pred = fresh_rot_model.predict(img_array)
        fresh_rot_class = ["Fresh", "Rot"][np.argmax(fresh_rot_pred)]
        fresh_rot_confidence = np.max(fresh_rot_pred) * 100

        # Ripe vs. Unripe Prediction
        ripe_unripe_pred = ripe_unripe_model.predict(img_array)
        ripe_unripe_class = ["Ripe", "Unripe"][np.argmax(ripe_unripe_pred)]
        print("🔍 Ripe/Unripe Prediction Raw Output:", ripe_unripe_pred)
        ripe_unripe_confidence = np.max(ripe_unripe_pred) * 100

        return {
            "fresh_rot_class": fresh_rot_class,
            "fresh_rot_confidence": float(fresh_rot_confidence),
            "ripe_unripe_class": ripe_unripe_class,
            "ripe_unripe_confidence": float(ripe_unripe_confidence),
        }
    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return {"error": "Prediction failed"}

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run predictions
        results = predict_image(filepath)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
