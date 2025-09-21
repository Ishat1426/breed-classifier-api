from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# -------------------------
# Initialize app
# -------------------------
app = Flask(__name__)

# Load your trained model
MODEL_PATH = "best_breed_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------
# Load class names
# -------------------------
# You need to have a JSON file with class names saved during training
# Example: {"0": "Gir", "1": "Sahiwal", "2": "Jersey", ...}
CLASS_NAMES_PATH = "class_names.json"
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# -------------------------
# Define image size
# -------------------------
IMG_SIZE = (224, 224)

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return jsonify({"message": "Breed Classifier API is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        # Load image and ensure RGB
        image = Image.open(file.stream).convert("RGB")
        image = image.resize(IMG_SIZE)
        image_array = np.array(image)

        # EfficientNet preprocessing
        image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        predictions = model.predict(image_array)
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        # Map index to actual breed name
        predicted_breed = class_names[str(predicted_index)]

        return jsonify({
            "predicted_breed": predicted_breed,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
