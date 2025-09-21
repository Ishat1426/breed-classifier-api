from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# Initialize app
# -------------------------
app = Flask(__name__)

# Load your trained model
MODEL_PATH = "best_breed_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define image size (must match training size!)
IMG_SIZE = (224, 224)  # change if your model was trained with different size

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
        image = Image.open(file.stream).convert("RGB")
        image = image.resize(IMG_SIZE)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        predictions = model.predict(image_array)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
