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
try:
    # A try-except block is good practice for model loading
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define image size (must match training size!)
# The error you showed previously indicated your model expects (225, 225)
# This is a critical detail to get right.
IMG_SIZE = (225, 225) 

# -------------------------
# Routes
# -------------------------

@app.route("/")
def home():
    return jsonify({"message": "Breed Classifier API is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if the request contains an image file
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Open the image using Pillow. 
        # The .convert("RGB") method handles the conversion for us,
        # making sure the image has 3 channels (Red, Green, Blue).
        image = Image.open(file.stream).convert("RGB")
        
        # Resize the image to match the size your model was trained on
        image = image.resize(IMG_SIZE)
        
        # Convert the image to a NumPy array and normalize pixel values to 0-1
        image_array = np.array(image) / 255.0
        
        # Add a batch dimension to the array.
        # Your model expects input in the shape (batch_size, height, width, channels).
        # We add 'axis=0' to create the batch dimension for a single image.
        image_array = np.expand_dims(image_array, axis=0)

        # Predict using the preprocessed image
        predictions = model.predict(image_array)
        
        # Get the class with the highest confidence
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        # Provide a more informative error message for debugging
        return jsonify({"error": f"Prediction failed: {str(e)}", "trace": "Make sure your model path is correct and the image is not corrupted."}), 500


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    if model:
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        print("Application will not run due to model loading error.")
