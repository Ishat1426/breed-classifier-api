from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf

def preprocess_image(image_path):
    # 1. Load image as RGB, resize to 224x224
    img = load_img(image_path, target_size=(224, 224))  # RGB by default
    img_array = img_to_array(img)  # shape: (224,224,3)

    # 2. Handle grayscale images just in case
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)

    # 3. EfficientNet preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # 4. Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
