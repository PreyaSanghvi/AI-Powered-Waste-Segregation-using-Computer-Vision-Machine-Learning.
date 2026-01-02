import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("waste_classifier_model.h5")

# Class names (same order as training)
class_names = [
    "biodegradable",
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic"
]

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]

    print("Predicted Waste Type:", class_names[class_index])
    print(f"Confidence: {confidence*100:.2f}%")

# Example usage
predict_image("waste.jpg")
