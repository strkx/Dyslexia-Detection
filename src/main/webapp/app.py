from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os
import joblib
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Absolute paths to models
DYSLEXIA_MODEL_PATH = "C:/Users/Umair/OneDrive/Desktop/simple frontend/src/main/webapp/dyslexia_reg_model.pkl"
HANDWRITING_MODEL_PATH = "C:/Users/Umair/OneDrive/Desktop/simple frontend/src/main/webapp/final_model.keras"

# Load the Dyslexia Prediction Model (for /predict)
if not os.path.exists(DYSLEXIA_MODEL_PATH):
    print(f"❌ Error: Model file '{DYSLEXIA_MODEL_PATH}' not found!")
    dyslexia_model = None
else:
    try:
        dyslexia_model = joblib.load(DYSLEXIA_MODEL_PATH)
        print("✅ Dyslexia Prediction Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading Dyslexia Prediction Model: {e}")
        dyslexia_model = None

# Load the Handwriting Analysis Model (for /handwriting-analysis)
if not os.path.exists(HANDWRITING_MODEL_PATH):
    print(f"❌ Error: Model file '{HANDWRITING_MODEL_PATH}' not found!")
    handwriting_model = None
else:
    try:
        handwriting_model = tf.keras.models.load_model(HANDWRITING_MODEL_PATH)
        print("✅ Handwriting Analysis Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading Handwriting Analysis Model: {e}")
        handwriting_model = None

# Route for checking API status
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Dyslexia Prediction API is running!"})

# Route for Dyslexia Prediction (Numerical Input)
@app.route("/predict", methods=["POST"])
def predict():
    if dyslexia_model is None:
        return jsonify({"error": "Dyslexia Prediction Model not loaded. Check your model file."}), 500

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Expected feature names
        expected_features = [
            "Reading_Speed",
            "Spelling_Accuracy",
            "Writing_Errors",
            "Cognitive_Score",
            "Phonemic_Awareness_Errors",
            "Attention_Span",
            "Response_Time"
        ]

        # Check for missing features
        missing_features = [f for f in expected_features if f not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Convert input data to NumPy array
        try:
            inputs = np.array([[float(data[f]) for f in expected_features]])
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid input format. Ensure all values are numbers."}), 400

        # Make a prediction
        prediction = dyslexia_model.predict(inputs)

        # Handle scalar vs. array prediction output
        if isinstance(prediction, (np.ndarray, list)):
            prediction_value = float(prediction[0])  # Works if output is an array
        else:
            prediction_value = float(prediction)  # Works if output is a scalar

        return jsonify({"prediction": prediction_value})

    except Exception as e:
        print(f"❌ Error during Dyslexia prediction: {e}")
        return jsonify({"error": str(e)}), 500


# Route for Handwriting Analysis (Image Input)
@app.route("/handwriting-analysis", methods=["POST"])
def handwriting_analysis():
    if handwriting_model is None:
        return jsonify({"error": "Handwriting Analysis Model not loaded. Check your model file."}), 500

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        image_file = request.files["image"]

        # Convert FileStorage to a file-like object using BytesIO
        image_bytes = BytesIO(image_file.read())

        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(image_bytes, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = handwriting_model.predict(img_array)
        predicted_prob = float(prediction[0][0])

        # Binary classification: 0 = Dyslexic, 1 = Non-Dyslexic
        predicted_class = "Non_Dyslexic" if predicted_prob > 0.5 else "Dyslexic"

        return jsonify({
            "predicted_probability": predicted_prob,
            "predicted_class": predicted_class
        })

    except Exception as e:
        print(f"❌ Error during handwriting analysis: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True)
