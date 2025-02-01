import os
import cv2
import numpy as np
import pytesseract

from flask import Flask, request, jsonify
from flask_cors import CORS

# For spell-check correction
from spellchecker import SpellChecker

# For text simplification
from textblob import TextBlob

# =====================================
# 1) Setup and Initialization
# =====================================

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests if needed

# Point this to your Tesseract installation:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize the SpellChecker
spell = SpellChecker()


# =====================================
# 2) Helper Functions
# =====================================

def ocr_image(image_bytes):
    """
    Given raw image bytes, read with OpenCV and run Tesseract OCR.
    Returns the uncorrected text extracted by Tesseract.
    """
    # Convert bytes to a NumPy array so OpenCV can decode it
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        # Could not decode image
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize up to help with OCR accuracy
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Binarize image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Optional: Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

    # OCR configuration
    custom_config = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(blurred, config=custom_config)
    return text


def correct_ocr_text(text):
    """
    Correct OCR text using pyspellchecker, preserving line breaks.
    """
    corrected_lines = []
    lines = text.split("\n")

    for line in lines:
        words = line.split()
        corrected_words = []
        for w in words:
            # Spell-check each word
            best_guess = spell.correction(w)
            # If spell.correction() returns None, or if it's the same, keep original
            if best_guess is None or best_guess == w:
                corrected_words.append(w)
            else:
                corrected_words.append(best_guess)

        # Rebuild the line
        if line.strip():
            corrected_lines.append(" ".join(corrected_words))
        else:
            # Preserve blank lines
            corrected_lines.append("")

    return "\n".join(corrected_lines)


def simplify_text(text):
    """
    Simplify text using TextBlob. (TextBlob also performs correction,
    but generally focuses on grammar/spelling in context.)
    """
    blob = TextBlob(text)
    simplified = str(blob.correct())
    return simplified


# =====================================
# 3) Flask Routes
# =====================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "OCR API is running!"})


@app.route("/upload", methods=["POST"])
def upload_ocr():
    """
    Accepts an uploaded image (form-data, key='image'),
    runs OCR + correction, and returns JSON with the results.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        image_file = request.files["image"]
        image_bytes = image_file.read()

        # 1) Raw OCR
        raw_text = ocr_image(image_bytes)

        if not raw_text.strip():
            return jsonify({"error": "No text found by OCR."}), 200

        # 2) Spell-correction
        corrected = correct_ocr_text(raw_text)

        # 3) Simplification
        simplified = simplify_text(corrected)

        return jsonify({
            "original_text": raw_text,
            "corrected_text": corrected,
            "simplified_text": simplified
        }), 200

    except Exception as e:
        print("❌ Error in /upload:", e)
        return jsonify({"error": str(e)}), 500


# =====================================
# 4) Run the Flask App
# =====================================

if __name__ == "__main__":
    # Start Flask in debug mode (development)
    print("🚀 Starting OCR Flask server on http://127.0.0.1:5000")
    app.run(debug=True)
