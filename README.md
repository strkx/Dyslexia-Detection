
<div align="center">

# Dyslexia Detection Project



</div>

This repository hosts the Dyslexia Early Detection System, an innovative solution that leverages cognitive tests, handwriting analysis, eye-tracking technology, and OCR to assess the risk of dyslexia. The project includes custom-built machine learning models for each of these features, ensuring high accuracy and efficiency.
- [Features](#features)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

🔹 Cognitive Tests – Evaluates reading speed, spelling accuracy, phonemic awareness, and response time.

🔹 Handwriting Analysis – Uses a CNN-based model to detect dyslexic handwriting patterns with high accuracy.

🔹 Eye-Tracking Technology – Analyzes reading patterns, including saccade analysis and blink rate determination.

🔹 OCR Tool – Converts book pages into text, applies the OpenDyslexic font for improved readability, and provides a text-to-speech option for accessibility.

🔹 Integrated Risk Assessment – Combines multiple evaluation methods for precise dyslexia detection.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/strkx/Dyslexia-Detection.git
    cd Dyslexia-Detection
    ```
2.  Navigate to the `src/main/webapp/my_ocr_app` directory.
3.  Install the necessary Python packages using pip:
    ```bash
    pip install flask flask_cors opencv-python pytesseract spellchecker textblob
    ```
    If you intend to use GPU with tensorflow, install tensorflow with GPU support.

## Running the Project

1.  Run the Flask API located in `src/main/webapp/app.py`:
    ```bash
    python src/main/webapp/app.py
    ```
2.  For the OCR functionality, navigate to `src/main/webapp/my_ocr_app` and run `ocr.py`:
    ```bash
    cd src/main/webapp/my_ocr_app
    python ocr.py
    ```
3.  Open the HTML files located in `src/main/webapp/` in your web browser to interact with the web interface. For example, open `index.html` to access the main page.

## Dependencies

*   Python 3.x
*   Flask
*   Flask-CORS
*   NumPy
*   TensorFlow
*   OpenCV
*   Joblib
*   Pytesseract
*   SpellChecker
*   TextBlob
*   Bootstrap 5
*   Font Awesome (optional, for web interface)

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear, descriptive messages.
4.  Submit a pull request to the main branch.

## Contact

*   Maintainer: [Umair Malim]
*   Contact Email: [malimumair2@gmail.com]
```
