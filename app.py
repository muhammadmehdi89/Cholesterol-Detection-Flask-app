import pytesseract
import PIL.Image
from PIL import UnidentifiedImageError
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, render_template, request
import cv2


moyconfig = r"--psm 6 --oem 3"

target_phrases = ["HDL-CHOLESTEROL", "LDL-CHOLESTEROL", "TRIGLYCERIDES"]

def perform_ocr(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Perform OCR on the image
    cropped_text = pytesseract.image_to_string(img, config=moyconfig)

    # Split the extracted text
    text = cropped_text.split()

    # Initialize an array to store extracted values
    array = [0, 0, 0]

    # Iterate over words to find target phrases and extract the next word
    for i, word in enumerate(text):
        for phrase in target_phrases:
            phrase_words = phrase.split()
            if i + len(phrase_words) - 1 < len(text):  # Ensure enough words remaining for the phrase
                if text[i:i+len(phrase_words)] == phrase_words:
                    next_word = text[i + len(phrase_words)]
                    try:
                        next_word = float(next_word)  # Convert to float
                    except ValueError:
                        pass  # Handle if conversion fails
                    if isinstance(next_word, float):
                        if phrase == "HDL-CHOLESTEROL":
                            array[0] = next_word
                        elif phrase == "LDL-CHOLESTEROL":
                            array[1] = next_word
                        elif phrase == "TRIGLYCERIDES":
                            array[2] = next_word
    return array

loaded_model = load_model("/path/to/directory/my_model_4.h5")

app = Flask(__name__)

@app.route('/')
@app.route('/about')
def index():
    about_section = request.path == '/about'
    return render_template('index.html', about_section=about_section, extracted_data=None)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/process_form', methods=['POST'])
def process_form():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part in the request', 400
        
        try:
            name = request.form['name']
            if not name:
                return "Please enter your name first", 400
            
            image_file = request.files['file']
            if image_file.filename == '':
                return 'No selected file', 400
            
            image_file.save('uploaded_image.jpg')
            extracted_data = perform_ocr('uploaded_image.jpg')

            array = np.array(extracted_data)
            array = array.reshape(1, -1)
            
            # Make predictions using the loaded model
            predictions = loaded_model.predict(array)
            predicted_class = np.argmax(predictions)
            if predicted_class == 0:
                predicted_class = "Boderline Cholesterol"
            elif predicted_class == 1:
                predicted_class = "High Cholesterol"
            else:
                predicted_class = "Normal/Optimal Cholesterol"
            return render_template('process_form.html', extracted_data=extracted_data, name=name, predicted_class=predicted_class)
        except UnidentifiedImageError:
            return "The uploaded file is not a valid image", 400
        
if __name__ == "__main__":
    app.run(debug=True)
