import base64
import io
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET'])  # Homepage for uploading images
def upload_form():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])  # Image analysis route
def analyze_image():
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')  # Ensure RGB format
    image_array = np.array(image)  # Convert to NumPy array

    # Replace this with your actual image analysis logic
    # (using OpenCV, TensorFlow, or other suitable libraries)
    results = "Image analysis results will be displayed here."

    # Generate response image (optional, example using pillow)
    processed_image = Image.fromarray(image_array)  # Convert back to PIL image
    processed_image_byte_array = io.BytesIO()
    processed_image.save(processed_image_byte_array, format='PNG')
    processed_image_b64 = base64.b64encode(processed_image_byte_array.getvalue()).decode('utf-8')

    response = jsonify({'text_results': results, 'processed_image': processed_image_b64})
    response.headers['Content-Type'] = 'application/json'
    return response

if __name__ == '__main__':
    app.run(port=5000,debug=True)
