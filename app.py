import base64
import io
from flask import Flask, request, jsonify, render_template
from model.supermarket import run_algorithm
import numpy as np
from PIL import Image
import os    
import base64
import numpy as np
from werkzeug.utils import secure_filename



app = Flask(__name__)

@app.route('/', methods=['GET'])  # Homepage for uploading images
def upload_form():
    return render_template('index.html')


# @app.route('/analyze', methods=['POST'])  # Image analysis route
# def analyze_image():
#     image_file = request.files['image']
#     image = Image.open(image_file).convert('RGB')  # Ensure RGB format
#     image_array = np.array(image)  # Convert to NumPy array

#     # Store the image in a folder
#     image_filename = 'uploaded_image.png'
#     # Generate a unique filename for each uploaded image
#     image_path = os.path.join('uploaded', image_filename)
#     image.save(image_path)

#     # Get the selected product from the dropdown
#     selected_product = request.form.get('product')

#     # Select the corresponding image based on the selected product
#     if selected_product == 'Fanta':
#         product_image_path = 'data_/products/c1_product.jpg'
#     elif selected_product == 'Rinso':
#         product_image_path = 'data_/products/t1_product.jpg'
#     elif selected_product == 'Bingo':
#         product_image_path = 'data_/products/b1_product.jpg'
#     elif selected_product == 'Stayfree':
#         product_image_path = 'data_/products/test_product3.jpg'
#     else:
#         # Handle invalid product selection
#         return jsonify({'error': 'Invalid product selection'})

#     # Perform image analysis here (replace with your own algorithm)
#     normalized_distance_y, percentage_area, matched_image, lighting_conditions  = run_algorithm(product_image_path, image_path)

#     results = {
#         'normalized_distance_y': normalized_distance_y,
#         'percentage_area': percentage_area,
#         'lighting_conditions': lighting_conditions,
#         'matched_image': matched_image
#     }

#     # Generate response image (optional, example using pillow)
#     processed_image = Image.fromarray(image_array)  # Convert back to PIL image
#     processed_image_byte_array = io.BytesIO()
#     processed_image.save(processed_image_byte_array, format='PNG')
#     processed_image_b64 = base64.b64encode(processed_image_byte_array.getvalue()).decode('utf-8')


#     # Assuming processed_image_b64 is a Base64-encoded string
#     decoded_bytes = base64.b64decode(processed_image_b64)
#     processed_image_array = np.frombuffer(decoded_bytes, dtype=np.uint8)

#     # Now you can call tolist()
#     processed_image_list = processed_image_array.tolist()
#     response = jsonify({'results': results, 'processed_image': processed_image_list})


#     response.headers['Content-Type'] = 'application/json'
#     return response

@app.route('/analyze', methods=['GET','POST'])
def analyze_image():
    try:
        # Validate and handle image file
        image_file = request.files['image']
        if not image_file:
            raise ValueError("No image file uploaded")
        if not image_file.filename or not '.' in image_file.filename:
            raise ValueError("Invalid image file format")

        # Ensure RGB format
        image = Image.open(image_file).convert('RGB')

        # Store the image with a unique filename
        image_filename = secure_filename(image_file.filename)
        image_path = os.path.join('uploaded', image_filename)
        image.save(image_path)

        # Get the selected product
        selected_product = request.form.get('product')
        PRODUCT_IMAGES = {
            'Fanta': 'data_/products/c1_product.jpg',
            'Rinso': 'data_/products/t1_product.jpg',
            'Bingo': 'data_/products/b1_product.jpg',
            'Stayfree': 'data_/products/test_product3.jpg'
        }

        if not selected_product or selected_product not in PRODUCT_IMAGES:
            raise ValueError("Invalid product selection")

        # Select the corresponding product image path
        product_image_path = PRODUCT_IMAGES[selected_product]

        # Perform image analysis (using placeholder function for now)
        normalized_distance_y, percentage_area, matched_image_base64, lighting_conditions,incentive_message = run_algorithm(product_image_path, image_path)

        # Prepare response data
        results = {
            'normalized_distance_y': normalized_distance_y,
            'percentage_area': percentage_area,
            'lighting_conditions': lighting_conditions,
            'matched_image': matched_image_base64,  # Change variable name
            'incentive_message': incentive_message
        }

        # Generate response image (optional)
        if request.args.get('return_processed_image'):
            processed_image_byte_array = io.BytesIO()
            processed_image = Image.fromarray(np.array(image))  # Convert back to PIL image
            processed_image.save(processed_image_byte_array, format='PNG')
            processed_image_b64 = base64.b64encode(processed_image_byte_array.getvalue()).decode('utf-8')
            results['processed_image'] = processed_image_b64
        else:
            results['matched_image'] = matched_image_base64  # Change variable name

        response = jsonify(results)
        response.headers['Content-Type'] = 'application/json'
        return response

    except Exception as e:
        error_message = str(e)
        return jsonify({'error': error_message}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,use_reloader=False, threaded=True)  # Multithreaded

