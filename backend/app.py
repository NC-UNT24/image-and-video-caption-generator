# Import additional modules at the top
from werkzeug.utils import secure_filename
from models import caption_generator
import cv2

from flask import Flask, request, jsonify
from flask_cors import CORS
import os


app = Flask(__name__)
CORS(app)

# Set the upload folder
UPLOAD_FOLDER = 'uploads/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Allowed extensions for video uploads
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/generate-image-caption', methods=['POST'])
def generate_image_caption():
    language = request.form.get('language', 'en')
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']

    # If user does not select a file
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        # Pass the language to generate_caption
        caption = caption_generator.generate_caption(filepath, language=language)

        return jsonify({'caption': caption}), 200
    else:
        return jsonify({'error': 'Invalid file type.'}), 400

@app.route('/api/generate-video-caption', methods=['POST'])
def generate_video_caption():
    language = request.form.get('language', 'en')

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if file and '.' in file.filename and \
            file.filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads/videos', filename)
        os.makedirs('uploads/videos', exist_ok=True)
        file.save(filepath)

        # Generate captions for video with language
        captions = caption_generator.generate_captions_for_video(filepath, language=language)

        return jsonify({'captions': captions}), 200
    else:
        return jsonify({'error': 'Invalid file type.'}), 400

if __name__ == '__main__':
    app.run(debug=True)