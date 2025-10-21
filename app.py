import os
import uuid
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import the main processing logic
import processing_service

# Load environment variables from .flaskenv
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
# Create a temporary directory for uploads
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

@app.route('/')
def index():
    """Render the simple HTML upload form."""
    return render_template('index.html')

@app.route('/process-pdf', methods=['POST'])
def process_pdf_endpoint():
    """The main API endpoint to process a PDF."""
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.endswith('.pdf'):
        # Create a unique temporary directory for this request
        request_id = str(uuid.uuid4())
        temp_dir = Path(app.config['UPLOAD_FOLDER']) / request_id
        temp_dir.mkdir()
        
        try:
            filename = secure_filename(file.filename)
            pdf_path = temp_dir / filename
            file.save(str(pdf_path))

            print(f"Processing request {request_id} for file {filename}...")
            
            # Call the main processing pipeline
            result_json = processing_service.process_tech_pack(str(pdf_path), str(temp_dir))

            print(f"✅ Successfully processed request {request_id}")
            return jsonify(result_json)

        except Exception as e:
            print(f"🔴 Error processing request {request_id}: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": "An internal error occurred", "details": str(e)}), 500
            
        finally:
            # Clean up the temporary directory after the request is complete
            if temp_dir.exists():
                # shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temporary directory: {temp_dir}")
                
    return jsonify({"error": "Invalid file type, please upload a PDF"}), 400

if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your local network
    app.run(debug=True, host='0.0.0.0')
