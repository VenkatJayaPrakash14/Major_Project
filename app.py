from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import torch
from ultralytics import YOLO
import uuid
import json

# Import the soil classifier and Gemini helper
from soil_classifier import SoilClassifier
from gemini_helper import GeminiHelper

app = Flask(__name__)
app.secret_key = "Enter Your Api Key Here"  # Set a secret key for session management

# Directory Configuration
UPLOAD_FOLDER = './static/uploads'
RESULTS_FOLDER = './static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create necessary directories
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)

# Model configuration
CORN_MODEL_PATH = './models/corn.pt'  # Path for corn analysis model

# Global models/services
soil_classifier = None
gemini_helper = None
corn_model = None

# Initialize soil classifier and Gemini services
def initialize_soil_services():
    global soil_classifier, gemini_helper
    try:
        # Initialize Soil Classifier
        if not os.path.exists('random_forest_model.pkl'):
            df = pd.read_csv('soil_dataset.csv')
            if 'pin_code' in df.columns:
                df = df.drop('pin_code', axis=1)
            soil_classifier = SoilClassifier()
            X, y = soil_classifier.preprocess_data(df)
            soil_classifier.train_model(X, y)
            soil_classifier.save_model()
        else:
            soil_classifier = SoilClassifier.load_model()
        
        # Initialize Gemini Helper
        os.environ["GEMINI_API_KEY"] = "AIzaSyBBVsw_PQhe2uDAuk9Hq64rhuiyC93rrnk" 
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        gemini_helper = GeminiHelper(gemini_api_key)
        
        return True, "Soil services initialized successfully!"
    except Exception as e:
        error_msg = f"Error initializing soil services: {str(e)}"
        print(error_msg)
        return False, error_msg

# Load YOLO model
def load_corn_model():
    global corn_model
    try:
        # Check if model file exists
        model_path = Path(CORN_MODEL_PATH)
        if not model_path.exists():
            print(f"Error: Corn model file not found at {CORN_MODEL_PATH}")
            return False, f"Corn model file not found at {CORN_MODEL_PATH}. Please check if file exists."
        
        # Force CPU usage to eliminate GPU issues
        device = 'cpu'  # Force CPU usage
        print(f"Using device: {device}")
        
        # Try to load the model with more detailed error handling
        print(f"Attempting to load corn model from {CORN_MODEL_PATH}")
        try:
            corn_model = YOLO(CORN_MODEL_PATH)
            # Explicitly move to device after loading
            corn_model.to(device)
        except Exception as e:
            print(f"Specific error during YOLO loading: {str(e)}")
            return False, f"YOLO loading error: {str(e)}"
        
        print(f"Corn model loaded: {corn_model}")
        
        # Verify model was loaded correctly
        if corn_model is None:
            print("Failed to load corn model (None value)")
            return False, "Corn model failed to load properly"
            
        # Test if model can be used for inference with a simple array
        try:
            dummy_input = torch.zeros((1, 3, 640, 640)).to(device)
            _ = corn_model.predict(source=dummy_input, verbose=False)
            print("Corn model test inference successful")
        except Exception as e:
            print(f"Corn model loaded but failed test inference: {str(e)}")
            return False, f"Corn model validation failed: {str(e)}"
            
        print("Corn model loaded successfully")
        return True, "Corn model loaded successfully"
        
    except Exception as e:
        print(f"Exception during corn model loading: {str(e)}")
        return False, f"Failed to load corn model: {str(e)}"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main landing page
@app.route('/')
def index():
    return render_template('index.html')

# Soil analysis routes
@app.route('/soil')
def soil_home():
    # Check if soil services are initialized
    global soil_classifier, gemini_helper
    if soil_classifier is None or gemini_helper is None:
        success, message = initialize_soil_services()
        if not success:
            flash(f"Warning: {message}", 'warning')
    return render_template('index_ml.html')

@app.route('/recommend')
def recommend():
    # Check if soil services are initialized
    global soil_classifier, gemini_helper
    if soil_classifier is None or gemini_helper is None:
        success, message = initialize_soil_services()
        if not success:
            flash(f"Warning: {message}", 'warning')
    return render_template('recommend.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if soil services are initialized
        global soil_classifier, gemini_helper
        if soil_classifier is None or gemini_helper is None:
            success, message = initialize_soil_services()
            if not success:
                flash(f"Error: {message}", 'error')
                return render_template('recommend.html')
        
        features = {
            'N': float(request.form['nitrogen']),
            'P': float(request.form['phosphorus']),
            'K': float(request.form['potassium']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall'])
        }
        
        # Input validation
        validation_rules = {
            'N': (0, 140, 'Nitrogen must be between 0 and 140 mg/kg'),
            'P': (0, 145, 'Phosphorus must be between 0 and 145 mg/kg'),
            'K': (0, 205, 'Potassium must be between 0 and 205 mg/kg'),
            'temperature': (0, 50, 'Temperature must be between 0°C and 50°C'),
            'humidity': (0, 100, 'Humidity must be between 0% and 100%'),
            'ph': (0, 14, 'pH must be between 0 and 14'),
            'rainfall': (0, 5000, 'Rainfall must be between 0 and 5000 mm')
        }
        
        # Validate each input
        for feature, (min_val, max_val, error_msg) in validation_rules.items():
            if not (min_val <= features[feature] <= max_val):
                flash(error_msg, 'error')
                return render_template('recommend.html')
        
        # Create DataFrame with only the required features
        input_df = pd.DataFrame([features])
        model_features = input_df[soil_classifier.feature_columns]
        
        if 'ph' in soil_classifier.feature_columns:
            model_features['ph'] = soil_classifier.scaler.transform(model_features[['ph']])
        
        recommended_crops = soil_classifier.predict_crops(model_features.iloc[0].to_dict(), threshold=0.1)
        
        if not recommended_crops:
            flash('No suitable crops found for the given parameters.', 'error')
            return render_template('recommend.html')
        
        # Get AI insights
        ai_insights = gemini_helper.get_crop_insights(recommended_crops, features)
        
        return render_template('result_ml.html', 
                             crops=recommended_crops,
                             ai_insights=ai_insights)
        
    except ValueError as ve:
        flash('Please enter valid numerical values for all fields.', 'error')
        return render_template('recommend.html')
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return render_template('recommend.html')

# Corn analysis routes
@app.route('/corn')
def corn_home():
    global corn_model
    if corn_model is None:
        success, message = load_corn_model()
        if not success:
            flash(f"Warning: {message}", 'warning')
    return render_template('index_corn.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global corn_model
    
    # Check if model is loaded
    if corn_model is None:
        success, message = load_corn_model()
        if not success:
            return jsonify({
                'success': False,
                'message': f'Corn model not loaded: {message}'
            }), 500
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Process the image
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Run prediction
            results_folder = Path(app.config['RESULTS_FOLDER'])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = results_folder / timestamp
            output_folder.mkdir(exist_ok=True)
            
            # Set default value for relative_output_path in case no results are found
            relative_output_path = None
            
            results = corn_model(
                source=file_path,
                save=True,
                project=str(output_folder),
                name='',  # Use empty name to avoid creating subdirectories
                conf=0.25,
                device=device
            )
            
            print(f"YOLO prediction completed, searching for output files...")
            
            # Look for jpg files at the top level only (non-recursive)
            image_files = list(output_folder.glob('*.jpg'))
            
            if image_files:
                # Use the first jpg file
                output_file = image_files[0]
                file_name = output_file.name
                
                # Create the correct web path
                relative_output_path = f'/static/results/{timestamp}/{output_file.name}'
                print(f"Found image: {output_file}, web path: {relative_output_path}")
            else:
                print("No jpg files found at the top level, trying recursive search...")
                # Try a recursive search as a fallback
                all_jpg_files = []
                for root, dirs, files in os.walk(str(output_folder)):
                    for file in files:
                        if file.endswith('.jpg'):
                            all_jpg_files.append(os.path.join(root, file))
                
                if all_jpg_files:
                    # Use the first jpg file found
                    jpg_path = Path(all_jpg_files[0])
                    # Get path relative to static folder
                    rel_path = os.path.relpath(jpg_path, 'static')
                    relative_output_path = f'/static/{rel_path}'.replace('\\', '/')
                    print(f"Found image with recursive search: {jpg_path}, web path: {relative_output_path}")
                else:
                    print("No jpg files found at all!")
                    relative_output_path = None
            
            # Process results
            results_list = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = box.cls[0].item()
                    class_name = corn_model.names[int(class_id)]
                    
                    results_list.append({
                        'class': class_name,
                        'confidence': round(confidence, 3),
                        'x1': round(x1, 2),
                        'y1': round(y1, 2),
                        'x2': round(x2, 2),
                        'y2': round(y2, 2)
                    })
            
            # Save results as JSON
            json_path = output_folder / 'results.json'
            with open(json_path, 'w') as f:
                json.dump(results_list, f)
            
            # Check if we found an image path
            if relative_output_path:
                return jsonify({
                    'success': True,
                    'message': 'Image processed successfully',
                    'image_path': relative_output_path,
                    'results': results_list,
                    'timestamp': timestamp
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Image processed but output file not found'
                }), 500
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({
                'success': False,
                'message': f"Error processing image: {str(e)}"
            }), 500
    
    return jsonify({
        'success': False,
        'message': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'
    }), 400

@app.route('/history')
def history():
    results_folder = Path(app.config['RESULTS_FOLDER'])
    history_data = []
    
    for folder in sorted(results_folder.glob('*'), reverse=True):
        if folder.is_dir():
            timestamp = folder.name
            json_file = folder / 'results.json'
            
            if json_file.exists():
                with open(json_file, 'r') as f:
                    results = json.load(f)
                
                # Search for images recursively
                all_jpg_files = []
                for root, dirs, files in os.walk(str(folder)):
                    for file in files:
                        if file.endswith('.jpg'):
                            all_jpg_files.append(os.path.join(root, file))
                
                if all_jpg_files:
                    # Get full path and create proper URL
                    image_file_path = all_jpg_files[0]
                    
                    # Convert absolute path to relative path from static folder
                    static_dir = os.path.abspath('./static')
                    rel_path = os.path.relpath(image_file_path, static_dir)
                    
                    # Create URL with proper path structure
                    image_path = f'/static/{rel_path}'.replace('\\', '/')
                    
                    # Get summary stats
                    class_counts = {}
                    for r in results:
                        class_name = r['class']
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                        else:
                            class_counts[class_name] = 1
                    
                    history_data.append({
                        'timestamp': timestamp,
                        'image_path': image_path,
                        'detection_count': len(results),
                        'classes': class_counts
                    })
    
    return render_template('history.html', history=history_data)

@app.route('/result/<timestamp>')
def result_detail(timestamp):
    results_folder = Path(app.config['RESULTS_FOLDER']) / timestamp
    json_file = results_folder / 'results.json'
    
    if json_file.exists():
        with open(json_file, 'r') as f:
            results = json.load(f)
        
        # Search for images recursively
        all_jpg_files = []
        for root, dirs, files in os.walk(str(results_folder)):
            for file in files:
                if file.endswith('.jpg'):
                    all_jpg_files.append(os.path.join(root, file))
        
        if all_jpg_files:
            # Get full path and create proper URL
            image_file_path = all_jpg_files[0]
            
            # Convert absolute path to relative path from static folder
            static_dir = os.path.abspath('./static')
            rel_path = os.path.relpath(image_file_path, static_dir)
            
            # Create URL with proper path structure
            image_path = f'/static/{rel_path}'.replace('\\', '/')
            
            return render_template('result_detail.html', 
                                  timestamp=timestamp, 
                                  image_path=image_path, 
                                  results=results)
    
    flash('Result not found')
    return redirect(url_for('history'))

# API endpoints
@app.route('/api/stats')
def get_stats():
    results_folder = Path(app.config['RESULTS_FOLDER'])
    all_results = []
    
    for folder in results_folder.glob('*'):
        if folder.is_dir():
            json_file = folder / 'results.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    results = json.load(f)
                    all_results.extend(results)
    
    # Generate statistics
    class_counts = {}
    confidence_by_class = {}
    
    for r in all_results:
        class_name = r['class']
        confidence = r['confidence']
        
        if class_name in class_counts:
            class_counts[class_name] += 1
            confidence_by_class[class_name].append(confidence)
        else:
            class_counts[class_name] = 1
            confidence_by_class[class_name] = [confidence]
    
    # Calculate average confidence per class
    avg_confidence = {}
    for cls, confs in confidence_by_class.items():
        avg_confidence[cls] = sum(confs) / len(confs)
    
    return jsonify({
        'total_detections': len(all_results),
        'class_distribution': class_counts,
        'average_confidence': avg_confidence
    })

@app.route('/check-model')
def check_model():
    """Route to check corn model status and provide diagnostics"""
    global corn_model
    
    # Basic file system checks
    model_path = Path(CORN_MODEL_PATH)
    file_exists = model_path.exists()
    file_size = model_path.stat().st_size if file_exists else 0
    
    # Model loading attempt if not already loaded
    if corn_model is None:
        success, message = load_corn_model()
    else:
        success, message = True, "Corn model already loaded"
    
    # Get available models directory structure
    models_dir = Path('./models')
    available_models = []
    if models_dir.exists() and models_dir.is_dir():
        available_models = [str(f) for f in models_dir.glob('*.pt')]
    
    return jsonify({
        'model_loaded': corn_model is not None,
        'model_path': CORN_MODEL_PATH,
        'file_exists': file_exists,
        'file_size_bytes': file_size,
        'available_models': available_models,
        'load_attempt': {
            'success': success,
            'message': message
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

# Utility routes for debugging
@app.route('/debug-files/<timestamp>')
def debug_files(timestamp):
    results_folder = Path(app.config['RESULTS_FOLDER']) / timestamp
    
    # Get file structure
    structure = []
    for root, dirs, files in os.walk(str(results_folder)):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, app.config['RESULTS_FOLDER'])
            structure.append({
                'full_path': full_path,
                'rel_path': rel_path,
                'filename': file
            })
    
    # Create HTML output
    html = '<html><body><h1>File Structure for ' + timestamp + '</h1>'
    html += '<ul>'
    for item in structure:
        html += f'<li><strong>Full path:</strong> {item["full_path"]}<br>'
        html += f'<strong>Relative path:</strong> {item["rel_path"]}<br>'
        html += f'<strong>Filename:</strong> {item["filename"]}<br>'
        html += f'<strong>Clean URL:</strong> /static/results/{timestamp}/{item["filename"]}'
        if item["filename"].endswith('.jpg'):
            html += f'<br><img src="/static/results/{timestamp}/{item["filename"]}" style="max-width:300px;border:1px solid #ccc;margin:10px 0;">'
        html += '</li><hr>'
    html += '</ul></body></html>'
    
    return html

@app.route('/image-test')
def image_test():
    results_folder = Path(app.config['RESULTS_FOLDER'])
    all_images = []
    
    # Find all jpg files in the results folder
    for jpg_file in results_folder.glob('**/*.jpg'):
        abs_path = str(jpg_file.absolute())
        
        # Get path relative to static folder
        static_dir = Path('./static').absolute()
        try:
            rel_to_static = jpg_file.absolute().relative_to(static_dir)
            web_url = f'/static/{rel_to_static}'.replace('\\', '/')
            
            all_images.append({
                'absolute_path': abs_path,
                'web_url': web_url
            })
        except ValueError:
            all_images.append({
                'absolute_path': abs_path,
                'web_url': 'ERROR: Not in static folder'
            })
    
    # Create a simple HTML page with all images
    html = '<html><body><h1>Image Path Test</h1>'
    html += '<p>Found ' + str(len(all_images)) + ' images</p>'
    html += '<ul>'
    
    for img in all_images:
        html += f'<li>'
        html += f'<p>Path: {img["absolute_path"]}</p>'
        html += f'<p>URL: {img["web_url"]}</p>'
        html += f'<img src="{img["web_url"]}" style="max-width: 300px; border: 1px solid #000;" />'
        html += '</li><hr>'
    
    html += '</ul></body></html>'
    
    return html

if __name__ == '__main__':
    # Try to initialize both services at startup
    soil_success, soil_message = initialize_soil_services()
    corn_success, corn_message = load_corn_model()
    
    print(f"Soil services: {soil_message}")
    print(f"Corn model: {corn_message}")
    
    if not soil_success:
        print("Warning: Soil classification services not initialized. Soil analysis functions will not work until services are correctly configured.")
    
    if not corn_success:
        print("Warning: Corn model not loaded. Corn seed analysis will not work until model is correctly configured.")
    
    app.run(debug=True)