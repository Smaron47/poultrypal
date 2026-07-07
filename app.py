import os
import io
import base64
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.cm as cm
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Global variables to store the active model and its specialized gradient graph
GLOBAL_MODEL = None
GRAD_MODEL = None
CLASS_LABELS = {0: "Coccidiosis", 1: "Healthy", 2: "New Castle Disease", 3: "Salmonella"} # Default placeholders

def rebuild_and_load_model(weights_path):
    """Rebuilds the exact topology structure and safely binds the saved weights."""
    global GLOBAL_MODEL, GRAD_MODEL
    
    keras.backend.clear_session()
    
    # 1. Rebuild the core architecture skeleton
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3)
    )
    base_model.trainable = False 

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(4, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    
    # 2. Extract and load the targeted weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights file found at target path: {weights_path}")
        
    model.load_weights(weights_path)
    GLOBAL_MODEL = model
    
    # 3. Intercept the outer graph connection to build the safe Grad-CAM model
    backbone_output_tensor = None
    for i, layer in enumerate(model.layers):
        if 'efficientnet' in layer.name.lower():
            backbone_output_tensor = model.layers[i+1].input
            break

    if backbone_output_tensor is None:
        raise ValueError("Could not trace EfficientNet backbone connections cleanly.")

    GRAD_MODEL = keras.Model(inputs=model.inputs, outputs=[backbone_output_tensor, model.output])
    print(f"--> Model successfully mounted from {weights_path}")

def get_base64_image(img_ndarray):
    """Converts a standard NumPy BGR/RGB image array directly into an HTML-ready Base64 string."""
    encoding_buffer = cv2.imencode('.png', img_ndarray)[1]
    base64_string = base64.b64encode(encoding_buffer).decode('utf-8')
    return f"data:image/png;base64,{base64_string}"

@app.route('/', methods=['GET'])
def index():
    # Displays the main user dashboard layout
    return render_template('index.html')

@app.route('/configure', methods=['POST'])
def configure():
    """Allows runtime configuration updates to the model path."""
    try:
        data = request.get_json()
        model_path = data.get('model_path', 'best_model_weights.h5')
        rebuild_and_load_model(model_path)
        return jsonify({"status": "success", "message": f"Successfully loaded weights: {os.path.basename(model_path)}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    """Ingests uploaded images, runs inference, and returns target Grad-CAM matrices."""
    global GLOBAL_MODEL, GRAD_MODEL
    CLASS_LABELS = [ "Coccidiosis", "Healthy", "New Castle Disease", "Salmonella"] # Default placeholders

    if GLOBAL_MODEL is None or GRAD_MODEL is None:
        return jsonify({"status": "error", "message": "Model not initialized. Please set a valid weights path first."}), 400
        
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename."}), 400

    try:
        # Ingest and format image into numeric arrays
        in_memory_file = io.BytesIO(file.read())
        pil_img = Image.open(in_memory_file).convert('RGB')
        
        # Keep original image size for mapping overlays cleanly later
        orig_w, orig_h = pil_img.size
        raw_cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Prepare target input sizes for EfficientNet
        resized_img = pil_img.resize((224, 224))
        img_array = np.array(resized_img).astype(np.float32)
        img_tensor = np.expand_dims(img_array, axis=0)
        
        # 1. Execute Gradient Tracking Graph
        with tf.GradientTape() as tape:
            last_conv_layer_outputs, predictions = GRAD_MODEL(img_tensor)
            pred_idx = tf.argmax(predictions[0])
            top_class_channel = predictions[:, pred_idx]

        # Calculate exact visual target matrices
        grads = tape.gradient(top_class_channel, last_conv_layer_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_outputs = last_conv_layer_outputs[0]
        heatmap = last_conv_layer_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU thresholds and scale outputs
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()
        
        # 2. Build the Color-Mapped Overlays
        heatmap_resized = cv2.resize(heatmap_np, (orig_w, orig_h))
        heatmap_colored = cm.jet(heatmap_resized)[..., :3] * 255
        heatmap_colored = cv2.cvtColor(heatmap_colored.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        alpha = 0.4
        superimposed_img = cv2.addWeighted(heatmap_colored, alpha, raw_cv_img, (1.0 - alpha), 0)
        
        # Extract metadata metrics
        confidence = float(predictions[0][pred_idx]) * 100
        predicted_label = f"Class  CLASS_LABELS{CLASS_LABELS[pred_idx]}" # Map dynamically to custom metadata strings if preferred
        
        return jsonify({
            "status": "success",
            "prediction": predicted_label,
            "confidence": f"{confidence:.2f}%",
            "original_image": get_base64_image(raw_cv_img),
            "gradcam_image": get_base64_image(superimposed_img)
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Optional auto-bootloader with an existing local file asset on startup
    default_weights = 'best_model_weights (1).h5'
    if os.path.exists(default_weights):
        try:
            rebuild_and_load_model(default_weights)
        except Exception as e:
            print(f"Initial setup skipped: {e}")
            
    app.run(debug=True, port=5000)

