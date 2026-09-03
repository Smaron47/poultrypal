# 🐔 PoultryPal — AI-Based Poultry Disease Detection

<p align="center">
  <strong>Deep-learning based poultry disease classification with Grad-CAM explainability</strong>
</p>

<p align="center">
  <em>Chicken image → EfficientNetB0 → disease classification → confidence score + Grad-CAM visualization</em>
</p>

---

## 📌 Overview

**PoultryPal** is an AI-powered computer-vision prototype for classifying poultry health conditions from images.

The application uses a TensorFlow/Keras **EfficientNetB0** backbone with a four-class softmax classification head. A Flask web application provides a browser-based interface for uploading an image and returns the predicted class, confidence score, original image, and a **Grad-CAM** visualization showing the regions that contributed to the prediction.

The current class mapping implemented by the application is:

| Index | Class |
|---:|---|
| 0 | Coccidiosis |
| 1 | Healthy |
| 2 | New Castle Disease |
| 3 | Salmonella |

The repository contains the Flask application, HTML template, trained Keras `.h5` weight files, and Python dependencies. citeturn1view0turn1view1

---

## ✨ Features

- 🐔 Poultry health image classification
- 🦠 Coccidiosis detection
- 🧫 New Castle Disease detection
- 🧪 Salmonella detection
- ✅ Healthy-class recognition
- 🤖 EfficientNetB0 image-classification backbone
- 🌐 Flask web application
- 📤 Browser-based image upload
- 📊 Prediction confidence
- 🔥 Grad-CAM explainability
- 🖼️ Original-image visualization
- 🌡️ Heatmap overlay visualization
- 🧠 Runtime model/weights configuration
- 💾 Keras `.h5` model-weight support
- 🚀 Gunicorn dependency for production-style serving

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────────┐
                    │       User / Farmer      │
                    │    Upload poultry image  │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │       Flask Web App       │
                    │        POST /predict      │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │      Image Processing     │
                    │                          │
                    │ • PIL image loading       │
                    │ • RGB conversion          │
                    │ • Resize 224 × 224        │
                    │ • Float32 tensor          │
                    │ • Batch dimension         │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │       EfficientNetB0      │
                    │     ImageNet backbone     │
                    │     Frozen feature base   │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Global Average Pooling   │
                    │         ↓                │
                    │       Dropout             │
                    │         ↓                │
                    │    Dense(4, Softmax)      │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │    Predicted Class       │
                    │    + Confidence          │
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          ┌───────────────────┐       ┌──────────────────┐
          │ Original Image    │       │     Grad-CAM     │
          │ Visualization     │       │ Heatmap Overlay  │
          └───────────────────┘       └──────────────────┘
```

---

# 🧠 Model Architecture

PoultryPal reconstructs an **EfficientNetB0** image-classification architecture:

```text
Input
224 × 224 × 3
     │
     ▼
EfficientNetB0
include_top=False
weights="imagenet"
     │
     ▼
GlobalAveragePooling2D
     │
     ▼
Dropout(0.2)
     │
     ▼
Dense(4, activation="softmax")
     │
     ▼
4-class prediction
```

The backbone is initialized with ImageNet weights and frozen during reconstruction:

```python
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

base_model.trainable = False
```

The classification head contains four softmax outputs. citeturn1view1

---

# 🦠 Supported Classes

PoultryPal currently maps the four model outputs to:

### 1. Coccidiosis

A poultry disease caused by protozoan parasites of the genus *Eimeria*.

### 2. Healthy

Images representing the healthy class used by the trained model.

### 3. New Castle Disease

A contagious viral disease affecting poultry.

### 4. Salmonella

A bacterial infection that can affect poultry and poultry production systems.

> The model's class names are labels learned from the training setup. A prediction should not be treated as a definitive veterinary diagnosis.

The exact class-index mapping in the current application is:

```python
CLASS_LABELS = {
    0: "Coccidiosis",
    1: "Healthy",
    2: "New Castle Disease",
    3: "Salmonella"
}
```

This mapping appears in the current source code. citeturn1view1

---

# 🔬 Image Preprocessing

The deployed Flask application prepares each uploaded image as follows:

```text
Uploaded Image
      │
      ▼
PIL.Image.open()
      │
      ▼
RGB conversion
      │
      ▼
Resize → 224 × 224
      │
      ▼
NumPy float32 array
      │
      ▼
Batch dimension
      │
      ▼
Model
```

The current implementation performs:

```python
pil_img = Image.open(in_memory_file).convert('RGB')

resized_img = pil_img.resize((224, 224))

img_array = np.array(resized_img).astype(np.float32)

img_tensor = np.expand_dims(img_array, axis=0)
```

Therefore the inference tensor has the shape:

```text
(1, 224, 224, 3)
```

and uses floating-point pixel values derived directly from the RGB image. citeturn1view1

### ⚠️ Preprocessing consistency

The preprocessing used during training should match the preprocessing used during deployment.

In particular, verify whether the training pipeline used:

```text
raw 0–255 pixels
```

or:

```text
normalized 0–1 pixels
```

or another EfficientNet-specific preprocessing scheme.

The current deployed code does **not** divide the input by 255 before inference. citeturn1view1

---

# 🔥 Grad-CAM Explainability

PoultryPal includes **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

This provides an approximate visual explanation of which image regions contributed to the predicted class.

The process is:

```text
Input Image
     │
     ▼
EfficientNet Feature Maps
     │
     ▼
Predicted Class
     │
     ▼
Gradient Calculation
     │
     ▼
Global Average of Gradients
     │
     ▼
Weighted Feature Maps
     │
     ▼
ReLU
     │
     ▼
Normalized Heatmap
     │
     ▼
Resize to Original Image
     │
     ▼
Heatmap Overlay
```

The implementation obtains the backbone feature output and model prediction from a specialized gradient model:

```python
GRAD_MODEL = keras.Model(
    inputs=model.inputs,
    outputs=[backbone_output_tensor, model.output]
)
```

Gradients are then calculated with `tf.GradientTape()`. citeturn1view1

---

# 🌡️ Heatmap Generation

After calculating the gradients, the application:

1. Computes the mean gradient over spatial dimensions.
2. Multiplies feature maps by the pooled gradients.
3. Applies ReLU.
4. Normalizes the heatmap.
5. Resizes it to the original image dimensions.
6. Applies a color map.
7. Blends the heatmap with the original image.

The current overlay uses:

```python
alpha = 0.4
```

for image/heatmap blending. citeturn1view1

The resulting visualization is returned as a Base64-encoded PNG.

---

# 📊 Prediction

The model generates four softmax scores:

```text
P(Coccidiosis)
P(Healthy)
P(New Castle Disease)
P(Salmonella)
```

The predicted class is selected using:

```python
pred_idx = tf.argmax(predictions[0])
```

and confidence is calculated as:

```python
confidence = float(predictions[0][pred_idx]) * 100
```

The application therefore reports the highest model probability as the displayed confidence percentage. citeturn1view1

### Example

```text
Prediction:
Coccidiosis

Confidence:
93.47%
```

> A softmax confidence score is not the same thing as clinically validated diagnostic certainty.

---

# 🌐 Web Application

PoultryPal uses Flask for its web server.

The main application object is:

```python
app = Flask(__name__)
```

The repository contains a `templates/` directory with:

```text
templates/
└── index.html
```

The homepage is served through:

```text
GET /
```

and the prediction endpoint is:

```text
POST /predict
```

The repository currently contains the Flask app and HTML template as separate components. citeturn1view0turn2view1

---

# 🔌 API Documentation

## `GET /`

Returns the PoultryPal web interface.

---

## `POST /predict`

Accepts an uploaded poultry image and returns the classification result.

### Request

The request must use:

```text
multipart/form-data
```

with the image field:

```text
image
```

Example:

```bash
curl -X POST \
  -F "image=@chicken.jpg" \
  http://127.0.0.1:5000/predict
```

### Successful response

```json
{
  "status": "success",
  "prediction": "Coccidiosis",
  "confidence": "93.47%",
  "original_image": "data:image/png;base64,...",
  "gradcam_image": "data:image/png;base64,..."
}
```

The current application returns:

- `status`
- `prediction`
- `confidence`
- `original_image`
- `gradcam_image`

as JSON. citeturn1view1

---

# ⚙️ Model Configuration

PoultryPal supports runtime model-weight configuration through:

```text
POST /configure
```

The request body is JSON:

```json
{
  "model_path": "best_model_weights.h5"
}
```

The endpoint rebuilds the EfficientNetB0 architecture and loads the specified weights.

Example:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d "{\"model_path\":\"best_model_weights.h5\"}" \
  http://127.0.0.1:5000/configure
```

A successful response is structured as:

```json
{
  "status": "success",
  "message": "Successfully loaded weights: best_model_weights.h5"
}
```

The current implementation performs this model reconstruction dynamically. citeturn1view1

---

# 📦 Model Files

The repository currently contains two Keras weight files:

```text
best_model_weights.h5
best_model_weights (1).h5
```

The application uses:

```text
best_model_weights (1).h5
```

as its default startup weight file when that file exists. citeturn1view0turn1view1

### Model loading sequence

```text
Application starts
       │
       ▼
Check default weights
       │
       ▼
Rebuild EfficientNetB0
       │
       ▼
Load .h5 weights
       │
       ▼
Create Grad-CAM graph
       │
       ▼
Model ready
```

---

# 📂 Repository Structure

The current GitHub repository contains:

```text
poultrypal/
│
├── templates/
│   └── index.html
│
├── app.py
├── best_model_weights.h5
├── best_model_weights (1).h5
├── requirements.txt
└── README.md
```

The repository currently has five commits and does not yet define a project description or website in its GitHub About section. citeturn1view0

---

# 🧩 File Responsibilities

| File | Purpose |
|---|---|
| `app.py` | Flask server, model reconstruction, inference, Grad-CAM, and API routes |
| `templates/index.html` | Web interface |
| `best_model_weights.h5` | Keras model weights |
| `best_model_weights (1).h5` | Default Keras model weights used by startup logic |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

# 📦 Dependencies

The current repository specifies:

```txt
Flask==3.0.2
tensorflow
numpy==1.26.4
opencv-python-headless==4.9.0.80
Pillow==10.2.0
matplotlib==3.8.3
gunicorn==21.2.0
```

The project deliberately uses `opencv-python-headless`, which avoids unnecessary GUI dependencies in server environments. The requirements also include Gunicorn for production-style WSGI serving. citeturn2view0

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/Smaron47/poultrypal.git
cd poultrypal
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

Start the Flask application:

```bash
python app.py
```

The current application runs on:

```text
http://127.0.0.1:5000
```

The Flask development server is configured with:

```python
app.run(debug=True, port=5000)
```

The default model weights are automatically loaded when the expected `.h5` file is available. citeturn1view1

---

# 🖥️ Using PoultryPal

### Step 1

Open:

```text
http://127.0.0.1:5000
```

### Step 2

Upload a poultry image.

### Step 3

Submit the image for analysis.

### Step 4

The backend:

```text
Loads image
   ↓
Converts to RGB
   ↓
Resizes to 224×224
   ↓
Runs EfficientNetB0
   ↓
Selects highest-probability class
   ↓
Calculates Grad-CAM
   ↓
Returns result
```

### Step 5

The client receives:

```text
Prediction
Confidence
Original image
Grad-CAM image
```

---

# 🔬 Example Prediction Flow

Suppose the model returns:

```text
[0.91, 0.02, 0.04, 0.03]
```

The class mapping gives:

```text
0 → Coccidiosis
1 → Healthy
2 → New Castle Disease
3 → Salmonella
```

Therefore:

```text
Prediction:
Coccidiosis

Confidence:
91%
```

The Grad-CAM target is the same predicted class.

---

# 🧪 Model Evaluation

This repository currently provides the inference application and model weights, but a verified held-out test-set evaluation is not documented in the repository content inspected for this README.

For a research-grade release, report at minimum:

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- Per-class performance

For this four-class problem, report performance separately for:

```text
Coccidiosis
Healthy
New Castle Disease
Salmonella
```

Do not rely only on overall accuracy, especially if the real-world class distribution is imbalanced.

---

# ⚠️ Important Limitations

## 1. Image quality

Blurred, poorly illuminated, obstructed, or atypical images can produce unreliable predictions.

## 2. Dataset generalization

A model trained on a particular image distribution may not perform identically on images from different:

- Farms
- Breeds
- Cameras
- Lighting conditions
- Backgrounds
- Ages
- Disease stages

## 3. Class mapping

The numerical class order must remain synchronized with the trained model.

If the training mapping changes, `CLASS_LABELS` must change accordingly.

## 4. Confidence interpretation

A high softmax score does not guarantee that the prediction is correct.

## 5. Grad-CAM interpretation

Grad-CAM highlights image regions associated with the model's prediction. It does not prove that the highlighted region is biologically responsible for the disease.

---

# 🩺 Veterinary Disclaimer

**PoultryPal is an experimental AI-based screening and research prototype.**

It is not a certified veterinary diagnostic system.

The predictions should not replace:

- Veterinary examination
- Laboratory testing
- Necropsy where appropriate
- Microbiological testing
- Clinical history
- Professional poultry-health assessment

Any suspected disease should be evaluated by an appropriately qualified veterinary professional.

---

# 🔐 Production & Security Considerations

The current application is configured for development:

```python
app.run(debug=True, port=5000)
```

Before public deployment:

- Disable Flask debug mode.
- Use Gunicorn or another production WSGI server.
- Put the application behind HTTPS.
- Validate uploaded files.
- Limit image size.
- Reject unsupported formats.
- Avoid exposing filesystem paths.
- Add authentication/rate limiting where appropriate.
- Avoid storing sensitive farm/customer data unnecessarily.
- Monitor inference failures.
- Version the model together with preprocessing and class mapping.

---

# 🚀 Production Example

A production architecture can be:

```text
                    Internet
                       │
                       ▼
                ┌─────────────┐
                │    HTTPS    │
                │ Reverse Proxy│
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │   Gunicorn  │
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │    Flask    │
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │ EfficientNet│
                │    Model    │
                └─────────────┘
```

---

# 🔮 Future Roadmap

Potential improvements include:

- [ ] Verify and document the final training dataset
- [ ] Add a dedicated model-training repository/script
- [ ] Add a reproducible train/validation/test split
- [ ] Add test-set metrics
- [ ] Add confusion matrix to the web application
- [ ] Add per-class probability visualization
- [ ] Add automatic image-quality detection
- [ ] Add batch image prediction
- [ ] Add real-time camera inference
- [ ] Add poultry/bird detection before disease classification
- [ ] Add mobile deployment
- [ ] Convert the model to TensorFlow Lite for edge devices
- [ ] Add farm-level prediction history
- [ ] Add database-backed monitoring
- [ ] Add API authentication
- [ ] Add Docker support
- [ ] Add automated tests
- [ ] Add CI/CD
- [ ] Improve Grad-CAM visualization
- [ ] Add calibrated probabilities
- [ ] Perform external validation on independent farm images

---

# 🧪 Research & Reproducibility

For reproducible experiments, keep these components synchronized:

```text
Dataset
   │
   ▼
Dataset split
   │
   ▼
Class mapping
   │
   ▼
Image preprocessing
   │
   ▼
EfficientNet architecture
   │
   ▼
Training configuration
   │
   ▼
Saved weights
   │
   ▼
Inference code
   │
   ▼
Evaluation
```

A change to any one of these components can affect the resulting predictions.

For academic work, record:

- Dataset version
- Number of images
- Class distribution
- Train/validation/test split
- Random seed
- Image preprocessing
- Model architecture
- Optimizer
- Learning rate
- Batch size
- Number of epochs
- Data augmentation
- Class weighting
- Evaluation metrics
- Model version/hash

---

# 🔏 Data & Privacy

If PoultryPal is used with real farm images:

- Avoid committing private images to GitHub.
- Remove unnecessary metadata.
- Protect uploaded files.
- Define an image-retention policy.
- Avoid exposing uploads publicly.
- Restrict API access when appropriate.

This is particularly important if the application is later integrated into commercial farm-management systems.

---

# 📚 Dataset / Training Data

The current GitHub repository does not document a specific dataset source in its README or project metadata.

Therefore, this README intentionally does **not** claim that a particular external dataset trained the committed model.

If the model was trained using a publicly available dataset, add the exact dataset URL, license, class distribution, preprocessing procedure, and citation here before using the project in academic or commercial work.

---

# 👨‍💻 Project

**PoultryPal**

GitHub:

urlSmaron47/poultrypalhttps://github.com/Smaron47/poultrypal

The repository is public and currently contains the Flask application, template, model-weight files, and dependency configuration. citeturn1view0

---

# ⭐ Contributing

Contributions are welcome.

Useful contribution areas include:

- Model improvements
- Better preprocessing
- Explainable AI
- Dataset documentation
- Evaluation
- Frontend improvements
- API development
- Mobile/edge deployment
- Testing
- Documentation

Typical workflow:

```bash
git clone https://github.com/Smaron47/poultrypal.git
cd poultrypal

git checkout -b feature/your-feature
```

Make your changes, test them, and open a pull request.

---

# 📄 License

No explicit software license is currently documented in the repository.

Before redistributing or commercially deploying PoultryPal, add an appropriate project license and verify the licenses/terms associated with the training data and model weights.

---

<p align="center">
  <strong>🐔 PoultryPal</strong><br>
  AI-assisted poultry disease classification with explainable computer vision
</p>
