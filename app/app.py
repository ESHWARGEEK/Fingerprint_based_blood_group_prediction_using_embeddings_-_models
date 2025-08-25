from flask import Flask, request, render_template, redirect, url_for, flash
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import oracledb
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
import array
import cv2
import io
import base64
import traceback

# --- App and Model Setup ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key_change_this' 
class_names = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

last_embedding = None

MODEL_PATHS = {
    "resnet50": "../models/ResNet50_best.pth",
    "efficientnet": "../models/EfficientNet_best.pth",
    "convnext": "../models/ConvNeXt_best.pth"
}

# --- Model Loading and Helper Functions (No Changes Here) ---
def build_resnet(num_classes):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def build_efficientnet(num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def build_convnext(num_classes):
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    return model

def load_model(path, build_func):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}. Please check the MODEL_PATHS variable.")
    model = build_func(len(class_names))
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

try:
    resnet = load_model(MODEL_PATHS["resnet50"], build_resnet)
    eff    = load_model(MODEL_PATHS["efficientnet"], build_efficientnet)
    conv   = load_model(MODEL_PATHS["convnext"], build_convnext)
    cnn_models = [resnet, eff, conv]
except FileNotFoundError as e:
    print(f"FATAL ERROR: {e}")
    exit()

clip_model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=False)
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_model.eval()

try:
    conn = oracledb.connect(user="vec_user", password="mypassword", dsn="localhost:1521/FREEPDB1")
    cur = conn.cursor()
except oracledb.Error as e:
    print(f"FATAL ERROR: Could not connect to Oracle Database: {e}")
    exit()

def convert_to_binary(pil_img, threshold=120):
    grayscale_img = pil_img.convert('L')
    binary_img = grayscale_img.point(lambda p: 255 if p > threshold else 0, '1')
    return binary_img.convert('RGB')

def enhance_fingerprint(pil_img):
    img = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    return Image.fromarray(enhanced).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def check_blur(image_pil, threshold=100.0):
    image_cv = np.array(image_pil.convert('RGB'))
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

def ensemble_predict(img_tensor):
    preds = []
    img_tensor = img_tensor.unsqueeze(0).to(device)
    for m in cnn_models:
        with torch.no_grad():
            out = m(img_tensor)
            preds.append(F.softmax(out, dim=1))
    avg_pred = torch.mean(torch.stack(preds), dim=0)
    return avg_pred, preds

def get_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb_list = clip_model.get_image_features(**inputs).squeeze().cpu().numpy().astype(np.float32).tolist()
    return array.array('f', emb_list)

def embedding_predict(emb_array):
    cur.execute("""
        SELECT label, VECTOR_DISTANCE(embedding, :1, COSINE) as distance
        FROM fingerprints ORDER BY distance ASC FETCH FIRST 1 ROWS ONLY
    """, [emb_array])
    result = cur.fetchone()
    if not result:
        return None, 0.0
    label, distance = result
    confidence = 1 - (distance / 2)
    return label, confidence

def store_embedding(label, emb_array):
    try:
        cur.execute("INSERT INTO fingerprints (label, embedding) VALUES (:1, :2)", [label, emb_array])
        conn.commit()
        return True
    except oracledb.Error as e:
        print(f"Database error while storing embedding: {e}")
        conn.rollback()
        return False

def combine_predictions(cnn_label, cnn_conf, emb_label, emb_conf):
    if cnn_conf < 0.50: return None, cnn_conf
    if cnn_conf > 0.75: return cnn_label, cnn_conf
    HIGH_CONF_THRESHOLD, MEDIUM_CONF_THRESHOLD, REJECTION_THRESHOLD = 0.90, 0.70, 0.50
    if emb_label is None: return (cnn_label, cnn_conf) if cnn_conf >= REJECTION_THRESHOLD else (None, cnn_conf)
    if cnn_label == emb_label: return cnn_label, (cnn_conf + emb_conf) / 2
    if emb_conf > HIGH_CONF_THRESHOLD: return emb_label, emb_conf
    if cnn_conf > MEDIUM_CONF_THRESHOLD and emb_conf > MEDIUM_CONF_THRESHOLD: return None, (cnn_conf + emb_conf) / 2
    final_label, final_conf = (cnn_label, cnn_conf) if cnn_conf > emb_conf else (emb_label, emb_conf)
    if final_conf < REJECTION_THRESHOLD: return None, final_conf
    return final_label, final_conf

# --- Main Flask Route ---
@app.route("/", methods=["GET", "POST"])
def index():
    global last_embedding
    prediction, error_message, confidence, cnn_details, emb_details, image_data, ensemble_details = (None,) * 7

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == '':
            error_message = "Please upload a file."
        else:
            try:
                original_image = Image.open(file.stream).convert("RGB")
                binary_image = convert_to_binary(original_image)
                enhanced_image = enhance_fingerprint(binary_image)

                buffered = io.BytesIO()
                enhanced_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_data = img_str

                is_blurry, variance = check_blur(enhanced_image, threshold=80.0)
                
                img_tensor = transform(enhanced_image)
                avg_pred, individual_preds = ensemble_predict(img_tensor)
                cnn_label = class_names[avg_pred.argmax().item()]
                cnn_conf = avg_pred.max().item()
                
                embedding_vector = get_embedding(enhanced_image)
                last_embedding = embedding_vector
                emb_label, emb_conf = embedding_predict(embedding_vector)

                if is_blurry and cnn_conf < 0.5 and emb_conf < 0.5:
                    error_message = "Please upload a clearer image. (Low Clarity & Confidence)"
                else:
                    model_names = ["ResNet50", "EfficientNet", "ConvNeXt"]
                    ensemble_details = []
                    for name, pred_tensor in zip(model_names, individual_preds):
                        pred_label = class_names[pred_tensor.argmax().item()]
                        pred_conf = pred_tensor.max().item()
                        ensemble_details.append(f"└ {name}: {pred_label} ({pred_conf*100:.2f}%)")

                    cnn_details = f"Ensemble Prediction: {cnn_label} ({cnn_conf*100:.2f}%)"
                    emb_details = f"Embedding Prediction: {emb_label} ({emb_conf*100:.2f}%)" if emb_label else "Embedding Prediction: No close match found."

                    final_label, final_conf = combine_predictions(cnn_label, cnn_conf, emb_label, emb_conf)
                    
                    if final_label is None:
                        error_message = f"Prediction is uncertain (Confidence: {final_conf*100:.2f}%). Please upload a clearer fingerprint."
                    else:
                        prediction = final_label
                        confidence = f"{final_conf*100:.2f}%"
            except Exception as e:
                print("--- AN ERROR OCCURRED ---")
                traceback.print_exc()
                error_message = "An internal error occurred. Please check the server logs for details."
    
    # --- FIXED: Explicitly pass variables to the template with the correct names ---
    return render_template("index.html", 
                           prediction=prediction, 
                           confidence=confidence, 
                           error=error_message,  # This maps error_message to 'error' in the template
                           cnn_details=cnn_details,
                           emb_details=emb_details,
                           ensemble_details=ensemble_details,
                           image_data=image_data,
                           class_names=class_names) # Pass class_names for the correction dropdown

# --- Route to handle user corrections ---
@app.route("/correct", methods=["POST"])
def correct():
    global last_embedding
    correct_label = request.form.get("correct_label")
    
    if correct_label and last_embedding is not None:
        if store_embedding(correct_label, last_embedding):
            flash(f"Thank you! The fingerprint has been added to the database as '{correct_label}'.", "success")
        else:
            flash("Sorry, there was a database error. The correction was not saved.", "danger")
    else:
        flash("Could not save correction. No previous image data found.", "warning")
        
    return redirect(url_for('index'))

# --- Run Flask App ---
if __name__ == "__main__":
    app.run(port=8000, debug=True)
