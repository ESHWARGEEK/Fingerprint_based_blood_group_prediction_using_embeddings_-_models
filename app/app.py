from flask import Flask, request, render_template
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

# --- App and Model Setup ---
app = Flask(__name__)
class_names = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "resnet50": "../models/ResNet50_best.pth",
    "efficientnet": "../models/EfficientNet_best.pth",
    "convnext": "../models/ConvNeXt_best.pth"
}

# --- Proper Model Loaders (same as training) ---
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
        raise FileNotFoundError(f"Model file not found: {path}")
    model = build_func(len(class_names))
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load trained models
resnet = load_model(MODEL_PATHS["resnet50"], build_resnet)
eff    = load_model(MODEL_PATHS["efficientnet"], build_efficientnet)
conv   = load_model(MODEL_PATHS["convnext"], build_convnext)
cnn_models = [resnet, eff, conv]

# CLIP setup
clip_model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=False)
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_model.eval()

# OracleDB setup
conn = oracledb.connect(user="vec_user", password="mypassword", dsn="localhost:1521/FREEPDB1")
cur = conn.cursor()

# --- Fingerprint Enhancement (CLAHE) ---
def enhance_fingerprint(pil_img):
    img = np.array(pil_img.convert("L"))  # grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    return Image.fromarray(enhanced).convert("RGB")  # back to RGB

# --- Preprocessing (match training pipeline) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])  # ImageNet normalization
])

# --- Blur Detection ---
def check_blur(image_pil, threshold=100.0):
    image_cv = np.array(image_pil.convert('RGB'))
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

# --- Ensemble CNN Prediction ---
def ensemble_predict(img_tensor):
    preds = []
    img_tensor = img_tensor.unsqueeze(0).to(device)
    for m in cnn_models:
        with torch.no_grad():
            out = m(img_tensor)
            preds.append(F.softmax(out, dim=1))
    avg_pred = torch.mean(torch.stack(preds), dim=0)
    return avg_pred

# --- CLIP Embedding Prediction ---
def embedding_predict(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb_list = clip_model.get_image_features(**inputs).squeeze().cpu().numpy().astype(np.float32).tolist()
    emb_array = array.array('f', emb_list)

    cur.execute("""
        SELECT label, VECTOR_DISTANCE(embedding, :1, COSINE) as distance
        FROM fingerprints
        ORDER BY distance ASC
        FETCH FIRST 1 ROWS ONLY
    """, [emb_array])
    
    result = cur.fetchone()
    if not result:
        return None, 0.0

    label, distance = result
    confidence = 1 - (distance / 2)  # Normalize distance to [0,1]
    return label, confidence

# --- Fusion Logic ---
def combine_predictions(cnn_label, cnn_conf, emb_label, emb_conf, w_cnn=0.65, w_emb=0.35):
    if emb_label is None:
        return cnn_label, cnn_conf

    # Case 1: Agreement
    if cnn_label == emb_label:
        final_label = cnn_label
        final_conf = w_cnn * cnn_conf + w_emb * emb_conf
    else:
        # Case 2: Disagreement → pick higher weighted confidence
        cnn_score = w_cnn * cnn_conf
        emb_score = w_emb * emb_conf
        if cnn_score >= emb_score:
            final_label, final_conf = cnn_label, cnn_score
        else:
            final_label, final_conf = emb_label, emb_score

    # Case 3: Low-confidence rejection
    if final_conf < 0.5:
        return None, final_conf

    return final_label, final_conf

# --- Flask Route ---
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_message = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == '':
            error_message = "Please upload a file."
        else:
            image = Image.open(file.stream).convert("RGB")
            image = enhance_fingerprint(image)  # 🔥 enhance before prediction

            # Blur detection AFTER enhancement
            is_blurry, variance = check_blur(image, threshold=80.0)  # slightly relaxed
            if is_blurry:
                error_message = f"Image is too blurry (Clarity Score: {variance:.2f}). Please use a clearer fingerprint."
            else:
                # CNN ensemble
                img_tensor = transform(image)
                avg_pred = ensemble_predict(img_tensor)
                cnn_label = class_names[avg_pred.argmax().item()]
                cnn_conf = avg_pred.max().item()

                # Embedding model
                emb_label, emb_conf = embedding_predict(image)

                # Combine predictions
                final_label, final_conf = combine_predictions(cnn_label, cnn_conf, emb_label, emb_conf)

                if final_label is None:
                    error_message = f"Low confidence ({final_conf:.2f}). Please upload a clearer fingerprint."
                else:
                    prediction = final_label
                    confidence = f"{final_conf*100:.2f}%"

    return render_template("index.html", prediction=prediction, confidence=confidence, error=error_message)

# --- Run Flask App ---
if __name__ == "__main__":
    app.run(port=8000, debug=True)
