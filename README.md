# 🧬 Fingerprint-Based Blood Group Detection using Deep Learning & Embeddings  
<img src="/results/Screenshot 2025-08-20 202737.png">
<img src="/results/Screenshot 2025-08-20 202758.png>


This project demonstrates the use of **fingerprint biometrics** and **deep learning** to predict a person’s blood group. The system combines **CNN-based classification models** with **fingerprint embeddings** stored in a **vector database** for efficient retrieval and similarity search.  

---

## 🚀 Features  
- **Multi-class Classification**: Detects **8 blood groups** (A+, A-, B+, B-, AB+, AB-, O+, O-).  
- **Deep Learning Models**: Implemented **ResNet50, VGG16, ConvNeXt**, and an **ensemble** for improved accuracy.  
- **Dataset Expansion**: Increased dataset size from **8K → 50K+ images** using augmentation.  
- **Embeddings Integration**: Extracted fingerprint embeddings and stored in **Oracle Vector DB (IVFFlat, cosine similarity)**.  
- **Deployment**: Real-time predictions via **Flask/Streamlit web app** with fingerprint upload.  
- **Performance**: Achieved **60%+ test accuracy** across 8 blood group classes.  

---

## 🛠️ Tech Stack  
- **Languages & Frameworks**: Python, TensorFlow/Keras, PyTorch, OpenCV  
- **Database**: Oracle Database with Vector Search  
- **Deployment**: Flask, Streamlit  
- **Libraries**: NumPy, Pandas, Matplotlib, Scikit-learn  

---

## 📂 Project Structure  
fingerprint-blood-group-detection/
|
|---data/#sample dataset/test images
|---notebooks/#jupyter notebooks for training & experiments
| |---data_preprocessing.ipynb
| |---model_training.ipynb
| |---embeddings.ipynb
|
|---src/#source code
| |---data_loader.py
| |---models_ensemble.py
| |---embeddings.py
|
|---docs
| |---architecture.png
| |---workflow.png
| |---results.png
|
|---requirements
|---README.md # project overview


---

## ⚡ Installation & Usage  

### 1. Clone the repository  
```bash
git clone https://github.com/username/fingerprint-blood-group-detection.git
cd fingerprint-blood-group-detection
```

### 2. Install dependices
```bash
pip install -r requirements.txt
```

### 3.Train the model
```bash
python src/train.py
```

###4 Run the web app
```bash
cd app
python app.py
```
###Acess the app at http://127.0.0.1:5000/

##📊Results
- **Achieved** > **75% accuracy** on 8-class fingerprint blood group classification.
- Embeddings stored in **Oracle DB** to enable **fast similarity search**
- Results are available at the docs/

##🔮Future Work
- Improve accuracy using Vision Transformers (ViTs) and attention-based models.
- Collect larger, more diverse fingerprint datasets.
- Extend to multimodal biometric + health record integration.

🤝 Acknowledgements

- TensorFlow/Keras & PyTorch for deep learning frameworks
- Oracle for vector database integration
- Fingerprint datasets used for experimentation


