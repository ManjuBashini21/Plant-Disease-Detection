# Plant Disease Detection

### Overview
A web-based application for detecting plant leaf diseases using a Flask (Python) backend and a deep learning model (PyTorch).  
This project is designed for early detection to assist farmers, researchers, and agricultural experts in improving crop health.

---

### 🛠 Tech Stack
- **Frontend:** HTML5, CSS3  
- **Backend:** Python 3, Flask  
- **Deep Learning Framework:** PyTorch, Torchvision  
- **Image Processing:** Pillow (PIL)  
- **Model Architecture:** Custom ResNet9 CNN  
- **Dataset:** PlantVillage (38 classes)

---

### ⚙ Project Setup
```bash
# 1. Clone repository
git clone <your-repo-link>
cd plant-disease-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows

# 3. Install Dependencies
Run the following commands one by one:

```bash
pip install flask
pip install torch
pip install torchvision
pip install pillow

# 4. Place trained .pth model in project directory

# 5. Run the Flask app
python app.py
```

### Workflow
- **Upload**: JPEG/PNG (max 10MB).
- **Preprocess**: Adjust to model input size.
- **Inference**: ResNet9 predicts disease class.
- **Display**: Show image with predicted label.

### Dataset
- **Source**: PlantVillage dataset.
- **Content**: Healthy & diseased leaves (apple, grape, potato, tomato, corn, etc.).
- **Categories**: 38.
- **Preprocessing**: Consistent preprocessing applied to all images.

### Disease Detection Architecture
- **Model**: ResNet9 CNN with residual connections.
- **Detection Features**: Color, spots, lesions, and texture.
- **Performance**: High accuracy & low latency for near real-time use.

### Reference
- **Title**: Plant Disease Detection and Classification Techniques: A Comparative Study of the Performances  
- **Journal**: Journal of Big Data, 2024  
- **Link**: [https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00863-9](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00863-9)

