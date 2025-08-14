# Plant Disease Detection

### Overview
A web-based application for detecting plant leaf diseases using a Flask (Python) backend and a deep learning model (PyTorch).  
This project is designed for early detection to assist farmers, researchers, and agricultural experts in improving crop health.

---

### Tech Stack
- **Frontend:** HTML5, CSS3  
- **Backend:** Python 3, Flask  
- **Deep Learning Framework:** PyTorch, Torchvision  
- **Image Processing:** Pillow (PIL)  
- **Model Architecture:** Custom ResNet9 CNN  
- **Dataset:** PlantVillage (38 classes)

---

### Project Setup
```bash

## 1. Clone repository
git clone https://github.com/ManjuBashini21/Plant-Disease-Detection
cd plant-disease-detection

## 2. Install Dependencies

```bash
pip install flask
pip install torch
pip install torchvision
pip install pillow

## 4. Place trained .pth model in project directory

## 5. Run the Flask app
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

### Implementation Screen Shots


<img width="1919" height="968" alt="Screenshot 2025-08-14 233107" src="https://github.com/user-attachments/assets/bcfcf0ef-bb66-4337-ad27-31fc4bdbd14f" />


<img width="1890" height="983" alt="Screenshot 2025-08-14 233125" src="https://github.com/user-attachments/assets/abddd6f2-ab75-417e-afa2-1857665a5563" />


<img width="1906" height="959" alt="Screenshot 2025-08-14 233140" src="https://github.com/user-attachments/assets/0c8bc1a7-6000-4f60-8ff4-c7241fc68a21" />


