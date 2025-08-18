import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify
import os
from flask import send_from_directory

# ---------------- Model Definition ----------------
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = nn.functional.cross_entropy(out, labels)
        return loss

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) 
        self.conv4 = ConvBlock(256, 512, pool=True) 
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# ---------------- Load Model ----------------
num_classes = 38
disease_labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Raspberry___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

model = ResNet9(3, num_classes)
model.load_state_dict(torch.load("plant-disease-model.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)

def predict_image(img_path):
    image = Image.open(img_path).convert('RGB')
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        preds = model(img_t)
        predicted_class = torch.argmax(preds, dim=1).item()
    return predicted_class

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    uploaded_image = None

    if request.method == "POST":
        if "file" not in request.files:
            prediction = "No file uploaded"
        else:
            file = request.files["file"]
            if file.filename == "":
                prediction = "No file selected"
            else:
                file_path = os.path.join("uploads", file.filename)
                file.save(file_path)
                class_id = predict_image(file_path)
                prediction = disease_labels[class_id]
                uploaded_image = file.filename

    return render_template("index.html", prediction=prediction, uploaded_image=uploaded_image)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    class_id = predict_image(file_path)
    prediction = disease_labels[class_id]

    return jsonify({
        "prediction": prediction,
        "filename": file.filename,
        "image_url": f"/uploads/{file.filename}"
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
