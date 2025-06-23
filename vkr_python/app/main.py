from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from io import BytesIO
import torchvision.transforms as transforms
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from app.model import CustomCNN
from facenet_pytorch import MTCNN

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://62.109.19.68:4200"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_names = ('anger','contempt','disgust','fear','happy','neutral','sad','surprise')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("models/best_custom_cnn.pth", map_location=device))
model.eval()

mtcnn = MTCNN(select_largest=False)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert('RGB')

        img_array = np.array(image)
        boxes, _ = mtcnn.detect(img_array)

        if boxes is None:
            return {"error": "На изображении не найдено лицо"}

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = class_names[predicted.item()]
        
        print(f"Распознанная эмоция: {emotion}")
        return {"emotion": emotion}

    except Exception as e:
        return {"error": str(e)}