from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from io import BytesIO
import torchvision.transforms as transforms
from app.model import ResNet50Emotion, EfficientNetEmotion

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ('anger','contempt','disgust','fear','happy','neutral','sad','surprise')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_resnet_model(path, device):
    state_dict = torch.load(path, map_location=device)

    new_state_dict = {f"model.{k}": v for k, v in state_dict.items()}

    model = ResNet50Emotion(num_classes=len(class_names)).to(device)
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def load_efficientnet_model(path, device):
    state_dict = torch.load(path, map_location=device)

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") 
        new_state_dict[name] = v

    model = EfficientNetEmotion(num_classes=len(class_names)).to(device)
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

resnet_model = load_resnet_model("models/resnet50.pth", device)
efficientnet_model = load_efficientnet_model("models/efficientnet-b0.pth", device)

@app.post("/predict/")
async def predict(file: UploadFile = File(...), model_name: str = "resnet"):
    try:
        image = Image.open(BytesIO(await file.read())).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            if model_name.lower() == "resnet":
                outputs = resnet_model(image)
            elif model_name.lower() == "efficientnet":
                outputs = efficientnet_model(image)
            else:
                return {"error": "Модель не найдена. Выберите 'resnet' или 'efficientnet'."}

            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            emotion = class_names[predicted_idx]

        return {
            "emotion": emotion,
        }

    except Exception as e:
        return {"error": str(e)}