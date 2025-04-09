import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import os

# Define class labels (match this with your training labels)
class_names = ['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Predict class for a single image
def predict(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alzheimer Stage Prediction")
    parser.add_argument('--img_path', type=str, required=True, help='Path to image')
    parser.add_argument('--model_path', type=str, default='resnet18_alzheimer_model.pth', help='Path to model (.pth) file')
    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        print(f"❌ Image path does not exist: {args.img_path}")
    elif not os.path.exists(args.model_path):
        print(f"❌ Model path does not exist: {args.model_path}")
    else:
        model = load_model(args.model_path)
        prediction = predict(model, args.img_path)
        print(f"🧠 Predicted Alzheimer Stage: {prediction}")
