import torch
from src.model import build_model
from torchvision import transforms
from PIL import Image
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path):
    model = build_model(backbone="resnet18", num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.to(device)
    return model

def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]
        pred = torch.argmax(probs).item()

    return pred, probs.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--checkpoint", default="outputs/best.ckpt", help="Model checkpoint")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    pred, probs = predict_image(model, args.image)
    label = "FAKE" if pred == 1 else "REAL"
    print(f"\n🧠 Prediction: {label} (probabilities: {probs})")
