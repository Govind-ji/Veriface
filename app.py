import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.model import build_model

# ----------------------------
# CONFIGURATION
# ----------------------------
CHECKPOINT_PATH = "outputs/best.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["fake", "real"]

# ----------------------------
# MODEL LOADING
# ----------------------------
@st.cache_resource
def load_model():
    model = build_model(backbone="resnet18", num_classes=len(CLASSES))
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # ✅ Handle checkpoints with nested keys
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# ----------------------------
# IMAGE PREPROCESSING
# ----------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_image(model, image):
    image_tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return CLASSES[pred.item()], conf.item()

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="DeepFake Detector", layout="centered")

st.title("🕵️‍♂️ DeepFake Detection App")
st.markdown(
    "Upload an image and this app will predict whether it's **Real** or **Fake** using a fine-tuned ResNet18 model."
)

uploaded_file = st.file_uploader("📤 Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        model = load_model()
        pred, conf = predict_image(model, image)

    st.subheader("✅ Prediction Result")
    if pred == "real":
        st.success(f"**Prediction:** REAL FACE 🧍‍♂️\n\n**Confidence:** {conf*100:.2f}%")
    else:
        st.error(f"**Prediction:** FAKE FACE 🤖\n\n**Confidence:** {conf*100:.2f}%")

else:
    st.info("Please upload an image to get a prediction.")

st.caption("Model: ResNet18 | Framework: PyTorch | Interface: Streamlit")
    