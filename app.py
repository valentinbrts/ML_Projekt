import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os

# Klassen aus deinem prepare_data.py
CLASS_NAMES = [
    "baroque",
    "cubism",
    "pop_art",
    "realism",
    "romanticism"
]

MODEL_PATH = "model/best_model.pth"

st.set_page_config(
    page_title="Epochen Klassifikator",
    page_icon="🎨",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image, model):
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = torch.argmax(probabilities).item()
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[predicted_index].item()

    return predicted_class, confidence, probabilities


st.title("🎨 Epochen Klassifikator")
st.write("Lade ein Kunstbild hoch und das Modell sagt vorher, zu welcher Epoche es am wahrscheinlichsten gehört.")

with st.sidebar:
    st.header("Bild hochladen")
    uploaded_file = st.file_uploader(
        "Wähle ein Bild aus",
        type=["jpg", "jpeg", "png", "webp"]
    )

    st.markdown("---")
    st.subheader("Über das Modell")
    st.write("**Modell:** ResNet18")
    st.write("**Bildgröße:** 224 × 224")
    st.write("**Klassen:** Barock, Kubismus, Pop-Art, Realismus, Romantik")
    st.markdown("---")
    st.subheader("Unser Team")
    st.write("**K-Team:** Anna Julitz, Valentin Bridts, Eric Schmidt")

model = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    predicted_class, confidence, probabilities = predict(image, model)

    with col2:
        st.subheader("📊 Vorhersage Ergebnis")
        st.metric(
            label="Wahrscheinlichste Epoche",
            value=predicted_class.replace("_", " ").title(),
            delta=f"{confidence * 100:.2f}% Sicherheit"
        )

        prob_df = pd.DataFrame({
            "Epoche": [c.replace("_", " ").title() for c in CLASS_NAMES],
            "Wahrscheinlichkeit": [p.item() * 100 for p in probabilities]
        }).sort_values("Wahrscheinlichkeit", ascending=False)

        st.bar_chart(
            prob_df.set_index("Epoche")
        )

    st.markdown("---")
    st.subheader("📈 Modell Performance")

    col3, col4 = st.columns(2)

    with col3:
        st.write("Confusion Matrix")
        if os.path.exists("outputs/confusion_matrix.png"):
            st.image("outputs/confusion_matrix.png", use_container_width=True)
        elif os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.warning("Confusion Matrix Bild wurde nicht gefunden.")

    with col4:
        st.write("Trainingsgraph")
        if os.path.exists("outputs/training_curves.png"):
            st.image("outputs/training_curves.png", use_container_width=True)
        elif os.path.exists("training_curves.png"):
            st.image("training_curves.png", use_container_width=True)
        else:
            st.warning("Trainingsgraph wurde nicht gefunden.")

else:
    st.info("Bitte lade links ein Bild hoch, um eine Vorhersage zu starten.")
