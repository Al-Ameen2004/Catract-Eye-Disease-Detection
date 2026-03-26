import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import timm
import os
from google import genai

# -------------------------
# Configure Gemini API (New SDK)
# -------------------------
api_key = ("AIzaSyBDTR0Qk9hV7ZiuN7eyWc62rnsu8noi3eI")  # safer way

if api_key:
    client = genai.Client(api_key=api_key)
else:
    st.warning("GEMINI_API_KEY not found. AI Assistant disabled.")
    client = None

st.set_page_config(page_title="AI Cataract Detection", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Hybrid Model Definition
# -------------------------
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()

        # ResNet18
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()

        # ViT Tiny
        self.vit = timm.create_model("vit_tiny_patch16_224", pretrained=False)
        self.vit.head = nn.Identity()

        # Classifier
        self.classifier = nn.Linear(704, 3)

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.vit(x)
        combined = torch.cat((cnn_features, vit_features), dim=1)
        output = self.classifier(combined)
        return output

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    model_path = "hybrid_model.pth"

    if not os.path.exists(model_path):
        return None

    model = HybridModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# -------------------------
# UI
# -------------------------
st.title("Hybrid ResNet18 + ViT Cataract Detection")

if model is None:
    st.error("⚠️ Model file 'hybrid_model.pth' not found.")
    st.stop()

st.write("Upload an eye image to classify.")

uploaded_file = st.file_uploader(
    "Upload Eye Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_names = [
        "Cataract",
        "Normal",
        "Other Disease"
    ]

    prediction = class_names[predicted.item()]
    conf = confidence.item() * 100

    st.subheader("Prediction Result")

    # -------------------------
    # Eye Detection Logic
    # -------------------------
    THRESHOLD = 60

    if conf < THRESHOLD:
        st.error("❌ Invalid Input: Eye not detected properly")

    else:
        st.success(f"**{prediction}**")
        st.write(f"Confidence: {conf:.2f}%")

        # -------------------------
        # AI Diagnosis Summary (AUTO)
        # -------------------------
        if client:
            st.divider()
            st.subheader("🩺 AI Diagnosis Summary")

            summary_prompt = f"""
            The AI model predicted: {prediction} with {conf:.2f}% confidence.

            Explain:
            - What this condition means
            - Possible symptoms
            - When to consult a doctor

            Keep it simple and short.
            """

            try:
                summary = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=summary_prompt
                )
                st.info(summary.text)

            except Exception as e:
                st.warning("Could not generate summary.")

        # -------------------------
        # AI Chatbot
        # -------------------------
        if client:
            st.divider()
            st.subheader("AI Eye Care Assistant")

            user_input = st.text_input("Ask about cataract treatment or eye care")

            if user_input:

                prompt = f"""
                The AI model predicted: {prediction}
                User question: {user_input}

                Provide simple eye care guidance.
                Encourage consulting an ophthalmologist.
                Do not give strict medical prescriptions.
                """

                with st.spinner("Thinking..."):
                    try:
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=prompt
                        )

                        st.write("🤖 AI Assistant:")
                        st.info(response.text)

                    except Exception as e:
                        st.error(f"Chatbot error: {e}")

# -------------------------
# Disclaimer
# -------------------------
st.divider()
st.caption(
    "⚠️ This system provides informational guidance only and does not replace professional medical advice."
)
