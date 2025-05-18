import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# -------------------------------
# IVF Healthcare UI Styling
# -------------------------------
st.set_page_config(page_title="IVF Embryo Grading", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #f5f9ff;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        h1, h4 {
            color: #145DA0;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            margin-top: 0.5rem;
        }
        .stButton>button:hover {
            background-color: #125582;
        }
        .stImage>img {
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            max-width: 150px;
            max-height: 150px;
            margin-bottom: 8px;
        }
        .uploadedFileName, .uploadedFileSize, .stFileUploader label + div {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Constants
# -------------------------------
class_names = ['Grade A', 'Grade B', 'Grade C']
MODEL_DIR = r'"C:\Users\91739\Downloads\project\project\d\models"'

# -------------------------------
# Load Grading Model (for Days)
# -------------------------------
def load_model_for_day(day):
    model_path = os.path.join(MODEL_DIR, f'Day_{day}_resnet_model.pth')  # Changed model path to match saved model names
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# -------------------------------
# Load Binary Classification Model
# -------------------------------
def load_binary_model():
    binary_model_path = os.path.join(MODEL_DIR, 'error_resnet_model.pth')  # Binary model path remains unchanged
    model = models.resnet18(pretrained=False)  # Assuming resnet18 for error detection as well
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: Embryo (0), Error_image (1)
    model.load_state_dict(torch.load(binary_model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# -------------------------------
# Binary Prediction - Check if Embryo
# -------------------------------
def is_embryo_image(image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = binary_model(input_tensor)
        pred = torch.argmax(outputs, 1).item()
    return pred == 0  # 0 = Embryo, 1 = Error

# -------------------------------
# Predict Grade
# -------------------------------
def predict_grade(image, model):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
    return class_names[pred_idx], probs[pred_idx].item()

# -------------------------------
# Implantation Logic
# -------------------------------
def make_implant_decision(grades):
    count_A = grades.count('Grade A')
    count_B = grades.count('Grade B')
    count_C = grades.count('Grade C')

    if count_C >= 2:
        return "❌ Not suitable for implantation. Two or more embryos are Grade C. Please consult your doctor."
    elif count_C == 1 and count_A == 0:
        return "⚠️ One embryo is Grade C and no Grade A. Medical consultation recommended."
    elif count_A >= 2 and count_C == 0:
        return "✅ Excellent quality! Two or more embryos are Grade A. Suitable for implantation."
    elif all(g in ['Grade A', 'Grade B'] for g in grades):
        return "🟡 Fair. No Grade C embryos. Implantation may be considered with medical advice."
    else:
        return "⚠️ Mixed grades. Consult your embryologist before proceeding."

# -------------------------------
# Load Models
# -------------------------------
model_day3 = load_model_for_day(3)
model_day4 = load_model_for_day(4)
model_day5 = load_model_for_day(5)
binary_model = load_binary_model()

# -------------------------------
# UI Header
# -------------------------------
st.markdown("<h1 style='text-align: center;'>🧬 IVF Embryo Grading & Implantation Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload embryo images for Day 3 (8-cell), Day 4 (Morula), and Day 5 (Blastocyst)</p>", unsafe_allow_html=True)

# -------------------------------
# Upload UI
# -------------------------------
col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 0.5])
images = {}

with col1:
    st.markdown("**Day 3 - 8-cell**")
    d3_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key="day3")
    if d3_file:
        images["Day 3"] = Image.open(d3_file).convert('RGB')

with col2:
    st.markdown("**Day 4 - Morula**")
    d4_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key="day4")
    if d4_file:
        images["Day 4"] = Image.open(d4_file).convert('RGB')

with col3:
    st.markdown("**Day 5 - Blastocyst**")
    d5_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key="day5")
    if d5_file:
        images["Day 5"] = Image.open(d5_file).convert('RGB')

# -------------------------------
# Predict Button
# -------------------------------
predict = False
with col4:
    st.markdown("###")
    if st.button("🔍 Predict", use_container_width=True):
        predict = True

# -------------------------------
# Prediction Logic
# -------------------------------
if len(images) == 3:
    cols = st.columns(3)
    grades = []
    valid_embryos = True  # Flag to track if all images are valid embryos

    for i, (day, img) in enumerate(images.items()):
        resized_img = img.resize((300, 300))
        with cols[i]:
            st.image(resized_img, caption=day)

            if predict:
                # Check if valid embryo
                if not is_embryo_image(img):
                    st.error(f"🚫 Not an embryo image for {day}. Please upload a valid embryo image.")
                    valid_embryos = False  # Mark as invalid embryo
                    grades.append("Grade C")  # Assume lowest grade for this day
                    continue

                # Predict Grade
                grade, conf = predict_grade(img, {
                    "Day 3": model_day3,
                    "Day 4": model_day4,
                    "Day 5": model_day5
                }[day])
                grades.append(grade)
                st.markdown(f"**{grade}**<br><sub>{conf:.2%} confidence</sub>", unsafe_allow_html=True)

    # Show implantation recommendation only if all images are valid embryos
    if predict and valid_embryos:
        st.divider()
        st.markdown("<h4 style='text-align: center;'>Implantation Recommendation</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{make_implant_decision(grades)}</p>", unsafe_allow_html=True)

        with st.expander("👩‍⚕️ Doctor's Advice"):
            st.markdown("""
            - **Grade A:** Embryos with the best morphology, high implantation potential.
            - **Grade B:** Average embryos, often still viable.
            - **Grade C:** Poor quality, typically low implantation chance. Further medical consultation recommended.
            - **Tip:** Maintain a healthy lifestyle, and follow embryologist or fertility specialist guidance closely.
            """)

else:
    st.info("Upload all 3 embryo images to proceed with prediction.")
