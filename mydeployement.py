# -*- coding: utf-8 -*-

import cv2
print(cv2.__version__)

import streamlit as st
st.set_page_config(page_title="Embryo Grading + Grad-CAM", layout="wide")  # ✅ FIRST Streamlit command
#pip install timm
import torch
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = r'C:\Users\91739\OneDrive\Documents\PROJECT\convnext_model.pth'
CLASS_NAMES = ['Grade A', 'Grade B', 'Grade C']
IMAGE_SIZE = (224, 224)

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    model = timm.create_model('convnext_base', pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ------------------------------
# Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# Grad-CAM Function
# ------------------------------
def generate_gradcam(model, img_tensor, class_idx):
    img_tensor = img_tensor.unsqueeze(0)

    # Hook the gradients and feature maps
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Use final conv layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    last_conv.register_forward_hook(forward_hook)
    last_conv.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    grad = gradients[0].detach()
    activation = activations[0].detach()

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1).squeeze()

    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, IMAGE_SIZE)
    return cam

# ------------------------------
# Prediction
# ------------------------------
def predict_and_visualize(image):
    img_tensor = transform(image)
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        prob = F.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(prob).item()
        confidence = prob[pred_idx].item()

    cam = generate_gradcam(model, img_tensor, pred_idx)
    return CLASS_NAMES[pred_idx], confidence, cam

# ------------------------------
# Implantation Logic
# ------------------------------
def make_implant_decision(grades):
    count_A = grades.count('Grade A')
    count_B = grades.count('Grade B')
    count_C = grades.count('Grade C')

    if count_C >= 2:
        return "❌ Not suitable for implantation. Two or more embryos are Grade C."
    elif count_C == 1 and count_A == 0:
        return "⚠️ One embryo is Grade C and no Grade A. Medical consultation recommended."
    elif count_A >= 2 and count_C == 0:
        return "✅ Excellent! Two or more embryos are Grade A. Suitable for implantation."
    elif all(g in ['Grade A', 'Grade B'] for g in grades):
        return "🟡 Fair. No Grade C embryos. Implantation may be considered with advice."
    else:
        return "⚠️ Mixed grades. Please consult your embryologist."

# ------------------------------
# Streamlit UI
# ------------------------------

st.title("🧬 Embryo Quality Grading with Grad-CAM")
st.write("Upload embryo images for Day 3, Day 4, and Day 5. The model will predict the grade and show activation heatmaps (Grad-CAM).")

uploaded = {}
cols = st.columns(3)
day_labels = ["Day 3 - 8-cell", "Day 4 - Morula", "Day 5 - Blastocyst"]

for i, day in enumerate(day_labels):
    with cols[i]:
        st.markdown(f"**{day}**")
        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key=f"upload_{i}")
        if file:
            uploaded[day] = Image.open(file).convert('RGB')


#-----------------------


#---------------------
def validate_image_class(image):
    # Implement your image classification logic here
    predicted_class = classify_image(image)
    if predicted_class not in ['8-cell', 'Morula', 'Blastocyst']:
        st.error("Error: Uploaded image does not belong to the expected classes.")
        return False
    return True

#---------------------
# integration
#---------------------
#!where python
#pip install streamlit_authenticator
import streamlit_authenticator as stauth
print("streamlit_authenticator imported successfully!")

#pip install streamlit-authenticator
#import streamlit_authenticator
print("Installed correctly!")

# First uninstall completely
#pip uninstall streamlit-authenticator -y

# Then install specific version
#pip install streamlit-authenticator==0.2.3
import streamlit as st
import streamlit_authenticator as stauth

passwords = ['12345']
hashed_passwords = stauth.Hasher(passwords).generate()

config = {
    'credentials': {
        'usernames': {
            'harika': {
                'name': 'Vyshnavi',
                'password': hashed_passwords[0]  # Use the hashed password
            }
        }
    },
    'cookie': {
        'name': 'embryo_app_cookie',
        'key': 'abcdef123456',
        'expiry_days': 1
    },
    'preauthorized': {
        'emails': []
    }
}
# pip install authenticator
#pip install --upgrade streamlit
st.runtime.caching.maybe_show_cached_st_function_warning

# ---- Initialize authenticator ----
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ---- Login Section ----
try:
    name, authentication_status, username = authenticator.login('Login', 'sidebar')
except TypeError:
    # Fallback for different versions
    name, authentication_status, username = authenticator.login('Login')

# Login logic
if authentication_status:
    st.sidebar.success(f"Welcome {name}!")
    st.title("Your App Content")
elif authentication_status is False:
    st.sidebar.error("Username/password incorrect")
elif authentication_status is None:
    st.sidebar.warning("Please enter credentials")




# ------------------------------
# Predict Button
# ------------------------------
if st.button("🔍 Predict Grades"):
    if len(uploaded) != 3:
        st.warning("Please upload images for all 3 days.")
    else:
        st.subheader("📊 Predictions and Grad-CAM")
        grade_results = []
        result_cols = st.columns(3)

        for i, (day, img) in enumerate(uploaded.items()):
            grade, conf, cam = predict_and_visualize(img)
            grade_results.append(grade)

            # Overlay Grad-CAM on image
            img_np = np.array(img.resize(IMAGE_SIZE))
            heatmap = (cam * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

            with result_cols[i]:
                st.image(overlay, caption=f"{day}: {grade} ({conf:.2%})", use_column_width=True)

        st.divider()
        st.subheader("🧬 Implantation Recommendation")
        st.success(make_implant_decision(grade_results))
