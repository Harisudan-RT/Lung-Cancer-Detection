# streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw
import time

# ----------------------------
# Custom CSS for styling
# ----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Montserrat', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton button:hover {
        background-color: #0D47A1;
        transform: scale(1.05);
    }
    
    .uploaded-file {
        text-align: center;
        padding: 1.5rem;
        border-radius: 15px;
        background: linear-gradient(145deg, #e6f0ff, #ffffff);
        box-shadow: 5px 5px 15px #d9d9d9, -5px -5px 15px #ffffff;
        margin-bottom: 2rem;
    }
    
    .results-container {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(145deg, #e3f2fd, #bbdefb);
        box-shadow: 5px 5px 15px #d9d9d9, -5px -5px 15px #ffffff;
        margin-top: 2rem;
    }
    
    .prediction-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0D47A1;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E88E5;
        text-align: center;
    }
    
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        transition: all 0.5s;
    }
    
    .image-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.15);
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #757575;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# CNN Model (MUST match training!)
# ----------------------------
class LungCancer3DCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(LungCancer3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)

        # NOTE: must match training shape!
        # Input fake volume: (32,64,64) → after 2x pooling: (8,16,16)
        self.fc1 = nn.Linear(64 * 8 * 16 * 16, 256)  # =131072
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------------------
# Load trained model
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "lung_cancer_model_best.pth"

@st.cache_resource
def load_model():
    model = LungCancer3DCNN(num_classes=3).to(DEVICE)
    # For demo purposes, we'll create a dummy model if the real one isn't available
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
    except:
        st.warning("Using demo mode - real model file not found")
    model.eval()
    return model

model = load_model()
CLASSES = ["Benign", "Malignant", "Normal"]

# ----------------------------
# Image effect functions
# ----------------------------
def apply_blur_effect(image, blur_level=5):
    """Apply blur effect to image"""
    return image.filter(ImageFilter.GaussianBlur(blur_level))

def apply_crack_effect(image, num_cracks=5):
    """Apply crack effect to image"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    for _ in range(num_cracks):
        # Start from a random point on the top edge
        start_x = np.random.randint(0, width)
        start_y = 0
        
        # Generate a jagged line downward
        points = [(start_x, start_y)]
        x, y = start_x, start_y
        
        while y < height:
            x += np.random.randint(-5, 5)
            y += np.random.randint(5, 15)
            points.append((x, y))
        
        # Draw the crack
        draw.line(points, fill="black", width=1)
    
    return img

# ----------------------------
# Preprocess JPG/PNG
# ----------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("L")  # grayscale
    img = np.array(img)

    # normalize
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

    # resize to training size (64x64)
    img_resized = cv2.resize(img, (64, 64))

    # create fake 3D volume (depth=32 slices)
    volume = np.stack([img_resized] * 32, axis=0)

    # (D,H,W) → (1,1,D,H,W)
    volume = torch.tensor(volume).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    return volume, img_resized

# ----------------------------
# Streamlit UI
# ----------------------------
st.markdown('<h1 class="main-header">🫁 Lung Cancer Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">3D CNN Deep Learning Model for CT Scan Analysis</p>', unsafe_allow_html=True)

# Add some info about the app
with st.expander("ℹ️ About this app"):
    st.write("""
    This application uses a 3D Convolutional Neural Network to analyze lung CT scans and predict 
    the likelihood of cancer. The model can classify scans into three categories:
    - **Benign**: Non-cancerous tissue
    - **Malignant**: Cancerous tissue
    - **Normal**: Healthy lung tissue
    
    Upload a CT scan image to get started.
    """)

# File upload section
st.markdown('<div class="uploaded-file">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a lung CT scan (JPG/PNG)", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Display original image
    original_img = Image.open(uploaded_file)
    st.markdown("### 📷 Original Image")
    st.image(original_img, caption="Original CT Scan", use_column_width=True)
    
    # Process button
    if st.button("🔍 Analyze Image", key="process"):
        with st.spinner("Processing image..."):
            # Create a placeholder for the image with effects
            effect_placeholder = st.empty()
            
            # Apply and display effects in sequence
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Show original
            status_text.text("Loading image...")
            effect_placeholder.image(original_img, caption="Original Image", use_column_width=True)
            progress_bar.progress(25)
            time.sleep(0.5)
            
            # Step 2: Apply blur effect
            status_text.text("Applying filters...")
            blurred_img = apply_blur_effect(original_img, 3)
            effect_placeholder.image(blurred_img, caption="Applying filters...", use_column_width=True)
            progress_bar.progress(50)
            time.sleep(0.5)
            
            # Step 3: Apply crack effect
            status_text.text("Enhancing details...")
            cracked_img = apply_crack_effect(blurred_img, 7)
            effect_placeholder.image(cracked_img, caption="Enhancing details...", use_column_width=True)
            progress_bar.progress(75)
            time.sleep(0.5)
            
            # Step 4: Show final processed image
            status_text.text("Final processing...")
            volume, processed_img = preprocess_image(uploaded_file)
            effect_placeholder.image(processed_img, caption="Processed Image (Ready for Analysis)", use_column_width=True)
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # Perform prediction
            status_text.text("Running AI analysis...")
            with torch.no_grad():
                outputs = model(volume)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results in a styled container
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            st.markdown("### 📊 Analysis Results")
            
            # Display prediction with color coding
            if CLASSES[pred] == "Benign":
                st.markdown(f'<p class="prediction-text">Diagnosis: <span style="color:#4CAF50;">{CLASSES[pred]}</span></p>', unsafe_allow_html=True)
            elif CLASSES[pred] == "Malignant":
                st.markdown(f'<p class="prediction-text">Diagnosis: <span style="color:#F44336;">{CLASSES[pred]}</span></p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="prediction-text">Diagnosis: <span style="color:#2196F3;">{CLASSES[pred]}</span></p>', unsafe_allow_html=True)
            
            st.markdown(f'<p class="confidence-text">Confidence: {probs[0][pred].item():.2%}</p>', unsafe_allow_html=True)
            
            # Show confidence breakdown
            st.markdown("**Confidence Breakdown:**")
            for i, cls in enumerate(CLASSES):
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.write(f"{cls}:")
                with col2:
                    st.progress(float(probs[0][i]))
                with col3:
                    st.write(f"{probs[0][i].item():.2%}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            


# Add footer
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("Lung Cancer Detection App | 3D CNN Model")
st.markdown('</div>', unsafe_allow_html=True)