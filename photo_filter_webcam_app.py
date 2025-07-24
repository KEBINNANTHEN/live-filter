import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Streamlit Page Setup ---
st.set_page_config(page_title="📸 Live Photo Filter App", layout="centered")

st.markdown("""
    <style>
        .stButton>button {
            transition: all 0.3s ease;
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        .uploaded-img {
            border-radius: 10px;
            box-shadow: 0px 0px 20px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📸 Live Photo Filter App")
st.write("Upload or capture a photo and apply stylish filters with a smooth interface.")

# --- Image Filtering Functions ---
def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(image, kernel)
    sepia = np.clip(sepia, 0, 255)
    return sepia.astype(np.uint8)

def apply_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

def apply_cartoon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def convert_to_pil(image):
    if len(image.shape) == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# --- Webcam Transformer ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.filter = "None"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.filter == "Grayscale":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.filter == "Sepia":
            return apply_sepia(img)
        elif self.filter == "Blur":
            return apply_blur(img)
        elif self.filter == "Cartoon":
            return apply_cartoon(img)
        return img

# --- Sidebar Filter Selector ---
st.sidebar.header("🎨 Choose a Filter")
filter_option = st.sidebar.radio(
    "Apply a filter to your webcam or uploaded photo:",
    ["None", "Grayscale", "Sepia", "Blur", "Cartoon"]
)

# --- WebCam Section ---
st.subheader("🎥 Live Webcam (Beta)")
ctx = webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    async_transform=True
)

if ctx.video_transformer:
    ctx.video_transformer.filter = filter_option

st.markdown("---")
st.subheader("🖼️ Upload Image Instead")

# --- Image Upload Section ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.image(img, caption="Original Image", use_column_width=True)

    if st.button("✨ Apply Filter"):
        if filter_option == "Grayscale":
            filtered = apply_grayscale(img_cv)
            st.image(filtered, caption="🖤 Grayscale", use_column_width=True, channels="GRAY")
        elif filter_option == "Sepia":
            filtered = apply_sepia(img_cv)
            st.image(convert_to_pil(filtered), caption="🤎 Sepia", use_column_width=True)
        elif filter_option == "Blur":
            filtered = apply_blur(img_cv)
            st.image(convert_to_pil(filtered), caption="💧 Blur", use_column_width=True)
        elif filter_option == "Cartoon":
            filtered = apply_cartoon(img_cv)
            st.image(convert_to_pil(filtered), caption="🎨 Cartoon", use_column_width=True)
        else:
            st.image(img, caption="Original", use_column_width=True)

        # --- Download Link ---
        if 'filtered' in locals():
            pil_image = convert_to_pil(filtered)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            b64 = base64.b64encode(buffer.getvalue()).decode()
            href = f'<a href="data:file/jpg;base64,{b64}" download="filtered.jpg">📥 Download Filtered Image</a>'
            st.markdown(href, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("📸 Built with Streamlit | Made by Kebinn Anthen ❤️")
