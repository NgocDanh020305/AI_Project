import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from src.model import CRNN
from src.utils import TextEncoder

# Cấu hình UI
st.set_page_config(page_title="Handwritten Text Recognition", layout="centered")
st.title("✍️ Nhận dạng chữ viết tay (HTR)")
st.write("Upload một ảnh chứa chữ viết tay (Tiếng Anh) để nhận dạng.")

# Cấu hình Model
IMG_HEIGHT = 32
IMG_WIDTH = 128
HIDDEN_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "saved_models/best_model.pth"


@st.cache_resource
def load_model():
    encoder = TextEncoder()
    n_class = len(encoder)
    model = CRNN(IMG_HEIGHT, 1, n_class, HIDDEN_SIZE)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        return model, encoder
    except FileNotFoundError:
        return None, None


def process_image(image_file):
    """Tiền xử lý ảnh giống như khi training"""
    # Convert PIL to Numpy Grayscale
    image = Image.open(image_file).convert('L')
    img = np.array(image)

    # Resize & Pad
    h, w = img.shape
    ratio = w / h
    new_w = int(IMG_HEIGHT * ratio)

    if new_w > IMG_WIDTH:
        new_w = IMG_WIDTH
        img = cv2.resize(img, (new_w, IMG_HEIGHT))
    else:
        img = cv2.resize(img, (new_w, IMG_HEIGHT))

    padded_img = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8) * 255
    padded_img[:, :new_w] = img

    # Normalize
    padded_img = padded_img.astype(np.float32) / 255.0
    padded_img = np.expand_dims(padded_img, axis=0)  # (1, H, W)
    padded_img = np.expand_dims(padded_img, axis=0)  # (1, 1, H, W) -> Batch dim

    return torch.from_numpy(padded_img)


# --- Main App ---
model, encoder = load_model()

if model is None:
    st.error(f"Không tìm thấy model tại `{MODEL_PATH}`. Vui lòng chạy `python src/train.py` trước.")
else:
    uploaded_file = st.file_uploader("Chọn ảnh...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Hiển thị ảnh gốc
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh đã upload', use_column_width=True)

        if st.button('Predict'):
            with st.spinner('Đang nhận dạng...'):
                # Preprocess
                img_tensor = process_image(uploaded_file).to(DEVICE)

                # Inference
                with torch.no_grad():
                    preds = model(img_tensor)  # (T, B, C)
                    # (T, B, C) -> (B, T, C) để decode dễ hơn (batch=1)
                    preds = preds.permute(1, 0, 2)

                # Decode
                decoded_text = encoder.decode(preds)[0]

            st.success("Kết quả:")
            st.markdown(f"## `{decoded_text}`")