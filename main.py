import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI(title="Handwritten Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CẤU HÌNH ĐƯỜNG DẪN (Đã chuẩn theo máy bạn) ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(image_bytes):
    # 1. Chuyển file tải lên thành mảng số (numpy array)
    nparr = np.frombuffer(image_bytes, np.uint8)

    # 2. Đọc ảnh chế độ UNCHANGED để lấy cả lớp trong suốt (Alpha)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # 3. Xử lý nền trong suốt (Biến nền trong suốt thành màu trắng)
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha_channel = img[:, :, 3]
        rgb_channels = img[:, :, :3]
        white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
        img = (rgb_channels * alpha_factor + white_background * (1 - alpha_factor)).astype(np.uint8)

    # 4. Chuyển sang ảnh xám
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 5. Tăng tương phản (Threshold): Chữ đen, nền trắng
    _, processed_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    return processed_img


@app.post("/recognize")
async def recognize_text(file: UploadFile = File(...)):
    try:
        content = await file.read()
        processed_img = preprocess_image(content)

        # Cấu hình đọc: --psm 6 (coi như một khối văn bản ngang)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_img, config=custom_config, lang='eng')

        return {"status": "success", "text": text.strip()}

    except Exception as e:
        print(f"Lỗi: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)