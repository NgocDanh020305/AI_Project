import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.utils import TextEncoder


class IAMDataset(Dataset):
    def __init__(self, root_dir, list_file, img_height=32, img_width=128, transform=None):
        """
        root_dir: Thư mục chứa ảnh (vd: data/iam_words)
        list_file: Đường dẫn file words_new.txt
        """
        self.root_dir = root_dir
        self.img_height = img_height
        self.img_width = img_width
        self.encoder = TextEncoder()
        self.transform = transform
        self.samples = self._load_samples(list_file)

    def _load_samples(self, list_file):
        samples = []
        if not os.path.exists(list_file):
            print(f"ERROR: Không tìm thấy file nhãn tại {list_file}")
            return []

        with open(list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                # 1. Bỏ qua dòng trống và comment
                if not line or line.startswith('#'):
                    continue

                # 2. Split thông minh (tự xử lý nhiều khoảng trắng)
                parts = line.split()

                # 3. Kiểm tra độ dài dòng (IAM format có ít nhất 9 cột)
                if len(parts) < 9:
                    continue

                # 4. Kiểm tra trạng thái ảnh (chỉ lấy 'ok')
                if parts[1] != 'ok':
                    continue

                # 5. Xử lý đường dẫn ảnh
                # File name gốc: a01-000u-00-00
                file_name = parts[0] + '.png'

                # Kiểm tra xem ảnh nằm ở đâu (Hỗ trợ cả cấu trúc gốc và cấu trúc Kaggle)
                # Trường hợp 1: Cấu trúc Kaggle (tất cả ảnh trong 1 folder)
                img_path = os.path.join(self.root_dir, file_name)

                # Trường hợp 2: Cấu trúc gốc (chia theo folder a01/a01-000u/...)
                # Nếu không tìm thấy ở root, thử tìm theo cấu trúc gốc
                if not os.path.exists(img_path):
                    parts_name = parts[0].split('-')
                    folder1 = parts_name[0]
                    folder2 = f"{parts_name[0]}-{parts_name[1]}"
                    img_path_nested = os.path.join(self.root_dir, folder1, folder2, file_name)
                    if os.path.exists(img_path_nested):
                        img_path = img_path_nested

                # 6. Lấy nhãn text (từ cột thứ 9 trở đi)
                text = " ".join(parts[8:]).replace("|", " ")

                # Chỉ thêm nếu file ảnh thực sự tồn tại
                if os.path.exists(img_path):
                    samples.append((img_path, text))

        print(f"Đã tải {len(samples)} mẫu dữ liệu hợp lệ.")
        return samples

    def _preprocess_image(self, img):
        # 1. Chuyển Grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Resize vẫn giữ tỉ lệ
        h, w = img.shape
        ratio = w / h
        new_w = int(self.img_height * ratio)

        if new_w > self.img_width:
            new_w = self.img_width
            img = cv2.resize(img, (new_w, self.img_height))
        else:
            img = cv2.resize(img, (new_w, self.img_height))

        # 3. Padding (nền trắng 255)
        padded_img = np.ones((self.img_height, self.img_width), dtype=np.uint8) * 255
        padded_img[:, :new_w] = img

        # 4. Normalize (0-1) và thêm channel dimension
        padded_img = padded_img.astype(np.float32) / 255.0
        padded_img = np.expand_dims(padded_img, axis=0)  # (1, H, W)

        return padded_img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]

        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            # Fallback nếu ảnh lỗi, tạo ảnh trắng
            img = np.ones((self.img_height, self.img_width), dtype=np.uint8) * 255

        img = self._preprocess_image(img)

        # Encode text
        encoded_text = self.encoder.encode(text)

        return {
            "image": torch.from_numpy(img),
            "target": torch.tensor(encoded_text, dtype=torch.long),
            "target_length": torch.tensor(len(encoded_text), dtype=torch.long),
            "text": text
        }


def collate_fn(batch):
    images = []
    targets = []
    target_lengths = []
    texts = []

    for item in batch:
        images.append(item['image'])
        targets.extend(item['target'])
        target_lengths.append(item['target_length'])
        texts.append(item['text'])

    images = torch.stack(images)
    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, targets, target_lengths, texts