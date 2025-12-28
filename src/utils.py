import torch
import numpy as np

# Danh sách ký tự được hỗ trợ (IAM dataset + số + ký tự đặc biệt)
VOCAB = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class TextEncoder:
    def __init__(self, vocab=VOCAB):
        self.vocab = vocab
        # Mapping char -> index (bắt đầu từ 1, 0 dành cho CTC blank)
        self.char2idx = {char: idx + 1 for idx, char in enumerate(vocab)}
        self.idx2char = {idx + 1: char for idx, char in enumerate(vocab)}

    def encode(self, text):
        """Chuyển chuỗi text thành danh sách index."""
        return [self.char2idx[char] for char in text if char in self.char2idx]

    def decode(self, preds, raw=False):
        """
        Giải mã output của CTC (Greedy Search).
        preds: Tensor (T, N, C) hoặc (T, C)
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.argmax(2).detach().cpu().numpy()  # Lấy index có xác suất cao nhất

        decoded_batch = []
        for sequence in preds:  # Duyệt qua từng mẫu trong batch
            decoded_text = []
            if raw:
                # Trả về chuỗi raw (bao gồm cả trùng lặp và blank)
                decoded_text = [self.idx2char[idx] for idx in sequence if idx in self.idx2char]
            else:
                # CTC decoding logic: gộp ký tự trùng và bỏ blank (0)
                prev_idx = -1
                for idx in sequence:
                    if idx != prev_idx and idx != 0:
                        if idx in self.idx2char:
                            decoded_text.append(self.idx2char[idx])
                    prev_idx = idx
            decoded_batch.append("".join(decoded_text))

        return decoded_batch

    def __len__(self):
        return len(self.vocab) + 1  # +1 cho blank token