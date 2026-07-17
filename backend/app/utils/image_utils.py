import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_black_border(img: np.ndarray, tol: int = 7) -> np.ndarray:
    mask = img.mean(axis=2) > tol
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def ben_graham(img: np.ndarray, sigma: int = 10) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    result  = cv2.addWeighted(img, 4, blurred, -4, 128)
    mask    = np.zeros(img.shape, dtype=np.float32)
    cv2.circle(
        mask,
        (img.shape[1] // 2, img.shape[0] // 2),
        int(img.shape[0] * 0.9 // 2),
        (1, 1, 1), -1,
    )
    return (result * mask + 128 * (1 - mask)).astype(np.uint8)


def preprocess(image_bytes: bytes, img_size: int = 224) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_black_border(img)
    img = cv2.resize(img, (img_size, img_size))
    img = ben_graham(img)
    return img


def to_tensor(img_np: np.ndarray, device):
    import torch
    x = img_np.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)
