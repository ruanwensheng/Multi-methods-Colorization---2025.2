import numpy as np
import cv2
from pathlib import Path


def create_scribble_from_color_image(
    color_img: np.ndarray,
    sample_ratio: float = 0.02,
    seed: int = 42
) -> np.ndarray:
    """
    Tự động tạo scribble từ ảnh màu gốc (dùng để test & đánh giá).
    Lấy ngẫu nhiên sample_ratio% số pixel làm "hint màu".
    
    Args:
        color_img  : ảnh màu BGR
        sample_ratio: tỉ lệ pixel được chọn làm scribble (mặc định 2%)
        seed       : random seed để reproducible
    Returns:
        scribble_img: ảnh BGR — pixel scribble giữ màu gốc, còn lại = đen
    """
    np.random.seed(seed)
    H, W = color_img.shape[:2]
    scribble = np.zeros_like(color_img)

    n_pixels = int(H * W * sample_ratio)
    rows = np.random.randint(0, H, n_pixels)
    cols = np.random.randint(0, W, n_pixels)

    scribble[rows, cols] = color_img[rows, cols]
    return scribble


def draw_scribble_strokes(
    canvas: np.ndarray,
    strokes: list[dict]
) -> np.ndarray:
    """
    Vẽ nhiều stroke lên canvas scribble.
    
    Args:
        canvas : ảnh nền đen (H, W, 3), dtype uint8
        strokes: list các stroke, mỗi stroke là dict:
                 {"points": [(x1,y1),(x2,y2),...], "color": (B,G,R), "thickness": 3}
    Returns:
        canvas: ảnh với các stroke đã vẽ
    """
    result = canvas.copy()
    for stroke in strokes:
        pts = stroke["points"]
        color = stroke["color"]
        thickness = stroke.get("thickness", 3)
        for i in range(len(pts) - 1):
            cv2.line(result, pts[i], pts[i+1], color, thickness)
    return result


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray
) -> dict:
    """
    Tính PSNR và SSIM giữa ảnh dự đoán và ground truth.
    
    Args:
        pred: ảnh colorized (H, W, 3) BGR uint8
        gt  : ảnh màu gốc  (H, W, 3) BGR uint8
    Returns:
        dict với "psnr" và "ssim"
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    pred_f = pred.astype(np.float64)
    gt_f   = gt.astype(np.float64)

    psnr = peak_signal_noise_ratio(gt_f, pred_f, data_range=255.0)
    ssim = structural_similarity(gt_f, pred_f, channel_axis=2, data_range=255.0)

    return {"psnr": round(psnr, 4), "ssim": round(ssim, 4)}


def visualize_result(
    gray_img  : np.ndarray,
    scribble_img: np.ndarray,
    result_img: np.ndarray,
    gt_img    : np.ndarray | None = None,
    save_path : str | None = None
):
    """
    Hiển thị side-by-side: grayscale | scribble | result | ground truth.
    """
    import matplotlib.pyplot as plt

    imgs   = [gray_img, scribble_img, result_img]
    titles = ["Grayscale (input)", "Scribble (hint)", "Colorized (output)"]

    if gt_img is not None:
        imgs.append(gt_img)
        titles.append("Ground truth")

    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    for ax, img, title in zip(axes, imgs, titles):
        # OpenCV dùng BGR, matplotlib dùng RGB
        if img.ndim == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()