"""
Gradio demo for Multi-methods Image Colorization.

Tabs:
  1. Deep Learning  - automatic colorization (Zhang16, Zhang17, DeOldify)
  2. Scribble-based - user draws color hints with a brush
  3. Example-based  - transfer color from a reference image

Run:
  python app/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gradio as gr


def _load_deep_colorizer(model_name: str):
    from src.deep_learning.colorizer import DeepColorizer
    from src.deep_learning.utils import load_config
    cfg = load_config("configs/config.yaml")
    model_paths = {
        "Zhang16 Pretrained": "models/pretrained/zhang16_eccv.pth",
        "Zhang16 Fine-tuned": "models/deep_learning/best_model.pth",
    }
    path = model_paths.get(model_name)
    return DeepColorizer(model_path=path, cfg=cfg)


def _load_scribble_colorizer():
    from src.scribble.colorizer import ScribbleColorizer
    return ScribbleColorizer()


def _load_example_colorizer():
    from src.example_based.colorizer import ExampleColorizer
    return ExampleColorizer()


def colorize_deep(image: np.ndarray, model_name: str):
    if image is None:
        return None, "Please upload an image."
    try:
        colorizer = _load_deep_colorizer(model_name)
        result, info = colorizer.colorize(image)
        return result, f"Model: {model_name} | Time: {info['time_seconds']:.2f}s"
    except Exception as e:
        return None, f"Error: {e}"


def colorize_scribble(editor_data):
    """
    editor_data: dict from gr.ImageEditor.
    Keys: 'background' (grayscale base), 'layers' (drawn color strokes).
    """
    if editor_data is None:
        return None, "Please upload an image and draw color hints."
    try:
        background = editor_data.get("background")
        layers = editor_data.get("layers", [])
        if background is None:
            return None, "No background image found."

        bg = np.array(background)
        scribble_overlay = np.zeros((*bg.shape[:2], 4), dtype=np.uint8)
        for layer in layers:
            if layer is not None:
                layer_arr = np.array(layer)
                mask = layer_arr[..., 3] > 0
                scribble_overlay[mask] = layer_arr[mask]

        colorizer = _load_scribble_colorizer()
        result, info = colorizer.colorize(bg, scribble_overlay)
        return result, f"Time: {info['time_seconds']:.2f}s"
    except Exception as e:
        return None, f"Error: {e}"


# ── Example-based: swatch color definitions ──────────────────────────────────
# Each tuple: (name, hex_for_brush, rgb_for_detection) — must stay in sync.
_SWATCH_COLORS = [
    ("red",    "#DC3232", (220,  50,  50)),
    ("green",  "#32B432", ( 50, 180,  50)),
    ("blue",   "#3232DC", ( 50,  50, 220)),
    ("yellow", "#DCC832", (220, 200,  50)),
]
_SWATCH_TOLERANCE = 60  # max Euclidean RGB distance to match a painted pixel


def _extract_color_mask(layers, target_rgb):
    """Return (H, W) bool mask of pixels painted with target_rgb (within tolerance)."""
    mask = None
    for layer in (layers or []):
        if layer is None:
            continue
        arr = np.array(layer)
        if arr.ndim < 3 or arr.shape[2] < 4:
            continue
        alpha = arr[:, :, 3] > 0
        dist = np.sqrt(np.sum(
            (arr[:, :, :3].astype(np.int32) - np.array(target_rgb)) ** 2, axis=2
        ))
        hit = alpha & (dist < _SWATCH_TOLERANCE)
        mask = hit if mask is None else (mask | hit)
    return mask


def _parse_swatch_pairs(editor_t, editor_r):
    """Extract (target_mask, ref_mask) pairs from two ImageEditor outputs."""
    tgt_layers = (editor_t or {}).get("layers", [])
    ref_layers = (editor_r or {}).get("layers", [])
    swatches = []
    for _, _, rgb in _SWATCH_COLORS:
        tm = _extract_color_mask(tgt_layers, rgb)
        rm = _extract_color_mask(ref_layers, rgb)
        if tm is not None and rm is not None and tm.any() and rm.any():
            swatches.append((tm, rm))
    return swatches


def colorize_example(gray_image: np.ndarray, reference_image: np.ndarray):
    """Mode 1 — automatic global matching."""
    if gray_image is None or reference_image is None:
        return None, "Please upload both a grayscale image and a color reference image."
    try:
        colorizer = _load_example_colorizer()
        result, info = colorizer.colorize(gray_image, reference_image)
        return result, f"Time: {info['time_seconds']:.2f}s"
    except Exception as e:
        return None, f"Error: {e}"


def colorize_example_swatch(editor_t, editor_r):
    """Mode 2 — user-guided swatch matching."""
    if editor_t is None or editor_r is None:
        return None, "Please upload both images and draw swatch regions."
    try:
        bg_t = editor_t.get("background")
        bg_r = editor_r.get("background")
        if bg_t is None or bg_r is None:
            return None, "No background image found. Upload images first."
        gray_image = np.array(bg_t)
        reference_image = np.array(bg_r)
        swatches = _parse_swatch_pairs(editor_t, editor_r)
        colorizer = _load_example_colorizer()
        result, info = colorizer.colorize(gray_image, reference_image,
                                          swatches=swatches or None)
        note = f"{len(swatches)} swatch pair(s)" if swatches else "no swatches — fell back to auto"
        return result, f"Time: {info['time_seconds']:.2f}s | {note}"
    except Exception as e:
        return None, f"Error: {e}"


with gr.Blocks(title="Image Colorization — CV 2025.2") as demo:
    gr.Markdown("# Image Colorization")
    gr.Markdown(
        "Compare three colorization methods: **Deep Learning** (automatic), "
        "**Scribble-based** (user-guided), and **Example-based** (reference transfer)."
    )

    with gr.Tab("Deep Learning"):
        gr.Markdown("### Automatic colorization — Zhang et al. 2016 CNN")
        with gr.Row():
            with gr.Column():
                dl_input = gr.Image(label="Grayscale Image", type="numpy")
                dl_model = gr.Dropdown(
                    choices=["Zhang16 Pretrained", "Zhang16 Fine-tuned"],
                    value="Zhang16 Pretrained",
                    label="Model",
                )
                dl_btn = gr.Button("Colorize", variant="primary")
            with gr.Column():
                dl_output = gr.Image(label="Colorized Result", type="numpy")
                dl_info = gr.Textbox(label="Info", interactive=False)
        dl_btn.click(colorize_deep, inputs=[dl_input, dl_model], outputs=[dl_output, dl_info])

    with gr.Tab("Scribble-based"):
        gr.Markdown("### Levin 2004 — draw color hints on the image")
        gr.Markdown(
            "1. Upload a grayscale image.\n"
            "2. Pick a color and draw strokes on the areas you want colored.\n"
            "3. Click **Colorize**."
        )
        with gr.Row():
            with gr.Column():
                sc_editor = gr.ImageEditor(
                    label="Draw Color Hints",
                    type="numpy",
                    brush=gr.Brush(
                        colors=["#FF0000", "#00AA00", "#0000FF", "#FFFF00",
                                 "#FF8800", "#00CCCC", "#FF00FF", "#8B4513"],
                        default_size=8,
                    ),
                )
                sc_btn = gr.Button("Colorize", variant="primary")
            with gr.Column():
                sc_output = gr.Image(label="Colorized Result", type="numpy")
                sc_info = gr.Textbox(label="Info", interactive=False)
        sc_btn.click(colorize_scribble, inputs=[sc_editor], outputs=[sc_output, sc_info])

    with gr.Tab("Example-based"):
        gr.Markdown("### Welsh 2002 — transfer color from a reference image")

        with gr.Tab("Mode 1 — Auto"):
            gr.Markdown("Automatic global matching. Every target pixel searches the entire reference.")
            with gr.Row():
                with gr.Column():
                    ex_gray = gr.Image(label="Grayscale Target", type="numpy")
                    ex_ref = gr.Image(label="Color Reference", type="numpy")
                    ex_btn = gr.Button("Colorize", variant="primary")
                with gr.Column():
                    ex_output = gr.Image(label="Colorized Result", type="numpy")
                    ex_info = gr.Textbox(label="Info", interactive=False)
            ex_btn.click(colorize_example, inputs=[ex_gray, ex_ref],
                         outputs=[ex_output, ex_info])

        with gr.Tab("Mode 2 — Swatch (user-guided)"):
            gr.Markdown(
                "Draw the **same brush color** on corresponding regions in both images.  \n"
                "Each color defines one swatch pair — target pixels in that region will only "
                "match against the paired reference region.  \n"
                "🔴 Red · 🟢 Green · 🔵 Blue · 🟡 Yellow  *(up to 4 pairs)*"
            )
            _brush = gr.Brush(
                colors=["#DC3232", "#32B432", "#3232DC", "#DCC832"],
                default_size=14,
            )
            with gr.Row():
                with gr.Column():
                    ex2_gray = gr.ImageEditor(
                        label="Grayscale Target — draw region masks",
                        type="numpy",
                        brush=_brush,
                    )
                    ex2_ref = gr.ImageEditor(
                        label="Color Reference — draw matching masks",
                        type="numpy",
                        brush=_brush,
                    )
                    ex2_btn = gr.Button("Colorize with Swatches", variant="primary")
                with gr.Column():
                    ex2_output = gr.Image(label="Colorized Result", type="numpy")
                    ex2_info = gr.Textbox(label="Info", interactive=False)
            ex2_btn.click(colorize_example_swatch, inputs=[ex2_gray, ex2_ref],
                          outputs=[ex2_output, ex2_info])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
