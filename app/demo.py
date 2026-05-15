"""
Gradio demo for Multi-methods Image Colorization.

Tabs:
  1. Deep Learning  - automatic colorization (Zhang16, Zhang17, DeOldify)
  2. Scribble-based - user draws color hints with a brush
  3. Example-based  - transfer color from a reference image

Run:
  python app/demo.py
"""

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


def colorize_example(gray_image: np.ndarray, reference_image: np.ndarray):
    if gray_image is None or reference_image is None:
        return None, "Please upload both a grayscale image and a color reference image."
    try:
        colorizer = _load_example_colorizer()
        result, info = colorizer.colorize(gray_image, reference_image)
        return result, f"Time: {info['time_seconds']:.2f}s"
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
        with gr.Row():
            with gr.Column():
                ex_gray = gr.Image(label="Grayscale Target", type="numpy")
                ex_ref = gr.Image(label="Color Reference", type="numpy")
                ex_btn = gr.Button("Colorize", variant="primary")
            with gr.Column():
                ex_output = gr.Image(label="Colorized Result", type="numpy")
                ex_info = gr.Textbox(label="Info", interactive=False)
        ex_btn.click(
            colorize_example,
            inputs=[ex_gray, ex_ref],
            outputs=[ex_output, ex_info],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
