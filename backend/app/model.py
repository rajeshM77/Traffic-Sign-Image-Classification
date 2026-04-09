import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL_PATH      = "models/traffic_sign_model.h5"
CLASS_NAMES_PATH = "models/class_names.json"

def load_model_and_classes():
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((32, 32))
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)   # shape (1, 32, 32, 3)

def generate_gradcam(model, img_array: np.ndarray, class_idx: int) -> str:
    """Returns a base64-encoded PNG of the Grad-CAM heatmap overlaid on the image."""
    import cv2, base64
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Find last conv layer
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # Resize heatmap to 32x32 and colorize
    heatmap_resized = cv2.resize(heatmap, (32, 32))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # drop alpha

    # Overlay on original image
    original = img_array[0]
    overlay = (heatmap_colored * 0.4 + original * 0.6)
    overlay = np.clip(overlay * 255, 0, 255).astype("uint8")

    # Encode to base64
    _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")
