# predict.py
# Drop this file in the same folder as app.py
# Import in app.py with: from predict import load_model, predict, get_gradcam

import numpy as np
import cv2
import tensorflow as tf
import streamlit as st

# Class order matches what flow_from_directory assigns alphabetically
CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary Tumor"
}

# Grad-CAM target layers for each backbone in the stacking model
VGG_LAYER = "block5_conv3"
RESNET_LAYER = "conv5_block3_out"


@st.cache_resource
def load_model(model_path: str):
    """Load the trained ensemble model once and cache it."""
    model = tf.keras.models.load_model(model_path, compile=False)
    model.build((None, 128, 128, 1))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def preprocess(image_array: np.ndarray) -> np.ndarray:
    """
    Takes a 2D or 3D numpy array (PIL image converted to numpy),
    resizes to 128x128, converts to grayscale, normalizes to [0,1].
    Returns shape (1, 128, 128, 1) — ready for model input.
    """
    # Convert to grayscale if RGB
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)

    resized = cv2.resize(image_array, (128, 128))
    normalized = resized.astype(np.float32) / 255.0
    # Shape: (1, 128, 128, 1)
    return np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)


def predict(model, input_tensor: np.ndarray) -> tuple[str, float, np.ndarray]:
    """
    Run forward pass. Returns:
    - predicted class name (str)
    - confidence score (float, 0–1)
    - full probability array (np.ndarray of shape (4,))
    """
    probs = model.predict(input_tensor, verbose=0)[0]  # shape (4,)
    class_idx = int(np.argmax(probs))
    return CLASS_NAMES[class_idx], float(probs[class_idx]), probs


def get_gradcam(model, input_tensor: np.ndarray, class_idx: int) -> np.ndarray:
    """
    Generates a blended Grad-CAM heatmap by averaging activations
    from VGG16 (block5_conv3) and ResNet50 (conv5_block3_out).
    Returns a colorized RGB heatmap as uint8 numpy array (128x128x3).
    """
    def _single_gradcam(layer_name: str) -> np.ndarray:
        # Build a sub-model that outputs (target_layer, final_output)
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            layer_output, predictions = grad_model(input_tensor)
            target_class_score = predictions[:, class_idx]

        # Gradients of target class score w.r.t. conv layer output
        grads = tape.gradient(target_class_score, layer_output)

        # Pool gradients across spatial dimensions → importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the channels by their importance
        layer_output = layer_output[0]
        heatmap = layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

        # ReLU + normalize to [0,1]
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    # Get heatmaps from both backbones
    vgg_heatmap = _single_gradcam(VGG_LAYER)
    resnet_heatmap = _single_gradcam(RESNET_LAYER)

    # Resize both to 128x128
    vgg_heatmap = cv2.resize(vgg_heatmap, (128, 128))
    resnet_heatmap = cv2.resize(resnet_heatmap, (128, 128))

    # Blend equally (50/50) from both backbones
    blended = cv2.addWeighted(vgg_heatmap, 0.5, resnet_heatmap, 0.5, 0)

    # Colorize with JET colormap
    blended_uint8 = np.uint8(255 * blended)
    colorized = cv2.applyColorMap(blended_uint8, cv2.COLORMAP_JET)
    colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

    return colorized_rgb


def overlay_gradcam(original_gray: np.ndarray, heatmap_rgb: np.ndarray, alpha=0.4) -> np.ndarray:
    """
    Blends the original grayscale scan with the Grad-CAM heatmap for display.
    original_gray: 2D array (128x128)
    heatmap_rgb: 3D array (128x128x3)
    Returns: blended RGB image (128x128x3) as uint8
    """
    # Convert grayscale to RGB for blending
    orig_uint8 = np.uint8(original_gray * 255)
    orig_rgb = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(orig_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay
