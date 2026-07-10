import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import io
import base64
import json
import pydicom
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="Medical Image Viewer", layout="wide")
st.title("My Streamlit Image Viewer")

# ==========================================
# --- 1. UPLOAD IMAGE ---
# ==========================================
st.header("1. Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp", "dcm"])

if uploaded_file is not None:
    is_dicom = uploaded_file.name.endswith(".dcm")

    if is_dicom:
        dicom = pydicom.dcmread(uploaded_file)
        pixel_array = dicom.pixel_array.squeeze()

        if len(pixel_array.shape) == 3:
            n_slices = pixel_array.shape[0]
            st.subheader("Multi-Slice DICOM")
            if "slice_idx" not in st.session_state:
                st.session_state.slice_idx = 0
            col_prev, col_slider, col_next = st.columns([1, 8, 1])
            with col_prev:
                if st.button("◀"):
                    st.session_state.slice_idx = max(0, st.session_state.slice_idx - 1)
            with col_slider:
                st.session_state.slice_idx = st.slider("Slice", 0, n_slices - 1, st.session_state.slice_idx)
            with col_next:
                if st.button("▶"):
                    st.session_state.slice_idx = min(n_slices - 1, st.session_state.slice_idx + 1)
            st.caption(f"Slice {st.session_state.slice_idx + 1} of {n_slices}")
            img_array = pixel_array[st.session_state.slice_idx]
        else:
            img_array = pixel_array

        img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        width, height = img_array.shape[1], img_array.shape[0]

        st.subheader("Image Summary")
        dicom_data = {
            "Attribute": [
                "File Name", "Width (pixels)", "Height (pixels)", "Format",
                "Patient Name", "Patient ID", "Modality", "Study Date",
                "Institution", "Manufacturer", "Rows", "Columns",
                "Pixel Spacing", "Slice Thickness", "Bits Stored"
            ],
            "Value": [
                str(uploaded_file.name), str(width), str(height), "DICOM",
                str(getattr(dicom, "PatientName", "N/A")),
                str(getattr(dicom, "PatientID", "N/A")),
                str(getattr(dicom, "Modality", "N/A")),
                str(getattr(dicom, "StudyDate", "N/A")),
                str(getattr(dicom, "InstitutionName", "N/A")),
                str(getattr(dicom, "Manufacturer", "N/A")),
                str(getattr(dicom, "Rows", "N/A")),
                str(getattr(dicom, "Columns", "N/A")),
                str(getattr(dicom, "PixelSpacing", "N/A")),
                str(getattr(dicom, "SliceThickness", "N/A")),
                str(getattr(dicom, "BitsStored", "N/A")),
            ]
        }
        df = pd.DataFrame(dicom_data)
        st.dataframe(df, hide_index=True, use_container_width=False)

        try:
            pixel_spacing_mm = float(dicom.PixelSpacing[0])
        except Exception:
            pixel_spacing_mm = 1.0

    else:
        img = Image.open(uploaded_file)
        img_array = np.array(img.convert("L")).squeeze()
        format_label = str(img.format)
        width, height = img.size
        pixel_spacing_mm = 1.0

        st.subheader("Image Summary")
        data = {
            "Attribute": ["File Name", "Width (pixels)", "Height (pixels)", "Format"],
            "Value": [str(uploaded_file.name), str(width), str(height), format_label]
        }
        df = pd.DataFrame(data)
        st.dataframe(df, hide_index=True, use_container_width=False)

    st.divider()

    # ==========================================
    # --- 2. IMAGE PREPROCESSING ---
    # ==========================================
    st.header("2. Image Preprocessing")
    normalize = st.checkbox("Normalize (0-255)")
    rescale = st.checkbox("Rescale (0-1)")
    equalize = st.checkbox("Histogram Equalization (brain-masked)")
    flip = st.checkbox("Horizontal Flip")
    rotate = st.checkbox("Rotate 90°")

    preprocessed = img_array.copy()

    if normalize:
        preprocessed = cv2.normalize(preprocessed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if rescale:
        preprocessed = (preprocessed / 255.0 * 255).astype(np.uint8)
    if equalize:
        processed_eq = preprocessed.copy()
        mask_eq = processed_eq > 15
        brain_pixels = processed_eq[mask_eq]
        brain_eq = cv2.equalizeHist(brain_pixels.reshape(-1, 1))
        processed_eq[mask_eq] = brain_eq.ravel()
        preprocessed = processed_eq
    if flip:
        preprocessed = cv2.flip(preprocessed, 1)
    if rotate:
        preprocessed = cv2.rotate(preprocessed, cv2.ROTATE_90_CLOCKWISE)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(img_array, use_container_width=True)
    with col2:
        st.subheader("Preprocessed")
        st.image(preprocessed, use_container_width=True)

    buf1 = io.BytesIO()
    Image.fromarray(preprocessed).save(buf1, format="PNG")
    st.download_button(
        label="Download Preprocessed Image",
        data=buf1.getvalue(),
        file_name="preprocessed_" + uploaded_file.name.rsplit(".", 1)[0] + ".png",
        mime="image/png"
    )

    st.divider()

    # ==========================================
    # --- 3. DENOISING ---
    # ==========================================
    st.header("3. Denoising")
    gaussian = st.checkbox("Gaussian Blur")
    if gaussian:
        gaussian_k = st.slider("Gaussian Kernel Size", min_value=1, max_value=15, value=5, step=2)

    median = st.checkbox("Median Filter")
    if median:
        median_k = st.slider("Median Kernel Size", min_value=1, max_value=15, value=5, step=2)

    nlm = st.checkbox("Non-Local Means Denoising")
    if nlm:
        nlm_h = st.slider("NLM Filter Strength (h)", min_value=1, max_value=30, value=10)

    denoised = preprocessed.copy()

    if gaussian:
        denoised = cv2.GaussianBlur(denoised, (gaussian_k, gaussian_k), 0)
    if median:
        denoised = cv2.medianBlur(denoised, median_k)
    if nlm:
        denoised = cv2.fastNlMeansDenoising(denoised, h=nlm_h)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Preprocessed")
        st.image(preprocessed, use_container_width=True)
    with col2:
        st.subheader("Denoised")
        st.image(denoised, use_container_width=True)

    buf2 = io.BytesIO()
    Image.fromarray(denoised).save(buf2, format="PNG")
    st.download_button(
        label="Download Denoised Image",
        data=buf2.getvalue(),
        file_name="denoised_" + uploaded_file.name.rsplit(".", 1)[0] + ".png",
        mime="image/png"
    )

    st.divider()

    # ==========================================
    # --- 4. K-MEANS CLUSTERING ---
    # ==========================================
    st.header("4. K-Means Clustering")
    k = st.slider("Number of clusters (K)", min_value=2, max_value=20, value=10)
    run_kmeans = st.button("Run K-Means")

    if run_kmeans:
        with st.spinner("Running K-Means..."):
            pixel_list = denoised.reshape((-1, 1)).astype(np.float32)
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = km.fit_predict(pixel_list)
            segmented_img = labels.reshape(denoised.shape)

            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(segmented_img, cmap='nipy_spectral')
            plt.colorbar(im, ax=ax, label='Cluster ID')
            ax.set_title(f"KMeans Anatomical Mapping (K={k})", fontsize=12)
            ax.axis('off')

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig)

            buf3 = io.BytesIO()
            fig.savefig(buf3, format="PNG", bbox_inches='tight')
            st.download_button(
                label="Download K-Means Image",
                data=buf3.getvalue(),
                file_name="kmeans_" + uploaded_file.name.rsplit(".", 1)[0] + ".png",
                mime="image/png"
            )

    st.divider()

    # ==========================================
    # --- 5. AI DIAGNOSIS ---
    # ==========================================
    st.header("5. AI Diagnosis")

    try:
        from predict import load_model, preprocess, predict, get_gradcam, overlay_gradcam, CLASS_NAMES
        MODEL_PATH = "brain_tumor_detector.keras"
        model = load_model(MODEL_PATH)
        model_loaded = True
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.info("Make sure `brain_tumor_classifier.keras` and `predict.py` are in the same folder as `app.py`.")
        model_loaded = False

    if model_loaded:
        input_tensor = preprocess(img_array)
        gray_128 = cv2.resize(img_array, (128, 128)).astype(np.float32) / 255.0

        with st.spinner("Running diagnosis..."):
            class_name, confidence, all_probs = predict(model, input_tensor)

        detected_class = class_name.lower().replace(" ", "")

        st.subheader("Prediction")
        if detected_class == "notumor":
            st.success(f"**{class_name}** — {confidence*100:.1f}% confidence")
        else:
            st.warning(f"**{class_name}** — {confidence*100:.1f}% confidence")

        st.subheader("Class Probabilities")
        for idx, name in CLASS_NAMES.items():
            st.write(name)
            st.progress(float(all_probs[idx]))
            st.caption(f"{all_probs[idx]*100:.1f}%")

        st.subheader("Grad-CAM Heatmap")
        st.caption("Highlights which regions of the scan influenced the prediction most.")

        with st.spinner("Generating Grad-CAM..."):
            class_idx = int(np.argmax(all_probs))
            heatmap = get_gradcam(model, input_tensor, class_idx)
            overlaid = overlay_gradcam(gray_128, heatmap)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(gray_128, caption="Original Scan", clamp=True, use_container_width=True)
        with col2:
            st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)
        with col3:
            st.image(overlaid, caption="Overlay", use_container_width=True)

        st.divider()

        # ==========================================
        # --- 6. BOUNDING BOX ---
        # ==========================================
        if detected_class != "notumor":
            st.header("6. Draw Bounding Box")
            st.caption(f"A **{class_name}** was detected. Draw a bounding box around the tumor, then click **Confirm Bounding Box** in the canvas — coordinates will appear below.")

            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            h_orig, w_orig = img_rgb.shape[:2]

            DISPLAY_W = 512
            display_h = int(h_orig * DISPLAY_W / w_orig)
            display_img = Image.fromarray(img_rgb).resize((DISPLAY_W, display_h))
            buf_b64 = io.BytesIO()
            display_img.save(buf_b64, format="PNG")
            img_b64 = base64.b64encode(buf_b64.getvalue()).decode()

            scale_x = round(w_orig / DISPLAY_W, 4)
            scale_y = round(h_orig / display_h, 4)

            canvas_html = f"""
            <div style="font-family:sans-serif;">
                <p style="font-size:13px;color:#555;margin-bottom:6px;">
                    Click and drag to draw a box. Then click <b>Confirm Bounding Box</b>.
                </p>
                <canvas id="bbox_canvas" width="{DISPLAY_W}" height="{display_h}"
                    style="border:2px solid #00bfff;cursor:crosshair;display:block;"></canvas>
                <div id="coord_display" style="font-family:monospace;font-size:12px;
                    color:#333;margin-top:6px;min-height:18px;"></div>
                <div style="margin-top:8px;">
                    <button onclick="confirmBox()" style="padding:8px 20px;
                        background:#00bfff;color:white;border:none;border-radius:4px;
                        cursor:pointer;font-size:14px;font-weight:bold;">
                        Confirm Bounding Box
                    </button>
                    <span id="confirm_msg" style="margin-left:12px;color:green;
                        font-weight:bold;font-size:13px;"></span>
                </div>
                <div id="result_display" style="font-family:monospace;font-size:13px;
                    color:#006600;margin-top:8px;font-weight:bold;min-height:20px;"></div>
            </div>

            <script>
            (function() {{
                const canvas = document.getElementById('bbox_canvas');
                const ctx = canvas.getContext('2d');
                const img = new Image();
                let startX, startY, isDrawing = false;
                let box = null;

                img.onload = () => ctx.drawImage(img, 0, 0);
                img.src = 'data:image/png;base64,{img_b64}';

                canvas.addEventListener('mousedown', e => {{
                    const r = canvas.getBoundingClientRect();
                    startX = e.clientX - r.left;
                    startY = e.clientY - r.top;
                    isDrawing = true;
                }});

                canvas.addEventListener('mousemove', e => {{
                    if (!isDrawing) return;
                    const r = canvas.getBoundingClientRect();
                    const cx = e.clientX - r.left;
                    const cy = e.clientY - r.top;
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);
                    ctx.strokeStyle = '#00ffff';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 3]);
                    ctx.strokeRect(startX, startY, cx - startX, cy - startY);
                    box = {{
                        x1: Math.round(Math.min(startX, cx)),
                        y1: Math.round(Math.min(startY, cy)),
                        x2: Math.round(Math.max(startX, cx)),
                        y2: Math.round(Math.max(startY, cy))
                    }};
                    document.getElementById('coord_display').innerText =
                        'Drawing: (' + box.x1 + ', ' + box.y1 + ') → (' + box.x2 + ', ' + box.y2 + ')';
                }});

                canvas.addEventListener('mouseup', () => {{ isDrawing = false; }});

                window.confirmBox = function() {{
                    if (!box) {{ alert('Draw a box first.'); return; }}
                    const scaled = {{
                        x1: Math.round(box.x1 * {scale_x}),
                        y1: Math.round(box.y1 * {scale_y}),
                        x2: Math.round(box.x2 * {scale_x}),
                        y2: Math.round(box.y2 * {scale_y})
                    }};
                    // Store in sessionStorage so Streamlit text input can pick it up
                    sessionStorage.setItem('bbox_confirmed', JSON.stringify(scaled));
                    document.getElementById('confirm_msg').innerText = '✅ Box confirmed!';
                    document.getElementById('result_display').innerText =
                        'Coordinates (original px): x_min=' + scaled.x1 +
                        ' y_min=' + scaled.y1 + ' x_max=' + scaled.x2 + ' y_max=' + scaled.y2;
                    // Auto-fill the hidden input and submit
                    const inp = window.parent.document.querySelector('input[data-testid="stTextInput"]');
                    if (inp) {{
                        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                        nativeInputValueSetter.call(inp, JSON.stringify(scaled));
                        inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    }}
                }};
            }})();
            </script>
            """

            components.html(canvas_html, height=display_h + 130)

            st.caption("After confirming the box above, paste the coordinates here and press Enter:")
            bbox_input = st.text_input(
                "Bounding box (auto-filled after confirm, or paste manually):",
                placeholder='{"x1":50,"y1":60,"x2":150,"y2":180}',
                label_visibility="collapsed"
            )

            bbox = None
            if bbox_input:
                try:
                    coords = json.loads(bbox_input)
                    bbox = [int(coords["x1"]), int(coords["y1"]), int(coords["x2"]), int(coords["y2"])]
                    preview = img_rgb.copy()
                    cv2.rectangle(preview, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 3)
                    st.image(preview, caption=f"Preview — box: {bbox}", use_container_width=True)
                    st.success(f"Bounding box ready: x_min={bbox[0]}, y_min={bbox[1]}, x_max={bbox[2]}, y_max={bbox[3]}")
                except Exception:
                    st.warning("Could not parse coordinates. Make sure the JSON is valid.")

        else:
            st.success("Clear scan — no tumor detected.")

else:
    st.info("Please upload an image to see it displayed here.")
