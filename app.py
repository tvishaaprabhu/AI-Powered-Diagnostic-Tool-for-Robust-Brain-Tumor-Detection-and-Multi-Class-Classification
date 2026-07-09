import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import io
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
        mask = processed_eq > 15
        brain_pixels = processed_eq[mask]
        brain_eq = cv2.equalizeHist(brain_pixels.reshape(-1, 1))
        processed_eq[mask] = brain_eq.ravel()
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
        MODEL_PATH = "brain_tumor_classifier.keras"
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
        # --- 6. MEDSAM SEGMENTATION ---
        # ==========================================
        if detected_class != "notumor":
            st.header("6. Tumor Segmentation (MedSAM)")

            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            h_orig, w_orig = img_rgb.shape[:2]

            st.caption(f"A **{class_name}** was detected. Use the Grad-CAM heatmap above to identify the tumor region, then enter its bounding box coordinates below.")
            st.image(img_rgb, caption=f"Scan — {w_orig} x {h_orig} px", use_container_width=True)

            st.info("Enter the bounding box coordinates around the tumor region, then click **Run MedSAM**.")

            col_a, col_b = st.columns(2)
            with col_a:
                x_min = st.number_input("X min", min_value=0, max_value=w_orig, value=int(w_orig * 0.2))
                y_min = st.number_input("Y min", min_value=0, max_value=h_orig, value=int(h_orig * 0.2))
            with col_b:
                x_max = st.number_input("X max", min_value=0, max_value=w_orig, value=int(w_orig * 0.8))
                y_max = st.number_input("Y max", min_value=0, max_value=h_orig, value=int(h_orig * 0.8))

            # Show preview of bounding box on scan
            preview_img = img_rgb.copy()
            cv2.rectangle(preview_img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
            st.image(preview_img, caption="Bounding box preview", use_container_width=True)

            run_medsam = st.button("Run MedSAM")

            if run_medsam:
                if x_min >= x_max or y_min >= y_max:
                    st.warning("Invalid bounding box — make sure X min < X max and Y min < Y max.")
                else:
                    bbox = np.array([x_min, y_min, x_max, y_max])
                    st.caption(f"Bounding box: {bbox.tolist()}")

                    try:
                        import torch
                        from segment_anything import sam_model_registry

                        checkpoint_path = "medsam_vit_b.pth"
                        if not os.path.exists(checkpoint_path):
                            st.error("MedSAM checkpoint not found. Make sure `medsam_vit_b.pth` is in the repo root.")
                        else:
                            with st.spinner("Running MedSAM segmentation..."):
                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                                medsam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
                                medsam.to(device)
                                medsam.eval()

                                img_1024 = cv2.resize(img_rgb, (1024, 1024))
                                img_1024_normalized = (img_1024 - img_1024.min()) / (img_1024.max() - img_1024.min() + 1e-10)
                                img_tensor = torch.tensor(img_1024_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

                                scale_x_sam = 1024 / w_orig
                                scale_y_sam = 1024 / h_orig
                                scaled_bbox = bbox * np.array([scale_x_sam, scale_y_sam, scale_x_sam, scale_y_sam])
                                box_tensor = torch.tensor(scaled_bbox, dtype=torch.float32).unsqueeze(0).to(device)

                                with torch.no_grad():
                                    image_embedding = medsam.image_encoder(img_tensor)
                                    sparse_embeddings, dense_embeddings = medsam.prompt_encoder(
                                        points=None, boxes=box_tensor, masks=None
                                    )
                                    low_res_masks, _ = medsam.mask_decoder(
                                        image_embeddings=image_embedding,
                                        image_pe=medsam.prompt_encoder.get_dense_pe(),
                                        sparse_prompt_embeddings=sparse_embeddings,
                                        dense_prompt_embeddings=dense_embeddings,
                                        multimask_output=False,
                                    )

                                low_res_np = low_res_masks.squeeze().cpu().numpy()
                                mask = cv2.resize(
                                    (low_res_np > 0.0).astype(np.uint8),
                                    (w_orig, h_orig),
                                    interpolation=cv2.INTER_NEAREST
                                )

                                box_img = img_rgb.copy()
                                cv2.rectangle(box_img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)

                                overlay_img = img_rgb.copy()
                                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if np.sum(mask) > 0:
                                    color_mask = np.zeros_like(overlay_img)
                                    color_mask[mask > 0] = [255, 0, 162]
                                    overlay_img = cv2.addWeighted(overlay_img, 0.75, color_mask, 0.25, 0)
                                    cv2.drawContours(overlay_img, contours, -1, (0, 255, 0), 2)

                            st.subheader("Segmentation Result")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(box_img, caption="1. Your Bounding Box Prompt", use_container_width=True)
                            with col2:
                                st.image(overlay_img, caption="2. MedSAM Target Segmentation", use_container_width=True)

                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                tumor_pixel_count = np.sum(mask == 1)
                                tumor_area = tumor_pixel_count * (pixel_spacing_mm ** 2)
                                _, _, w_c, h_c = cv2.boundingRect(largest_contour)
                                width_val = w_c * pixel_spacing_mm
                                height_val = h_c * pixel_spacing_mm
                                (_, _), radius = cv2.minEnclosingCircle(largest_contour)
                                max_dia_val = (radius * 2) * pixel_spacing_mm
                                unit = "mm" if pixel_spacing_mm != 1.0 else "pixels"

                                st.subheader("Tumor Metrics")
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("Area", f"{tumor_area:.2f} {unit}²")
                                m2.metric("Width", f"{width_val:.2f} {unit}")
                                m3.metric("Height", f"{height_val:.2f} {unit}")
                                m4.metric("Max Diameter", f"{max_dia_val:.2f} {unit}")

                            buf_seg = io.BytesIO()
                            Image.fromarray(overlay_img).save(buf_seg, format="PNG")
                            st.download_button(
                                label="Download Segmentation Result",
                                data=buf_seg.getvalue(),
                                file_name="segmentation_" + uploaded_file.name.rsplit(".", 1)[0] + ".png",
                                mime="image/png"
                            )

                    except ImportError:
                        st.error("MedSAM not installed. Add segment-anything and medsam to requirements.txt.")
                    except Exception as e:
                        st.error(f"MedSAM error: {e}")

        else:
            st.success("Clear scan — MedSAM segmentation not required.")

else:
    st.info("Please upload an image to see it displayed here.")
