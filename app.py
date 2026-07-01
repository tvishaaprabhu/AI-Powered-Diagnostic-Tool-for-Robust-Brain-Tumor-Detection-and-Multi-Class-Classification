import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import io
import pydicom
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import auth
import admin

st.set_page_config(page_title="Medical Image Viewer", layout="wide")

# ==========================================
# --- 1. SESSION STATE INITIALIZATION ---
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

def switch_to_reset():
    st.session_state.auth_mode = "reset"

def switch_to_login():
    st.session_state.auth_mode = "login"

# ==========================================
# --- 2. AUTHENTICATION GATEWAY ---
# ==========================================
if not st.session_state.authenticated:
    st.title("Welcome to the Medical Image Viewer")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.auth_mode == "login":
            st.subheader("System Login")
            with st.form("login_form"):
                email = st.text_input("Email Address")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log In")

                if submitted:
                    success, message, role = auth.authenticate_user(email, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.current_user = email
                        st.session_state.role = role
                        st.rerun()
                    else:
                        st.error(message)

            st.button("Forgot Password?", on_click=switch_to_reset)

        elif st.session_state.auth_mode == "reset":
            st.subheader("Reset Password")
            with st.form("reset_form"):
                reset_email = st.text_input("Enter your registered email")
                submit_reset = st.form_submit_button("Reset Password")

                if submit_reset:
                    success, result = auth.reset_user_password(reset_email)
                    if success:
                        st.success("Password reset!")
                        st.code(f"Your temporary password is: {result}")
                    else:
                        st.error(result)

            st.button("Back to Login", on_click=switch_to_login)

    st.stop()

# ==========================================
# --- 3. MAIN APPLICATION ROUTING ---
# ==========================================
with st.sidebar:
    st.write(f"Logged in as: **{st.session_state.current_user}**")
    st.write(f"Role: **{st.session_state.role}**")

    if st.session_state.role == "Admin":
        st.divider()
        app_mode = st.radio("Navigation", ["Image Viewer", "Admin Dashboard"])
        st.divider()
    else:
        app_mode = "Image Viewer"

    if st.button("Log Out"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.role = None
        st.rerun()

if app_mode == "Admin Dashboard":
    admin.show_admin_dashboard()
else:
    # ==========================================
    # --- 4. IMAGE VIEWER MODULE ---
    # ==========================================
    st.title("My Streamlit Image Viewer")

    st.header("1. Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp", "dcm"])

    if uploaded_file is not None:
        is_dicom = uploaded_file.name.endswith(".dcm")

        if is_dicom:
            dicom = pydicom.dcmread(uploaded_file)
            img_array = dicom.pixel_array.squeeze()
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
                    uploaded_file.name, width, height, "DICOM",
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

        else:
            img = Image.open(uploaded_file)
            img_array = np.array(img.convert("L")).squeeze()
            format_label = img.format
            width, height = img.size

            st.subheader("Image Summary")
            data = {
                "Attribute": ["File Name", "Width (pixels)", "Height (pixels)", "Format"],
                "Value": [uploaded_file.name, width, height, format_label]
            }
            df = pd.DataFrame(data)
            st.dataframe(df, hide_index=True, use_container_width=False)

        st.divider()

        # --- SECTION 2: IMAGE PREPROCESSING ---
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

        # --- SECTION 3: DENOISING ---
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

        # --- SECTION 4: K-MEANS CLUSTERING ---
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

    else:
        st.info("Please upload an image to see it displayed here.")
