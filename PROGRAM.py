import streamlit as st
import numpy as np
from PIL import Image
import io

# Page setup
st.set_page_config(
    page_title="SVD Image Compression",
    layout="wide"
)

st.title("SVD Image Compression")
st.write(
    """
โปรเจกต์นี้แสดงการบีบอัดภาพด้วย **Singular Value Decomposition (SVD)**
พร้อมการเปรียบเทียบการใช้หน่วยความจำของภาพ
"""
)

# Session state
if "compressed_image" not in st.session_state:
    st.session_state["compressed_image"] = None

# SVD functions
def compress_channel(channel, k):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    k = min(k, len(S))
    S_k = np.diag(S[:k])
    return U[:, :k] @ S_k @ Vt[:k, :]

def svd_compress_image(image, k):
    image = image.copy()
    image.thumbnail((900, 900))

    arr = np.array(image)

    R = compress_channel(arr[:, :, 0], k)
    G = compress_channel(arr[:, :, 1], k)
    B = compress_channel(arr[:, :, 2], k)

    result = np.stack([R, G, B], axis=2)
    result = np.clip(result, 0, 255).astype("uint8")

    return Image.fromarray(result)

# Memory calculation
def calculate_memory(image, k):
    m, n, c = np.array(image).shape

    original_bytes = m * n * c
    svd_bytes = 3 * k * (m + n + 1) * 8

    ratio = svd_bytes / original_bytes

    return original_bytes, svd_bytes, ratio

# Upload & controls
st.subheader("อัปโหลดภาพและเลือก Rank")

uploaded = st.file_uploader(
    "เลือกไฟล์รูปภาพ (JPEG / PNG)",
    type=["jpg", "jpeg", "png"]
)

rank = st.slider(
    "เลือก Rank สำหรับการบีบอัด",
    min_value=5,
    max_value=300,
    value=50
)

if uploaded:
    original_image = Image.open(uploaded).convert("RGB")

    if st.button("บีบอัดภาพ"):
        compressed_image = svd_compress_image(original_image, rank)
        st.session_state["compressed_image"] = compressed_image

# Image & memory comparison
if uploaded and st.session_state["compressed_image"] is not None:
    st.subheader("การเปรียบเทียบภาพและหน่วยความจำ")

    col1, col2 = st.columns(2)

    with col1:
        st.image(original_image, caption="Original Image", width=400)

        orig_bytes, _, _ = calculate_memory(original_image, rank)
        st.write(f"Memory: {orig_bytes / 1024:.2f} KB")

    with col2:
        st.image(
            st.session_state["compressed_image"],
            caption=f"SVD Image (rank = {rank})",
            width=400
        )

        _, svd_bytes, ratio = calculate_memory(original_image, rank)
        st.write(f"Memory: {svd_bytes / 1024:.2f} KB")
        st.write(f"Ratio vs original: {ratio:.3f}")

        buf = io.BytesIO()
        st.session_state["compressed_image"].save(buf, format="JPEG")

        st.download_button(
            label="ดาวน์โหลดภาพ",
            data=buf.getvalue(),
            file_name=f"svd_rank_{rank}.jpg",
            mime="image/jpeg"
        )
