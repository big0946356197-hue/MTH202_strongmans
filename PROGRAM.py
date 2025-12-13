import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io

# Page setup
st.set_page_config(
    page_title="SVD Image Compression & Memory Comparison",
    layout="wide"
)

st.title("SVD Image Compression")
st.write(
    """
โปรเจกต์นี้แสดงการบีบอัดภาพด้วย **Singular Value Decomposition (SVD)** 
"""
)

# Session state
if "compressed_images" not in st.session_state:
    st.session_state["compressed_images"] = []

# SVD functions
def compress_channel(channel, k):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    k = min(k, len(S))
    S_k = np.diag(S[:k])
    return U[:, :k] @ S_k @ Vt[:k, :]

def svd_compress_image(image, k):
    image = image.copy()
    image.thumbnail((900, 900))  # limit size for performance

    arr = np.array(image)

    R = compress_channel(arr[:, :, 0], k)
    G = compress_channel(arr[:, :, 1], k)
    B = compress_channel(arr[:, :, 2], k)

    result = np.stack([R, G, B], axis=2)
    result = np.clip(result, 0, 255).astype("uint8")

    return Image.fromarray(result)

# Memory calculation
def memory_comparison_table(image, k_list):
    """
    Create memory comparison table like the reference image
    """
    m, n, c = np.array(image).shape

    # original image: uint8
    orig_bytes = m * n * c

    rows = []

    for k in k_list:
        params = 3 * k * (m + n + 1)      # U + S + V for RGB
        svd_bytes = params * 8            # float64
        ratio = svd_bytes / orig_bytes

        rows.append({
            "k": k,
            "params(k)": params,
            "factors bytes": svd_bytes,
            "orig bytes": orig_bytes,
            "ratio vs orig": ratio
        })

    return pd.DataFrame(rows)

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

colA, colB = st.columns(2)

with colA:
    if uploaded:
        original_image = Image.open(uploaded).convert("RGB")
        st.image(original_image, caption="ภาพต้นฉบับ", width=350)

with colB:
    if uploaded and st.button("บีบอัดภาพด้วย SVD"):
        compressed_image = svd_compress_image(original_image, rank)

        st.session_state["compressed_images"].append({
            "name": uploaded.name,
            "rank": rank,
            "image": compressed_image
        })

        st.success("บีบอัดภาพสำเร็จ")

# Compressed images gallery
st.subheader("ภาพที่บีบอัดแล้ว")

compare_list = []

if len(st.session_state["compressed_images"]) == 0:
    st.info("ยังไม่มีภาพที่ถูกบีบอัด")
else:
    grid = st.columns(4)
    for i, item in enumerate(st.session_state["compressed_images"]):
        with grid[i % 4]:
            st.image(
                item["image"],
                caption=f"{item['name']} (k={item['rank']})",
                width=150
            )
            if st.checkbox("เลือกเพื่อเปรียบเทียบ", key=f"cmp_{i}"):
                compare_list.append(item)

# Image comparison section
st.subheader("การเปรียบเทียบภาพ")

if len(compare_list) == 0:
    st.info("เลือกภาพด้านบนเพื่อเปรียบเทียบ")
else:
    for img in compare_list:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.image(
                img["image"],
                caption=f"{img['name']} (k={img['rank']})",
                width=450
            )

        with col2:
            buf = io.BytesIO()
            img["image"].save(buf, format="JPEG")

            st.download_button(
                label="ดาวน์โหลดภาพ",
                data=buf.getvalue(),
                file_name=f"{img['name']}_k{img['rank']}.jpg",
                mime="image/jpeg",
                key=f"dl_{img['name']}_{img['rank']}"
            )

# Memory comparison table section
st.subheader("Memory Comparison")

if uploaded:
    k_values = [5, 20, 50, 100, 500, 2500]

    df = memory_comparison_table(original_image, k_values)

    st.dataframe(
        df.style.format({
            "params(k)": "{:,}",
            "factors bytes": "{:,}",
            "orig bytes": "{:,}",
            "ratio vs orig": "{:.3f}"
        }),
        use_container_width=True
    )

    st.caption(
        "ตารางนี้แสดงการเปรียบเทียบการใช้หน่วยความจำของการบีบอัดแบบ SVD "
        "เมื่อเลือกค่า Rank (k) ต่างกัน เทียบกับภาพต้นฉบับ"
    )

)
