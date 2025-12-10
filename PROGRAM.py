import streamlit as st
import numpy as np
from PIL import Image
import io

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="SVD Image Compression")

st.title("SVD Image Compression")
st.write("เว็บไซต์นี้เป็นส่วนหนึ่งของโปรเจกต์สำหรับแสดงการบีบอัดภาพด้วยเทคนิค Singular Value Decomposition (SVD)")

# ตัวเก็บรูปใน Session
if "compressed_images" not in st.session_state:
    st.session_state["compressed_images"] = []

# ฟังก์ชัน SVD
def compress_channel(channel, k):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    S_k = np.diag(S[:k])
    return U[:, :k] @ S_k @ Vt[:k, :]

# ลดขนาดรูปอัตโนมัติ (เพื่อให้เว็บทำงานเร็วขึ้น)
def svd_compress_image(image, k):

    max_size = 900
    image.thumbnail((max_size, max_size))

    arr = np.array(image)

    R = compress_channel(arr[:, :, 0], k)
    G = compress_channel(arr[:, :, 1], k)
    B = compress_channel(arr[:, :, 2], k)

    result = np.stack([R, G, B], axis=2)
    result = np.clip(result, 0, 255).astype("uint8")

    return Image.fromarray(result)

# ส่วนอัปโหลดและบีบอัดภาพ
uploaded = st.file_uploader("เลือกไฟล์รูปภาพ (JPEG/PNG)", type=["jpg", "jpeg", "png"])
rank = st.slider("เลือกค่า Rank สำหรับการบีบอัด", 5, 300, 50)

colA, colB = st.columns([1, 1])

with colA:
    if uploaded:
        original = Image.open(uploaded).convert("RGB")
        st.image(original, caption="ภาพต้นฉบับ", width=350)

with colB:
    if uploaded and st.button("บีบอัดภาพ"):
        compressed = svd_compress_image(original, rank)
        st.session_state["compressed_images"].append({
            "name": uploaded.name,
            "rank": rank,
            "image": compressed
        })
        st.success("บีบอัดภาพสำเร็จ!")

# แสดงรายการภาพที่บีบอัดแล้ว
st.subheader("ภาพที่บีบอัดแล้ว")

compare_list = []

if len(st.session_state["compressed_images"]) == 0:
    st.info("ยังไม่มีภาพที่ถูกบีบอัด")
else:
    grid = st.columns(4)
    for i, item in enumerate(st.session_state["compressed_images"]):
        with grid[i % 4]:
            st.image(item["image"], caption=f"{item['name']} (rank={item['rank']})", width=150)
            if st.checkbox("เลือกเพื่อเปรียบเทียบ", key=f"compare_{i}"):
                compare_list.append(item)

# ส่วนแสดงเปรียบเทียบภาพ
st.subheader("การเปรียบเทียบภาพ")

if len(compare_list) > 0:
    for img in compare_list:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.image(img["image"], caption=f"{img['name']} (rank={img['rank']})", width=450)

        with col2:
            buf = io.BytesIO()
            img["image"].save(buf, format="JPEG")
            st.download_button(
                label="ดาวน์โหลดภาพ",
                data=buf.getvalue(),
                file_name=f"{img['name']}_rank{img['rank']}.jpg",
                mime="image/jpeg",
                key=f"dl_{img['name']}_{img['rank']}"
            )
else:
    st.info("เลือกภาพด้านบนเพื่อแสดงผลการเปรียบเทียบ")

