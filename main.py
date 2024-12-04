import streamlit as st
from PIL import Image
from src.predict import predict

# Tiêu đề ứng dụng
st.title("Demo Streamlit với Ảnh và Văn bản")

# Input văn bản
text_input = st.text_input("Nhập văn bản của bạn:")

# Input ảnh
image_file = st.file_uploader("Chọn một file ảnh", type=["jpg", "png", "jpeg"])

# Kiểm tra và hiển thị văn bản nhập
if text_input:
    st.write(f"Văn bản bạn nhập là: {text_input}")

# Kiểm tra và hiển thị ảnh
if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Ảnh của bạn", use_container_width=True)

result = None
# Nút dự đoán
if st.button("Dự đoán"):
    result = predict(text_input, image_file)


# Hiển thị kết quả dự đoán
st.subheader("Kết quả dự đoán:")
st.write(result)