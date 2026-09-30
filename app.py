import streamlit as st

st.set_page_config(
    page_title="CropDoctorAI",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 CropDoctorAI")
st.subheader("AI-Powered Plant Disease Detection")

st.write(
    "Upload a plant image or use your camera to identify "
    "the plant, detect possible diseases, and get solutions."
)

# Language
language = st.selectbox(
    "🌐 Select Language",
    ["English", "தமிழ்"]
)

# Image input
st.subheader("📷 Plant Image")

uploaded_image = st.file_uploader(
    "Upload a plant image",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input("📸 Take a picture")

if uploaded_image is not None:
    st.image(
        uploaded_image,
        caption="Uploaded Plant Image",
        use_container_width=True
    )

elif camera_image is not None:
    st.image(
        camera_image,
        caption="Camera Image",
        use_container_width=True
    )

st.info("Upload an image or take a picture to start diagnosis.")
