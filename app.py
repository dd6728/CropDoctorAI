import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Crop Doctor AI",
    page_icon="🌱",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 Crop Doctor AI")

st.write(
    "AI-powered preliminary crop disease screening."
)


# ============================================================
# PROBLEM
# ============================================================

with st.expander("⚠️ Problem", expanded=True):

    st.write(
        """
        Farmers may find it difficult to identify crop diseases
        at an early stage. Similar symptoms can make disease
        identification difficult, especially when immediate
        access to agricultural experts is unavailable.
        """
    )


# ============================================================
# SOLUTION
# ============================================================

with st.expander("💡 Solution", expanded=True):

    st.write(
        """
        Crop Doctor AI allows users to capture a crop image
        using a camera or upload an existing image.

        The AI model analyzes the image and provides a
        preliminary plant-health classification and confidence
        score in English or Tamil.
        """
    )


# ============================================================
# MODEL
# ============================================================

MODEL_ID = "ayerr/plant-disease-classification"


@st.cache_resource
def load_model():

    processor = ViTImageProcessor.from_pretrained(
        MODEL_ID
    )

    model = ViTForImageClassification.from_pretrained(
        MODEL_ID
    )

    model.eval()

    return processor, model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    processor, model = load_model()

    st.success("✅ AI model loaded successfully!")

except Exception as e:

    st.error("❌ AI model loading failed.")

    st.exception(e)

    st.stop()


# ============================================================
# LANGUAGE
# ============================================================

language = st.radio(
    "Choose Language / மொழியை தேர்வு செய்யவும்",
    ["English", "Tamil"],
    horizontal=True
)


# ============================================================
# IMAGE SOURCE
# ============================================================

st.subheader("📷 Crop Image")

input_method = st.radio(
    "Choose Image Source / படத்தை தேர்வு செய்யவும்",
    ["Upload Image", "Use Camera"],
    horizontal=True
)


uploaded_file = None


# ============================================================
# UPLOAD
# ============================================================

if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "📁 Upload Crop Image",
        type=["jpg", "jpeg", "png"]
    )


# ============================================================
# CAMERA
# ============================================================

else:

    uploaded_file = st.camera_input(
        "📸 Take a photo of the crop"
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="🌱 Crop Image",
        use_container_width=True
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze Crop",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "🔬 Analyzing crop..."
            ):

                inputs = processor(
                    images=image,
                    return_tensors="pt"
                )

                with torch.no_grad():

                    outputs = model(
                        **inputs
                    )

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=-1
                )

                predicted_class = torch.argmax(
                    probabilities,
                    dim=-1
                ).item()

                label = model.config.id2label[
                    predicted_class
                ]

                confidence = probabilities[
                    0,
                    predicted_class
                ].item()


            # =================================================
            # RESULT
            # =================================================

            st.divider()

            st.subheader(
                "🌱 Analysis Result"
            )

            st.write(
                f"**Detected Class:** {label}"
            )


            # =================================================
            # HEALTH STATUS
            # =================================================

            if "healthy" in label.lower():

                if language == "Tamil":

                    st.success(
                        "✅ செடி ஆரோக்கியமாக இருப்பது போல் தெரிகிறது."
                    )

                    st.write(
                        "🌱 தொடர்ந்து செடியை கண்காணிக்கவும்."
                    )

                else:

                    st.success(
                        "✅ The plant appears healthy."
                    )

                    st.write(
                        "🌱 Continue regular monitoring."
                    )

            else:

                if language == "Tamil":

                    st.warning(
                        "⚠️ செடியில் நோய் அறிகுறிகள் இருக்கலாம்."
                    )

                    st.write(
                        "📋 வேளாண்மை நிபுணரிடம் பரிசோதனை "
                        "செய்து உறுதி செய்யவும்."
                    )

                else:

                    st.warning(
                        "⚠️ Possible disease symptoms detected."
                    )

                    st.write(
                        "📋 Please consult an agricultural "
                        "expert for confirmation."
                    )


            # =================================================
            # CONFIDENCE
            # =================================================

            st.metric(
                "Confidence / நம்பகத்தன்மை",
                f"{confidence * 100:.2f}%"
            )


            # =================================================
            # DISCLAIMER
            # =================================================

            st.info(
                "📋 This is a preliminary AI screening only. "
                "It is not a guaranteed diagnosis."
            )


        except Exception as e:

            st.error(
                "❌ Analysis failed."
            )

            st.exception(e)


else:

    st.info(
        "📷 Please upload or capture a crop image to begin."
    )


# ============================================================
# BENEFITS
# ============================================================

st.divider()

st.subheader("🌾 Key Benefits")

col1, col2 = st.columns(2)

with col1:

    st.write("📷 **Camera Support**")
    st.write(
        "Capture crop images directly from your phone."
    )

    st.write("
