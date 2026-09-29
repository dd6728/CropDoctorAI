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
# HEADER
# ============================================================

st.title("🌱 Crop Doctor AI")

st.write(
    "AI-powered preliminary crop disease screening "
    "using image analysis."
)


# ============================================================
# PROBLEM
# ============================================================

with st.expander("⚠️ Problem", expanded=True):

    st.write(
        """
        Farmers may find it difficult to identify crop diseases
        at an early stage. Disease symptoms can look similar,
        and getting immediate access to agricultural experts
        may not always be possible.

        Late identification of crop diseases can lead to crop
        damage, reduced yield and economic loss.
        """
    )


# ============================================================
# SOLUTION
# ============================================================

with st.expander("💡 Solution", expanded=True):

    st.write(
        """
        Crop Doctor AI provides a simple AI-based preliminary
        screening system.

        Users can capture a crop image using their mobile camera
        or upload an existing image. The AI model analyzes the
        image and identifies the predicted plant-health class
        along with a confidence score.

        The system provides the result in English or Tamil and
        recommends consulting an agricultural expert for
        confirmation.
        """
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.subheader("⚙️ How It Works")

st.write(
    """
    📷 **1. Capture / Upload**
    
    Take a crop photo using the camera or upload an existing image.

    ↓

    🤖 **2. AI Analysis**
    
    The Vision Transformer (ViT) model analyzes the crop image.

    ↓

    🔎 **3. Disease Screening**
    
    The model predicts the most likely plant-health class.

    ↓

    📊 **4. Confidence**
    
    The system displays the prediction confidence.

    ↓

    👨‍🌾 **5. Expert Confirmation**
    
    The result is only a preliminary screening. Consult an
    agricultural expert before taking treatment decisions.
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
    # ANALYZE
    # ========================================================

    if st.button(
        "🔍 Analyze Crop",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "🔬 AI is analyzing the crop..."
            ):

                # Image preprocessing
                inputs = processor(
                    images=image,
                    return_tensors="pt"
                )


                # AI prediction
                with torch.no_grad():

                    outputs = model(
                        **inputs
                    )


                # Probabilities
                probabilities = torch.softmax(
                    outputs.logits,
                    dim=-1
                )


                # Predicted class
                predicted_class = torch.argmax(
                    probabilities,
                    dim=-1
                ).item()


                # Label
                label = model.config.id2label[
                    predicted_class
                ]


                # Confidence
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
            # HEALTHY
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


            # =================================================
            # DISEASE
            # =================================================

            else:

                if language == "Tamil":

                    st.warning(
                        "⚠️ செடியில் நோய் அறிகுறிகள் "
                        "இருக்கலாம்."
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
                "📋 This AI result is a preliminary screening "
                "only and is not a guaranteed diagnosis."
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
# PROJECT BENEFITS
# ============================================================

st.divider()

st.subheader("🌾 Key Benefits")

col1, col2 = st.columns(2)

with col1:

    st.write("📷 **Easy Access**")
    st.write(
        "Use a smartphone camera to capture crop images."
    )

    st.write("🤖 **AI-Based Screening**")
    st.write(
        "Automatically analyzes the uploaded crop image."
    )

with col2:

    st.write("🌐 **Tamil Support**")
    st.write(
        "Provides results in English and Tamil."
    )

    st.write("⚡ **Quick Screening**")
    st.write(
        "Provides a preliminary result within seconds."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌱 Crop Doctor AI | AI-assisted preliminary plant-health screening"
)

Project flow

Farmer → 📷 Camera/Upload → 🖼️ Crop Image → 🤖 ViT AI → 🔎 Disease Screening → 📊 Confidence → 👨‍🌾 Expert Confirmation

This makes the app suitable for demonstrating the problem, proposed solution, working process, and actual AI implementation in a hackathon or college project.
