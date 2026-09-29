import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Crop Doctor AI",
    page_icon="🌱",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🌱 Crop Doctor AI")

st.write(
    "Upload or capture a crop image for preliminary "
    "plant-health screening."
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

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


# --------------------------------------------------
# LOAD AI MODEL
# --------------------------------------------------

try:

    processor, model = load_model()

    st.success("✅ AI model loaded successfully!")

except Exception as e:

    st.error("❌ AI model loading failed.")

    st.exception(e)

    st.stop()


# --------------------------------------------------
# LANGUAGE SELECTION
# --------------------------------------------------

language = st.radio(
    "Choose Language / மொழியை தேர்வு செய்யவும்",
    ["English", "Tamil"],
    horizontal=True
)


# --------------------------------------------------
# IMAGE SOURCE
# --------------------------------------------------

input_method = st.radio(
    "📷 Choose Image Source / படத்தை தேர்வு செய்யவும்",
    ["Upload Image", "Use Camera"],
    horizontal=True
)


uploaded_file = None


# --------------------------------------------------
# UPLOAD IMAGE
# --------------------------------------------------

if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "📁 Upload Crop Image / பயிர் படத்தை Upload செய்யவும்",
        type=["jpg", "jpeg", "png"]
    )


# --------------------------------------------------
# CAMERA
# --------------------------------------------------

else:

    uploaded_file = st.camera_input(
        "📸 Take a photo of the crop / பயிரின் புகைப்படத்தை எடுக்கவும்"
    )


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="🌱 Crop Image",
            use_container_width=True
        )


        # --------------------------------------------------
        # ANALYZE BUTTON
        # --------------------------------------------------

        if st.button(
            "🔍 Analyze Crop",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "🔬 Analyzing crop... / பயிரை ஆய்வு செய்கிறது..."
                ):

                    # Process image
                    inputs = processor(
                        images=image,
                        return_tensors="pt"
                    )


                    # AI prediction
                    with torch.no_grad():

                        outputs = model(
                            **inputs
                        )


                    # Convert logits to probabilities
                    probabilities = torch.softmax(
                        outputs.logits,
                        dim=-1
                    )


                    # Get predicted class
                    predicted_class = torch.argmax(
                        probabilities,
                        dim=-1
                    ).item()


                    # Get label
                    label = model.config.id2label[
                        predicted_class
                    ]


                    # Get confidence
                    confidence = probabilities[
                        0,
                        predicted_class
                    ].item()


                # --------------------------------------------------
                # RESULT
                # --------------------------------------------------

                st.divider()

                st.subheader(
                    "🌱 Analysis Result / ஆய்வு முடிவு"
                )


                # Display detected disease
                st.write(
                    f"**Detected Class:** {label}"
                )


                # --------------------------------------------------
                # HEALTHY / DISEASE CHECK
                # --------------------------------------------------

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


                # --------------------------------------------------
                # CONFIDENCE
                # --------------------------------------------------

                st.metric(
                    "Confidence / நம்பகத்தன்மை",
                    f"{confidence * 100:.2f}%"
                )


                # --------------------------------------------------
                # DISCLAIMER
                # --------------------------------------------------

                st.info(
                    "📋 This is a preliminary AI screening only. "
                    "It is not a guaranteed diagnosis."
                )


            except Exception as e:

                st.error(
                    "❌ Analysis failed."
                )

                st.exception(e)


    except Exception as e:

        st.error(
            "❌ Unable to read the image."
        )

        st.exception(e)


# --------------------------------------------------
# NO IMAGE
# --------------------------------------------------

else:

    st.info(
        "📷 Please upload or capture a crop image to begin."
    )

What this version adds

- 📷 Camera capture
- 📁 Image upload
- 🌱 Crop preview
- 🔍 Analyze Crop button
- 🇬🇧 English / 🇮🇳 Tamil
- 🤖 ViT disease classification
- 📊 Confidence percentage
- ✅ Healthy detection
- ⚠️ Disease detection
- 📋 Preliminary-diagnosis disclaimer

Important: I also changed the healthy check from exact matching:

label.lower() == "healthy"

to:

"healthy" in label.lower()

This handles labels such as "Tomato___healthy" as well.
