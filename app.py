import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

st.set_page_config(
page_title="Crop Doctor AI",
page_icon="🌱"
)

st.title("🌱 Crop Doctor AI")
st.write(
"Upload a crop image for preliminary plant-health screening."
)

MODEL_ID = "ayerr/plant-disease-classification"
MODEL_SUBFOLDER = "ayerr/plant-disease-classification"

@st.cache_resource
def load_model():

processor = ViTImageProcessor.from_pretrained(  
    MODEL_ID,  
    subfolder=MODEL_SUBFOLDER  
)  

model = ViTForImageClassification.from_pretrained(  
    MODEL_ID,  
    subfolder=MODEL_SUBFOLDER  
)  

model.eval()  

return processor, model

try:
processor, model = load_model()
st.success("✅ AI model loaded successfully!")

except Exception as e:
st.error("❌ AI model loading failed.")
st.exception(e)
st.stop()

language = st.radio(
"Choose Language / மொழியை தேர்வு செய்யவும்",
["English", "Tamil"]
)

uploaded_file = st.file_uploader(
"📷 Upload Crop Image / பயிர் படத்தை Upload செய்யவும்",
type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

image = Image.open(uploaded_file).convert("RGB")  

st.image(  
    image,  
    caption="Uploaded Crop Image",  
    use_container_width=True  
)  

if st.button("🔍 Analyze Crop", type="primary"):  

    try:  

        with st.spinner("🔬 Analyzing crop..."):  

            inputs = processor(  
                images=image,  
                return_tensors="pt"  
            )  

            with torch.no_grad():  
                outputs = model(**inputs)  

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


        st.divider()  

        st.subheader("🌱 Analysis Result")  

        if label.lower() == "healthy":  

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

        st.metric(  
            "Confidence / நம்பகத்தன்மை",  
            f"{confidence * 100:.2f}%"  
        )  

        st.info(  
            "📋 This is a preliminary AI screening only. "  
            "It is not a guaranteed diagnosis."  
        )  

    except Exception as e:  

        st.error("❌ Analysis failed.")  
        st.exception(e)

else:

st.info(  
    "📷 Please upload a crop image to begin."  
)
