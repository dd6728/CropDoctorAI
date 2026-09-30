import datetime
import io
import json
import os

import requests
import streamlit as st
import torch
from PIL import Image, ImageStat
from transformers import AutoImageProcessor, AutoModelForImageClassification


# ================================================================
# PAGE
# ================================================================
st.set_page_config(
    page_title="CropDoctor AI",
    page_icon="🌱",
    layout="wide",
)

# CPU-friendly settings for Streamlit Cloud
try:
    torch.set_num_threads(1)
except Exception:
    pass


# ================================================================
# STYLE
# ================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
    --ink:#1d1d1f;
    --mute:#6e6e73;
    --bg:#fbfbfd;
    --card:#ffffff;
    --line:#e8e8ed;
    --green:#0a7d3b;
    --lime:#c8f169;
    --red:#d92d20;
}

html,body,[class*="css"],.stApp{
    font-family:-apple-system,'SF Pro Display','Inter',sans-serif;
    color:var(--ink);
}

.stApp{background:var(--bg);}
#MainMenu,footer,header{visibility:hidden;}

.block-container{
    max-width:1080px;
    padding-top:0;
    padding-bottom:4rem;
}

.hero{
    text-align:center;
    padding:5.5rem 1rem 3rem;
}

.hero h1{
    font-size:clamp(2.6rem,7vw,5rem);
    font-weight:800;
    letter-spacing:-.045em;
    line-height:1.02;
    margin:0;
}

.hero p{
    font-size:1.35rem;
    color:var(--mute);
    max-width:640px;
    margin:1.2rem auto 0;
    line-height:1.4;
}

.pill{
    display:inline-block;
    background:#111;
    color:var(--lime);
    border-radius:99px;
    padding:.35rem .9rem;
    font-size:.8rem;
    font-weight:600;
    margin-bottom:1.4rem;
}

.card{
    background:var(--card);
    border:1px solid var(--line);
    border-radius:24px;
    padding:1.6rem 1.8rem;
    margin-bottom:1rem;
}

.card h3{
    margin:0 0 .6rem;
    font-size:1.25rem;
    font-weight:700;
    letter-spacing:-.02em;
}

.card li,.card p{
    color:#3a3a3c;
    font-size:1rem;
    line-height:1.6;
}

.kpi{
    background:#111;
    color:#fff;
    border-radius:24px;
    padding:1.5rem;
    height:100%;
}

.kpi.light{
    background:var(--card);
    color:var(--ink);
    border:1px solid var(--line);
}

.kpi small{
    color:#a1a1a6;
    font-size:.85rem;
    font-weight:500;
}

.kpi.light small{color:var(--mute);}

.kpi b{
    display:block;
    font-size:1.45rem;
    font-weight:700;
    letter-spacing:-.03em;
    margin-top:.3rem;
    line-height:1.15;
}

.kpi b.big{
    font-size:2.8rem;
    color:var(--lime);
}

.kpi.light b.big{color:var(--green);}

.bar{
    height:8px;
    border-radius:9px;
    background:#efeff3;
    margin:.35rem 0 1rem;
    overflow:hidden;
}

.bar i{
    display:block;
    height:100%;
    background:linear-gradient(90deg,#0a7d3b,#c8f169);
    border-radius:9px;
}

.section{
    font-size:2rem;
    font-weight:800;
    letter-spacing:-.035em;
    margin:3rem 0 1rem;
}

.stButton>button{
    background:#111;
    color:#fff;
    border:0;
    border-radius:99px;
    padding:.8rem 2.2rem;
    font-weight:600;
    font-size:1rem;
}

.stButton>button:hover{
    background:var(--green);
    color:#fff;
}

[data-testid="stFileUploader"] section{
    border:2px dashed #c7c7cc;
    border-radius:24px;
    background:#fff;
    padding:2rem;
}

.stTabs [data-baseweb="tab"]{
    font-weight:600;
    font-size:1rem;
}

.foot{
    background:#111;
    color:#f5f5f7;
    border-radius:32px;
    padding:3rem 2.5rem;
    margin-top:4rem;
}

.foot h2{
    font-size:2.2rem;
    font-weight:800;
    letter-spacing:-.03em;
    margin:0 0 .3rem;
    color:#fff;
}

.foot h4{
    color:var(--lime);
    margin:1.6rem 0 .3rem;
    font-size:1rem;
    font-weight:600;
}

.foot p{
    color:#d2d2d7;
    line-height:1.6;
    margin:0;
}

.team span{
    display:inline-block;
    border:1px solid #3a3a3c;
    border-radius:99px;
    padding:.4rem 1rem;
    margin:.25rem .3rem .25rem 0;
    font-size:.95rem;
}

.small-note{
    color:var(--mute);
    font-size:.88rem;
    line-height:1.5;
}

.good{
    color:var(--green);
    font-weight:700;
}
</style>
""",
    unsafe_allow_html=True,
)


# ================================================================
# TEXT
# ================================================================
T = {
    "English": dict(
        badge="Code2Communities Hackathon",
        h1="Know your crop's health in seconds.",
        sub=(
            "Upload a leaf photo. CropDoctor AI screens the image, "
            "identifies a disease class, explains what it sees, and tells you what to do next."
        ),
        upload="Drop a clear, close-up photo of one leaf",
        analyze="Analyze crop",
        empty="Upload a photo to start. Use daylight and keep one leaf in focus.",
        diag="AI Screening",
        conf="Confidence",
        status="Health status",
        sev="Severity",
        healthy="Healthy",
        diseased="Needs attention",
        top="Other possibilities",
        quality="Photo quality",
        tabs=[
            "Overview",
            "Symptoms & causes",
            "Treatment",
            "Prevention",
        ],
        low=(
            "Low confidence. Retake the photo in daylight, closer to one leaf, "
            "then try again."
        ),
        disc=(
            "Preliminary screening only. Confirm with a local agriculture officer "
            "before spraying or making costly decisions."
        ),
        key="Gemini API key (optional)",
        running="Analyzing your crop...",
        model="Local disease model",
        gemini="Gemini Vision cross-check",
        coverage="Model coverage",
        supported=(
            "This local model contains 38 PlantVillage classes covering crops such as "
            "tomato, maize, grape, pepper, potato, apple, peach, orange, strawberry, "
            "cherry, squash, soybean, raspberry and blueberry."
        ),
        weather="Weather risk",
        ask="Ask AI",
    ),
    "Tamil": dict(
        badge="Code2Communities ஹேக்கத்தான்",
        h1="உங்கள் பயிரின் ஆரோக்கியத்தை உடனே அறியுங்கள்.",
        sub=(
            "இலையின் படத்தைப் பதிவேற்றுங்கள். CropDoctor AI படத்தை ஆய்வு செய்து, "
            "நோய் வகையை கண்டறிந்து, அடுத்து என்ன செய்யலாம் என்று சொல்லும்."
        ),
        upload="ஒரு இலையின் தெளிவான, அருகிலான படத்தைப் பதிவேற்றவும்",
        analyze="பயிரை ஆய்வு செய்",
        empty="தொடங்க ஒரு படத்தைப் பதிவேற்றவும். பகல் வெளிச்சத்தில் ஒரு இலையை தெளிவாக எடுக்கவும்.",
        diag="AI ஆய்வு",
        conf="நம்பகத்தன்மை",
        status="ஆரோக்கிய நிலை",
        sev="தீவிரம்",
        healthy="ஆரோக்கியம்",
        diseased="கவனம் தேவை",
        top="பிற வாய்ப்புகள்",
        quality="படத்தின் தரம்",
        tabs=["சுருக்கம்", "அறிகுறிகள் & காரணங்கள்", "சிகிச்சை", "தடுப்பு"],
        low=(
            "நம்பகத்தன்மை குறைவு. பகல் வெளிச்சத்தில், ஒரு இலையை அருகில் படம் "
            "எடுத்து மீண்டும் முயலவும்."
        ),
        disc=(
            "இது ஆரம்பநிலை ஆய்வு மட்டுமே. மருந்து தெளிக்கும் முன் வேளாண் "
            "அலுவலரிடம் உறுதி செய்யவும்."
        ),
        key="Gemini API key (விருப்பம்)",
        running="பயிரை ஆய்வு செய்கிறது...",
        model="உள்ளூர் நோய் கண்டறிதல் மாடல்",
        gemini="Gemini Vision சரிபார்ப்பு",
        coverage="மாடல் உள்ளடக்கம்",
        supported=(
            "இந்த உள்ளூர் மாடல் 38 PlantVillage வகைகளை கொண்டுள்ளது. தக்காளி, "
            "மக்காச்சோளம், திராட்சை, மிளகாய், உருளைக்கிழங்கு, ஆப்பிள், பீச், "
            "ஆரஞ்சு, ஸ்ட்ராபெர்ரி, செர்ரி, ஸ்குவாஷ், சோயாபீன் போன்ற பயிர்கள் இதில் உள்ளன."
        ),
        weather="வானிலை அபாயம்",
        ask="AI-யிடம் கேள்",
    ),
}


# ================================================================
# MODEL
# ================================================================
# This model has 38 disease/healthy classes and uses SafeTensors.
# It is much better suited to this app than the previous 2-class model.
MODEL_ID = "Kathir56/plant-disease-tamilnadu"


@st.cache_resource(show_spinner=False)
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


# ================================================================
# HELPERS
# ================================================================
def humanize_label(label):
    """Convert model labels such as Tomato___Late_blight into readable text."""
    label = str(label).replace("___", " — ").replace("__", " — ")
    label = label.replace("_", " ")
    label = label.replace("Haunglongbing", "Huanglongbing")
    label = label.replace("Citrus greening", "Citrus greening")
    label = label.replace("  ", " ")
    return label.strip()


def split_label(label):
    """Return crop and disease from PlantVillage-style labels."""
    raw = str(label)

    if "___" in raw:
        crop, disease = raw.split("___", 1)
    elif "__" in raw:
        crop, disease = raw.split("__", 1)
    else:
        crop, disease = "Plant", raw

    crop = crop.replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()

    crop = crop.replace("(", " (").replace("  ", " ")
    disease = disease.replace("Haunglongbing", "Huanglongbing")

    return crop, disease


def is_healthy_label(label):
    return "healthy" in str(label).lower()


def severity(conf, is_healthy):
    if is_healthy:
        return "None", 5
    if conf >= 0.90:
        return "High", 85
    if conf >= 0.70:
        return "Moderate", 60
    return "Uncertain", 35


def photo_quality(img):
    """Simple brightness + resolution check."""
    gray = img.convert("L")
    brightness = ImageStat.Stat(gray).mean[0]
    w, h = img.size

    issues = []

    if brightness < 60:
        issues.append("too dark")
    if brightness > 210:
        issues.append("overexposed")
    if min(w, h) < 224:
        issues.append("low resolution")

    if not issues:
        return "Good", 90

    return "Needs retake: " + ", ".join(issues), 45


def predict_disease(image, processor, model):
    """Run the local 38-class disease classifier."""
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    k = min(3, probabilities.shape[-1])
    top_probs, top_indices = torch.topk(probabilities, k=k)

    results = []

    for probability, index in zip(top_probs, top_indices):
        idx = int(index.item())
        score = float(probability.item())
        label = model.config.id2label.get(idx, str(idx))
        results.append((label, score))

    return results


def get_gemini_client(api_key):
    from google import genai
    return genai.Client(api_key=api_key)


def gemini_report(image, classifier_results, language, api_key):
    """
    Gemini sees the actual image and the local model's top predictions.
    Returns a JSON report or None.
    """
    if not api_key:
        return None

    try:
        client = get_gemini_client(api_key)

        top_text = "\n".join(
            f"- {humanize_label(label)}: {score * 100:.1f}%"
            for label, score in classifier_results
        )

        prompt = f"""
You are an agricultural plant-health assistant.

Analyze the attached leaf image yourself. A local computer-vision model produced
these possible classes:

{top_text}

Your job is to CROSS-CHECK the image against those predictions.

Important:
1. Do not blindly agree with the local model.
2. If the image does not clearly support the prediction, say that confidence is limited.
3. Do not invent a crop or disease that cannot reasonably be supported by the image.
4. If the image is not a plant leaf, say that clearly.
5. Give preliminary agricultural guidance only.
6. Do not provide exact pesticide doses.
7. Write in {language}.
8. Keep the wording simple enough for a farmer.

Return ONLY valid JSON with exactly these keys:

overview: string
gemini_assessment: string
visible_symptoms: list of strings
likely_causes: list of strings
spread_risk: string
treatment_steps: list of strings
prevention: list of strings
when_to_call_expert: string
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image],
            config={"response_mime_type": "application/json"},
        )

        data = json.loads(response.text)

        # Basic validation so a malformed Gemini response cannot break the app.
        required = [
            "overview",
            "gemini_assessment",
            "visible_symptoms",
            "likely_causes",
            "spread_risk",
            "treatment_steps",
            "prevention",
            "when_to_call_expert",
        ]

        if not all(key in data for key in required):
            return None

        return data

    except Exception as exc:
        st.session_state["gem_err"] = str(exc)
        return None


FALLBACK = {
    "overview": (
        "The local image classifier produced the result shown above. "
        "Add a Gemini API key for photo-specific symptom analysis."
    ),
    "gemini_assessment": (
        "Gemini analysis was not available. Use the local classifier as a preliminary screening only."
    ),
    "visible_symptoms": [
        "Check for spots, yellowing, curling, mold, holes or unusual leaf colour."
    ],
    "likely_causes": [
        "Possible fungal, bacterial, viral, pest or nutrient-related stress."
    ],
    "spread_risk": "Unknown. Inspect neighbouring plants.",
    "treatment_steps": [
        "Remove severely affected leaves when appropriate.",
        "Avoid unnecessary overhead watering.",
        "Ask a local agriculture officer to confirm the disease before using chemicals.",
    ],
    "prevention": [
        "Keep good airflow between plants.",
        "Use clean tools.",
        "Monitor nearby plants regularly.",
    ],
    "when_to_call_expert": (
        "Contact an agriculture expert if symptoms spread quickly or the diagnosis is uncertain."
    ),
}


def weather_risk(city):
    """Free Open-Meteo API."""
    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
        geo_response.raise_for_status()
        geo = geo_response.json()

        if not geo.get("results"):
            return None

        place = geo["results"][0]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m",
                "daily": "precipitation_sum",
                "forecast_days": 5,
                "timezone": "auto",
            },
            timeout=8,
        )
        weather_response.raise_for_status()
        weather = weather_response.json()

        current = weather.get("current", {})
        daily = weather.get("daily", {})

        temp = float(current.get("temperature_2m", 0))
        humidity = float(current.get("relative_humidity_2m", 0))
        rain_values = daily.get("precipitation_sum", []) or []
        rain = sum(float(x or 0) for x in rain_values)

        score = (
            int(humidity > 80)
            + int(rain > 20)
            + int(22 <= temp <= 32)
        )

        levels = ["Low", "Moderate", "High", "Very high"]

        return {
            "place": place.get("name", city),
            "temp": temp,
            "hum": humidity,
            "rain": rain,
            "level": levels[min(score, 3)],
            "score": score,
        }

    except Exception:
        return None


def speak(text, language):
    """Text-to-speech."""
    try:
        from gtts import gTTS

        buffer = io.BytesIO()
        gTTS(
            text=text,
            lang="ta" if language == "Tamil" else "en",
        ).write_to_fp(buffer)

        return buffer.getvalue()

    except Exception:
        return None


def report_text(crop, disease, conf, severity_text, report):
    lines = [
        f"CropDoctor AI report - {datetime.date.today()}",
        f"Crop: {crop}",
        f"AI screening: {disease}",
        f"Classifier confidence: {conf * 100:.1f}%",
        f"Severity: {severity_text}",
        "",
    ]

    for key, value in report.items():
        lines.append(key.replace("_", " ").title() + ":")

        if isinstance(value, list):
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"  {value}")

    lines.extend(
        [
            "",
            "Preliminary screening only.",
            "Confirm important treatment decisions with a qualified agriculture officer.",
            "Kisan Call Centre: 1800-180-1551 (toll-free)",
        ]
    )

    return "\n".join(lines)


def ask(question, context, language, api_key):
    """Gemini follow-up Q&A."""
    try:
        client = get_gemini_client(api_key)

        prompt = f"""
You are a simple-language agricultural assistant.

Answer the farmer's question in {language}.

Context:
{context}

Rules:
- Keep the answer practical and concise.
- Do not give exact pesticide doses.
- Do not claim certainty when the diagnosis is uncertain.
- Recommend local agricultural expert confirmation for important treatment decisions.

Farmer's question:
{question}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text

    except Exception as exc:
        return f"Could not answer: {exc}"


def bullets(value):
    if isinstance(value, list):
        if not value:
            return "<p>No information available.</p>"

        return (
            "<ul>"
            + "".join(f"<li>{item}</li>" for item in value)
            + "</ul>"
        )

    return f"<p>{value}</p>"


def card(title, body):
    st.markdown(
        f'<div class="card"><h3>{title}</h3>{bullets(body)}</div>',
        unsafe_allow_html=True,
    )


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    language = st.radio(
        "Language / மொழி",
        ["English", "Tamil"],
    )

    t = T[language]

    try:
        default_key = os.getenv("GEMINI_API_KEY", "")
        if not default_key:
            default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        default_key = ""

    api_key = st.text_input(
        t["key"],
        type="password",
        value=default_key,
    )

    st.caption("Kisan Call Centre: 1800-180-1551 (toll-free)")

    st.markdown("---")

    st.markdown(
        f"""
        **{t["model"]}**

        MobileNetV2 • 38 classes

        **{t["coverage"]}**

        {t["supported"]}
        """,
    )


# ================================================================
# HERO
# ================================================================
st.markdown(
    f"""
    <div class="hero">
        <span class="pill">{t["badge"]}</span>
        <h1>{t["h1"]}</h1>
        <p>{t["sub"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# LOAD MODEL
# ================================================================
try:
    with st.spinner("Loading disease model..."):
        processor, model = load_model()
except Exception as exc:
    st.error("The disease model could not be loaded.")
    st.code(str(exc))
    st.info(
        "Make sure transformers, torch, safetensors and Pillow are installed, "
        "then restart the Streamlit app."
  
