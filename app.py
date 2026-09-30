import datetime
import io
import json
import os

import requests

import streamlit as st
import torch
from PIL import Image, ImageStat
from transformers import ViTImageProcessor, ViTForImageClassification

st.set_page_config(page_title="CropDoctor AI", page_icon="🌱", layout="wide")

# ------------------------------------------------------------------
# STYLE  (Apple-like: big type, generous space, soft surfaces)
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--ink:#1d1d1f;--mute:#6e6e73;--bg:#fbfbfd;--card:#ffffff;--line:#e8e8ed;--green:#0a7d3b;--lime:#c8f169;}
html,body,[class*="css"],.stApp{font-family:-apple-system,'SF Pro Display','Inter',sans-serif;color:var(--ink);}
.stApp{background:var(--bg);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{max-width:1080px;padding-top:0;padding-bottom:4rem;}
.hero{text-align:center;padding:5.5rem 1rem 3rem;}
.hero h1{font-size:clamp(2.6rem,7vw,5rem);font-weight:800;letter-spacing:-.045em;line-height:1.02;margin:0;}
.hero p{font-size:1.35rem;color:var(--mute);max-width:640px;margin:1.2rem auto 0;line-height:1.4;}
.pill{display:inline-block;background:#111;color:var(--lime);border-radius:99px;padding:.35rem .9rem;font-size:.8rem;font-weight:600;margin-bottom:1.4rem;}
.card{background:var(--card);border:1px solid var(--line);border-radius:24px;padding:1.6rem 1.8rem;margin-bottom:1rem;}
.card h3{margin:0 0 .6rem;font-size:1.25rem;font-weight:700;letter-spacing:-.02em;}
.card li,.card p{color:#3a3a3c;font-size:1rem;line-height:1.6;}
.kpi{background:#111;color:#fff;border-radius:24px;padding:1.5rem;height:100%;}
.kpi.light{background:var(--card);color:var(--ink);border:1px solid var(--line);}
.kpi small{color:#a1a1a6;font-size:.85rem;font-weight:500;}
.kpi.light small{color:var(--mute);}
.kpi b{display:block;font-size:1.7rem;font-weight:700;letter-spacing:-.03em;margin-top:.3rem;line-height:1.15;}
.kpi b.big{font-size:2.8rem;color:var(--lime);}
.kpi.light b.big{color:var(--green);}
.bar{height:8px;border-radius:9px;background:#efefF3;margin:.35rem 0 1rem;overflow:hidden;}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#0a7d3b,#c8f169);border-radius:9px;}
.section{font-size:2rem;font-weight:800;letter-spacing:-.035em;margin:3rem 0 1rem;}
.stButton>button{background:#111;color:#fff;border:0;border-radius:99px;padding:.8rem 2.2rem;font-weight:600;font-size:1rem;}
.stButton>button:hover{background:var(--green);color:#fff;}
[data-testid="stFileUploader"] section{border:2px dashed #c7c7cc;border-radius:24px;background:#fff;padding:2rem;}
.stTabs [data-baseweb="tab"]{font-weight:600;font-size:1rem;}
.foot{background:#111;color:#f5f5f7;border-radius:32px;padding:3rem 2.5rem;margin-top:4rem;}
.foot h2{font-size:2.2rem;font-weight:800;letter-spacing:-.03em;margin:0 0 .3rem;color:#fff;}
.foot h4{color:var(--lime);margin:1.6rem 0 .3rem;font-size:1rem;font-weight:600;}
.foot p{color:#d2d2d7;line-height:1.6;margin:0;}
.team span{display:inline-block;border:1px solid #3a3a3c;border-radius:99px;padding:.4rem 1rem;margin:.25rem .3rem .25rem 0;font-size:.95rem;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# TEXT (English / Tamil)
# ------------------------------------------------------------------
T = {
    "English": dict(
        badge="Code2Communities Hackathon", h1="Know your crop's health in seconds.",
        sub="Upload a leaf photo. CropDoctor AI screens it, explains what it sees, and tells you what to do next.",
        upload="Drop a clear, close-up photo of one leaf", analyze="Analyze crop",
        empty="Upload a photo to start. Use daylight and keep the leaf in focus.",
        diag="Diagnosis", conf="Confidence", status="Health status", sev="Severity",
        healthy="Healthy", diseased="Needs attention", top="Other possibilities",
        quality="Photo quality", tabs=["Overview", "Symptoms & causes", "Treatment", "Prevention"],
        low="Low confidence. Retake the photo in daylight, closer to one leaf, then try again.",
        disc="Preliminary screening only. Confirm with a local agriculture officer before spraying or making costly decisions.",
        lang="Language", key="Gemini API key (optional)", running="Analyzing your crop...",
    ),
    "Tamil": dict(
        badge="Code2Communities ஹேக்கத்தான்", h1="உங்கள் பயிரின் ஆரோக்கியத்தை உடனே அறியுங்கள்.",
        sub="இலையின் படத்தைப் பதிவேற்றுங்கள். CropDoctor AI அதை ஆய்வு செய்து, அடுத்து என்ன செய்யலாம் என்று சொல்லும்.",
        upload="ஒரு இலையின் தெளிவான, அருகிலான படத்தைப் பதிவேற்றவும்", analyze="பயிரை ஆய்வு செய்",
        empty="தொடங்க ஒரு படத்தைப் பதிவேற்றவும். பகல் வெளிச்சத்தில் எடுக்கவும்.",
        diag="கண்டறிதல்", conf="நம்பகத்தன்மை", status="ஆரோக்கிய நிலை", sev="தீவிரம்",
        healthy="ஆரோக்கியம்", diseased="கவனம் தேவை", top="பிற வாய்ப்புகள்",
        quality="படத்தின் தரம்", tabs=["சுருக்கம்", "அறிகுறிகள் & காரணங்கள்", "சிகிச்சை", "தடுப்பு"],
        low="நம்பகத்தன்மை குறைவு. பகல் வெளிச்சத்தில், ஒரு இலையை அருகில் படம் எடுத்து மீண்டும் முயலவும்.",
        disc="இது ஆரம்பநிலை ஆய்வு மட்டுமே. மருந்து தெளிக்கும் முன் வேளாண் அலுவலரிடம் உறுதி செய்யவும்.",
        lang="மொழி", key="Gemini API key (விருப்பம்)", running="பயிரை ஆய்வு செய்கிறது...",
    ),
}

# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
MODEL_ID = "ayerr/plant-disease-classification"


@st.cache_resource
def load_model():
    p = ViTImageProcessor.from_pretrained(MODEL_ID)
    m = ViTForImageClassification.from_pretrained(MODEL_ID)
    m.eval()
    return p, m


def clean(label):
    crop, _, dis = label.partition("___")
    if not dis:
        return label.replace("_", " ")
    return crop.replace("_", " ").strip(), dis.replace("_", " ").strip()


def photo_quality(img):
    """Simple brightness + size check so users know if the photo is usable."""
    g = img.convert("L")
    b = ImageStat.Stat(g).mean[0]
    w, h = img.size
    issues = []
    if b < 60: issues.append("too dark")
    if b > 210: issues.append("overexposed")
    if min(w, h) < 224: issues.append("low resolution")
    return ("Good", 90) if not issues else ("Needs retake: " + ", ".join(issues), 45)


def severity(conf, is_healthy):
    if is_healthy: return "None", 5
    if conf >= .9: return "High", 85
    if conf >= .7: return "Moderate", 60
    return "Uncertain", 35


def gemini_report(image, label, conf, language, key):
    """Deep analysis with Gemini Vision. Returns dict or None."""
    try:
        from google import genai
        client = genai.Client(api_key=key)
        prompt = f"""You are an agricultural plant-health assistant helping a farmer.
A classifier suggests: "{label}" ({conf*100:.0f}% confidence). Look at the photo yourself and
give a preliminary assessment. If the photo disagrees with the classifier, say so.
Write everything in {language}, in simple words. Return ONLY JSON with keys:
overview (string), visible_symptoms (list), likely_causes (list), spread_risk (string),
treatment_steps (list, organic options first, then chemical with general guidance, no exact doses),
prevention (list), when_to_call_expert (string)."""
        r = client.models.generate_content(
            model="gemini-2.5-flash", contents=[prompt, image],
            config={"response_mime_type": "application/json"})
        return json.loads(r.text)
    except Exception as e:
        st.session_state["gem_err"] = str(e)
        return None


FALLBACK = {
    "overview": "AI classifier result only. Add a Gemini API key in the sidebar for a detailed, photo-specific report.",
    "visible_symptoms": ["Check for spots, yellowing, curling, mold or holes on the leaf."],
    "likely_causes": ["Fungal, bacterial or viral infection, pests, or nutrient stress."],
    "spread_risk": "Unknown. Inspect neighbouring plants.",
    "treatment_steps": ["Remove and destroy badly affected leaves.", "Avoid overhead watering.", "Ask an agriculture officer for the right product."],
    "prevention": ["Rotate crops.", "Space plants for airflow.", "Use clean tools and certified seed."],
    "when_to_call_expert": "If symptoms spread within 2-3 days.",
}


def weather_risk(city):
    """Free Open-Meteo API (no key). Humidity + rain + warmth favour fungal spread."""
    try:
        g = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": city, "count": 1}, timeout=8).json()["results"][0]
        w = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": g["latitude"], "longitude": g["longitude"],
            "current": "temperature_2m,relative_humidity_2m",
            "daily": "precipitation_sum", "forecast_days": 5}, timeout=8).json()
        temp = w["current"]["temperature_2m"]
        hum = w["current"]["relative_humidity_2m"]
        rain = sum(x or 0 for x in w["daily"]["precipitation_sum"])
        score = int(hum > 80) + int(rain > 20) + int(22 <= temp <= 32)
        return dict(place=g["name"], temp=temp, hum=hum, rain=rain,
                    level=["Low", "Moderate", "High", "Very high"][score], score=score)
    except Exception:
        return None


def speak(text, language):
    """Text-to-speech so farmers who prefer listening can hear the advice."""
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="ta" if language == "Tamil" else "en").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def report_text(crop, disease, conf, sev, rep):
    lines = [f"CropDoctor AI report - {datetime.date.today()}", f"Crop: {crop}",
             f"Diagnosis: {disease} ({conf*100:.0f}% confidence)", f"Severity: {sev}", ""]
    for k, v in rep.items():
        lines.append(k.replace("_", " ").title() + ":")
        lines += [f"  - {x}" for x in v] if isinstance(v, list) else [f"  {v}"]
    lines += ["", "Preliminary screening only. Confirm with an agriculture officer.",
              "Kisan Call Centre: 1800-180-1551 (toll-free)"]
    return "\n".join(lines)


def ask(question, ctx, language, key):
    try:
        from google import genai
        r = genai.Client(api_key=key).models.generate_content(
            model="gemini-2.5-flash",
            contents=f"You are a farm advisor. Context: {ctx}\nAnswer briefly in {language}, simple words, "
                     f"no exact pesticide doses.\nFarmer asks: {question}")
        return r.text
    except Exception as e:
        return f"Could not answer: {e}"


def bullets(v):
    if isinstance(v, list):
        return "<ul>" + "".join(f"<li>{x}</li>" for x in v) + "</ul>"
    return f"<p>{v}</p>"


def card(title, body):
    st.markdown(f'<div class="card"><h3>{title}</h3>{bullets(body)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    language = st.radio("Language / மொழி", ["English", "Tamil"])
    t = T[language]
    try:
        default_key = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
    except Exception:
        default_key = ""
    api_key = st.text_input(t["key"], type="password", value=default_key)
    st.caption("Kisan Call Centre: 1800-180-1551 (toll-free)")

# ------------------------------------------------------------------
# HERO
# ------------------------------------------------------------------
st.markdown(f'<div class="hero"><span class="pill">{t["badge"]}</span>'
            f'<h1>{t["h1"]}</h1><p>{t["sub"]}</p></div>', unsafe_allow_html=True)

processor, model = load_model()
mode = st.radio("Input", ["Upload photo", "Use camera"], horizontal=True, label_visibility="collapsed")
file = (st.file_uploader(t["upload"], type=["jpg", "jpeg", "png"]) if mode == "Upload photo"
        else st.camera_input(t["upload"]))

if not file:
    st.info(t["empty"])
else:
    image = Image.open(file).convert("RGB")
    image.thumbnail((1024, 1024))
    left, right = st.columns([1, 1], gap="large")
    left.image(image, use_container_width=True)
    with right:
        q_text, q_score = photo_quality(image)
        st.markdown(f'<div class="card"><h3>{t["quality"]}</h3><p>{q_text}</p>'
                    f'<div class="bar"><i style="width:{q_score}%"></i></div></div>', unsafe_allow_html=True)
        go = st.button(t["analyze"], type="primary")

    if go:
        with st.spinner(t["running"]):
            with torch.no_grad():
                out = model(**processor(images=image, return_tensors="pt"))
            probs = torch.softmax(out.logits, dim=-1)[0]
            tp, ti = torch.topk(probs, k=min(3, len(probs)))
            top = [(model.config.id2label[i.item()], p.item()) for p, i in zip(tp, ti)]
            report = gemini_report(image, top[0][0], top[0][1], language, api_key) if api_key else None
        st.session_state["res"] = dict(top=top, report=report or FALLBACK, gem=bool(report))
        c0 = clean(top[0][0])
        st.session_state.setdefault("hist", []).append({
            "Time": datetime.datetime.now().strftime("%H:%M"),
            "Result": " - ".join(c0) if isinstance(c0, tuple) else c0,
            "Confidence %": round(top[0][1] * 100)})

    res = st.session_state.get("res")
    if res:
        label, conf = res["top"][0]
        parsed = clean(label)
        crop, disease = parsed if isinstance(parsed, tuple) else ("Plant", parsed)
        is_healthy = "healthy" in disease.lower()
        sev, sev_pct = severity(conf, is_healthy)
        rep = res["report"]

        st.markdown('<div class="section">' + t["diag"] + '</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="kpi"><small>{crop}</small><b>{disease}</b></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi light"><small>{t["conf"]}</small><b class="big">{conf*100:.0f}%</b></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi light"><small>{t["status"]}</small><b>{t["healthy"] if is_healthy else t["diseased"]}</b></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="kpi light"><small>{t["sev"]}</small><b>{sev}</b>'
                    f'<div class="bar"><i style="width:{sev_pct}%"></i></div></div>', unsafe_allow_html=True)

        if conf < 0.6:
            st.warning(t["low"])

        extra = ["Weather risk", "Ask AI"] if language == "English" else ["வானிலை அபாயம்", "AI-யிடம் கேள்"]
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(t["tabs"] + extra)
        with tab1:
            card(t["tabs"][0], rep.get("overview", ""))
            card("Spread risk" if language == "English" else "பரவும் அபாயம்", rep.get("spread_risk", ""))
            st.markdown(f'<div class="card"><h3>{t["top"]}</h3>' + "".join(
                f'<div><b>{" – ".join(clean(l)) if isinstance(clean(l), tuple) else clean(l)}</b> · {p*100:.1f}%'
                f'<div class="bar"><i style="width:{p*100:.1f}%"></i></div></div>' for l, p in res["top"]
            ) + "</div>", unsafe_allow_html=True)
        with tab2:
            card("Visible symptoms" if language == "English" else "தெரியும் அறிகுறிகள்", rep.get("visible_symptoms", []))
            card("Likely causes" if language == "English" else "சாத்தியமான காரணங்கள்", rep.get("likely_causes", []))
        with tab3:
            card(t["tabs"][2], rep.get("treatment_steps", []))
            card("When to call an expert" if language == "English" else "நிபுணரை எப்போது அழைப்பது", rep.get("when_to_call_expert", ""))
        with tab4:
            card(t["tabs"][3], rep.get("prevention", []))

        with tab5:
            en = language == "English"
            city = st.text_input("Your village or town" if en else "உங்கள் ஊர்", placeholder="e.g. Madurai")
            if city:
                wx = weather_risk(city)
                if wx:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Temperature" if en else "வெப்பநிலை", f"{wx['temp']}°C")
                    m2.metric("Humidity" if en else "ஈரப்பதம்", f"{wx['hum']}%")
                    m3.metric("5-day rain" if en else "5 நாள் மழை", f"{wx['rain']:.0f} mm")
                    m4.metric("Spread risk" if en else "பரவும் அபாயம்", wx["level"])
                    if wx["score"] >= 2 and not is_healthy:
                        st.warning("Weather favours fast spread. Act within 1-2 days and avoid evening irrigation."
                                   if en else "வானிலை நோய் பரவ உகந்தது. 1-2 நாட்களில் நடவடிக்கை எடுக்கவும்.")
                    else:
                        st.success("Weather risk is manageable. Keep monitoring." if en else "வானிலை அபாயம் கட்டுக்குள் உள்ளது.")
                else:
                    st.warning("Could not fetch weather. Check the place name or connection.")
        with tab6:
            if not api_key:
                st.info("Add a Gemini API key in the sidebar to ask follow-up questions.")
            else:
                q = st.text_input("Ask anything about this diagnosis" if language == "English" else "இந்த நோய் பற்றி கேளுங்கள்")
                if q:
                    with st.spinner("..."):
                        st.markdown(ask(q, f"{crop} {disease}, {conf*100:.0f}% confidence. Report: {json.dumps(rep)[:1500]}", language, api_key))

        a, b = st.columns(2)
        if a.button("🔊 Listen to advice" if language == "English" else "🔊 ஆலோசனையைக் கேள்"):
            steps = rep.get("treatment_steps", [])
            audio = speak(str(rep.get("overview", "")) + " " + " ".join(map(str, steps)), language)
            if audio:
                st.audio(audio, format="audio/mp3")
            else:
                st.warning("Audio unavailable. Check your internet connection.")
        b.download_button("⬇ Download report" if language == "English" else "⬇ அறிக்கையைப் பதிவிறக்கு",
                          report_text(crop, disease, conf, sev, rep), file_name="cropdoctor_report.txt")

        if not res["gem"] and api_key:
            st.error("Gemini report failed: " + st.session_state.get("gem_err", "unknown error"))
        st.caption(t["disc"])

# ------------------------------------------------------------------
# HACKATHON FOOTER
# ------------------------------------------------------------------
if st.session_state.get("hist"):
    import pandas as pd
    df = pd.DataFrame(st.session_state["hist"])
    st.markdown('<div class="section">Scan history</div>', unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    h1.dataframe(df, use_container_width=True, hide_index=True)
    h2.bar_chart(df["Result"].value_counts())

st.markdown("""
<div class="foot">
  <span class="pill">Code2Communities Hackathon</span>
  <h2>CropDoctor AI</h2>
  <p>Team Snaptech</p>
  <div class="team" style="margin-top:.8rem">
    <span>Deva Dharshini G</span><span>Ameera Banu R</span><span>Harshini D</span><span>Deepika S</span>
  </div>
  <h4>Problem</h4>
  <p>Farmers may not have immediate access to agricultural experts when they notice unusual symptoms on their crops.</p>
  <h4>Solution</h4>
  <p>Farmers upload a crop photo. Gemini Vision analyzes visible symptoms, provides a preliminary assessment, and suggests appropriate next steps in the farmer's preferred language.</p>
  <h4>What makes it different</h4>
  <p>Two AI models cross-check each other (ViT classifier + Gemini Vision). Local-language text and voice. Weather-aware spread risk. Works on a phone camera. Follow-up Q&amp;A and a downloadable report.</p>
  <h4>Impact</h4>
  <p>Faster first response for small farmers, fewer wrong or excessive pesticide sprays, and earlier expert escalation when symptoms are serious.</p>
  <h4>Tech stack</h4>
  <p>Streamlit, PyTorch, Hugging Face Transformers (ViT), Gemini Vision, Open-Meteo, gTTS.</p>
  <h4>Roadmap</h4>
  <p>More Indian languages, offline mobile app, WhatsApp bot, district-level outbreak map, and integration with Kisan Call Centre (1800-180-1551).</p>
</div>
""", unsafe_allow_html=True)
