# import streamlit as st
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib.pagesizes import A4
# import tempfile
# import warnings
# warnings.filterwarnings("ignore")

# # -------------------------------
# # PAGE CONFIG
# # -------------------------------
# st.set_page_config(
#     page_title="Crop Recommendation System",
#     page_icon="🌾",
#     layout="centered"
# )

# st.title("🌾 Crop Recommendation System")
# st.caption("XGBoost-based ML model with realistic farming inputs")

# # -------------------------------
# # LOAD MODEL FILES
# # -------------------------------
# @st.cache_resource
# def load_model():
#     with open("xgb_crop_model.pkl", "rb") as f:
#         model = pickle.load(f)
#     with open("label_encoder.pkl", "rb") as f:
#         encoder = pickle.load(f)
#     with open("feature_names.pkl", "rb") as f:
#         features = pickle.load(f)
#     return model, encoder, features

# try:
#     model, encoder, features = load_model()
# except FileNotFoundError:
#     st.error("❌ Model files not found. Upload .pkl files.")
#     st.stop()

# # -------------------------------
# # USER INPUTS (SLIDERS)
# # -------------------------------
# st.subheader("🔢 Enter Soil & Climate Parameters")

# N = st.slider("Nitrogen (N) kg/ha", 0, 140, 50)
# P = st.slider("Phosphorus (P) kg/ha", 0, 140, 40)
# K = st.slider("Potassium (K) kg/ha", 0, 205, 40)
# temperature = st.slider("Temperature (°C)", 5.0, 45.0, 25.0)
# humidity = st.slider("Humidity (%)", 10.0, 100.0, 70.0)
# ph = st.slider("Soil pH", 3.5, 9.5, 6.5)
# rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

# input_dict = {
#     'N': N,
#     'P': P,
#     'K': K,
#     'temperature': temperature,
#     'humidity': humidity,
#     'ph': ph,
#     'rainfall': rainfall
# }

# # -------------------------------
# # PREDICTION
# # -------------------------------
# if st.button("🌱 Recommend Crop"):
#     user_input = np.array([input_dict[f] for f in features]).reshape(1, -1)

#     predicted_class = model.predict(user_input)
#     predicted_proba = model.predict_proba(user_input)

#     crop_name = encoder.inverse_transform(predicted_class)[0]
#     confidence = predicted_proba[0][predicted_class[0]] * 100

#     st.success(f"🌾 Recommended Crop: **{crop_name.upper()}**")
#     st.info(f"📊 Confidence: **{confidence:.2f}%**")

#     # -------------------------------
#     # TOP 3 CROPS
#     # -------------------------------
#     st.subheader("🔝 Top 3 Crop Recommendations")

#     top_3_idx = np.argsort(predicted_proba[0])[-3:][::-1]
#     top_crops = [
#         (encoder.inverse_transform([i])[0], predicted_proba[0][i] * 100)
#         for i in top_3_idx
#     ]

#     for i, (crop, prob) in enumerate(top_crops, 1):
#         st.write(f"{i}. **{crop}** — {prob:.2f}%")

#     # Bar Chart
#     crops, probs = zip(*top_crops)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     sns.barplot(x=list(crops), y=list(probs), palette="crest", ax=ax)
#     ax.set_ylabel("Confidence (%)")
#     ax.set_xlabel("Crop")
#     ax.set_title("Top 3 Crop Recommendations")
#     st.pyplot(fig)

#     # -------------------------------
#     # PDF REPORT GENERATION
#     # -------------------------------
#     def generate_pdf():
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#         doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
#         styles = getSampleStyleSheet()
#         story = []

#         story.append(Paragraph("<b>Crop Recommendation Report</b>", styles['Title']))
#         story.append(Spacer(1, 12))

#         story.append(Paragraph("<b>Input Parameters</b>", styles['Heading2']))
#         for k, v in input_dict.items():
#             story.append(Paragraph(f"{k} : {v}", styles['Normal']))

#         story.append(Spacer(1, 12))
#         story.append(Paragraph("<b>Prediction Result</b>", styles['Heading2']))
#         story.append(Paragraph(f"Recommended Crop: <b>{crop_name}</b>", styles['Normal']))
#         story.append(Paragraph(f"Confidence: {confidence:.2f}%", styles['Normal']))

#         story.append(Spacer(1, 12))
#         story.append(Paragraph("<b>Top 3 Crops</b>", styles['Heading2']))
#         for crop, prob in top_crops:
#             story.append(Paragraph(f"{crop}: {prob:.2f}%", styles['Normal']))

#         doc.build(story)
#         return temp_file.name

#     pdf_path = generate_pdf()

#     with open(pdf_path, "rb") as f:
#         st.download_button(
#             label="📄 Download Prediction Report (PDF)",
#             data=f,
#             file_name="crop_recommendation_report.pdf",
#             mime="application/pdf"
#         )








import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import tempfile
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------
# PAGE CONFIG
# -----------------------------------------------
st.set_page_config(
    page_title="CropSense | Crop Recommendation",
    page_icon="🌾",
    layout="centered"
)

# -----------------------------------------------
# GLOBAL CSS
# -----------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ---- Root Variables ---- */
:root {
    --green-dark:  #1a3a2a;
    --green-mid:   #2d6a4f;
    --green-light: #52b788;
    --cream:       #f5f0e8;
    --amber:       #e9a84c;
    --text-main:   #1a3a2a;
    --text-muted:  #5a7a6a;
    --card-bg:     #ffffff;
    --radius:      16px;
}

/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-main);
}

.stApp {
    background: linear-gradient(160deg, #e8f5ec 0%, #f5f0e8 60%, #d8eed8 100%);
    min-height: 100vh;
}

/* ---- Hero Banner ---- */
.hero-banner {
    background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 60%, #40916c 100%);
    border-radius: var(--radius);
    padding: 52px 40px 44px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(26,58,42,0.25);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}

.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}

.hero-icon {
    font-size: 64px;
    display: block;
    margin-bottom: 12px;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-8px); }
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px;
    line-height: 1.2;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.82);
    font-weight: 300;
    margin: 0 0 20px;
    letter-spacing: 0.3px;
}

.hero-badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}

.badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    color: #ffffff;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.4px;
    backdrop-filter: blur(4px);
}

/* ---- Welcome Card ---- */
.welcome-card {
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 32px 36px;
    margin-bottom: 28px;
    border-left: 5px solid var(--green-light);
    box-shadow: 0 4px 20px rgba(26,58,42,0.08);
}

.welcome-card h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    color: var(--green-dark);
    margin-bottom: 10px;
}

.welcome-card p {
    color: var(--text-muted);
    line-height: 1.7;
    font-size: 0.97rem;
}

/* ---- Feature Pills Row ---- */
.feature-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 32px;
}

.feature-pill {
    flex: 1;
    min-width: 130px;
    background: var(--card-bg);
    border: 1.5px solid #d4e8d8;
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(26,58,42,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
}

.feature-pill:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(26,58,42,0.12);
}

.feature-pill .pill-icon { font-size: 1.7rem; display: block; margin-bottom: 6px; }
.feature-pill .pill-label { font-size: 0.8rem; color: var(--text-muted); font-weight: 500; }

/* ---- Section Headers ---- */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid #d4e8d8;
}

.section-header .sh-icon { font-size: 1.4rem; }

.section-header h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: var(--green-dark);
    margin: 0;
}

/* ---- Slider Group Card ---- */
.slider-card {
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 3px 16px rgba(26,58,42,0.07);
    border-top: 3px solid var(--green-light);
}

.slider-card h4 {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: var(--green-mid);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
    font-size: 0.85rem;
}

/* ---- Sliders ---- */
.stSlider > div > div > div > div {
    background: var(--green-light) !important;
}

.stSlider [data-testid="stThumbValue"] {
    background: var(--green-mid) !important;
    color: white !important;
    border-radius: 6px;
}

/* ---- CTA Button ---- */
.stButton > button {
    background: linear-gradient(135deg, var(--green-mid) 0%, #40916c 100%) !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 14px 40px !important;
    border-radius: 50px !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(45,106,79,0.35) !important;
    transition: all 0.25s ease !important;
    width: 100%;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(45,106,79,0.45) !important;
    background: linear-gradient(135deg, #1e5c3a 0%, var(--green-mid) 100%) !important;
}

/* ---- Result Cards ---- */
.result-main {
    background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 100%);
    border-radius: var(--radius);
    padding: 28px 32px;
    text-align: center;
    color: white;
    margin-bottom: 16px;
    box-shadow: 0 8px 28px rgba(26,58,42,0.3);
}

.result-main .crop-label {
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.75;
    margin-bottom: 6px;
}

.result-main .crop-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 4px;
}

.result-main .confidence-badge {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    display: inline-block;
    padding: 5px 18px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 500;
}

/* Success / Info override */
.stSuccess, .stInfo {
    border-radius: 12px !important;
}

/* ---- Download Button ---- */
.stDownloadButton > button {
    background: var(--amber) !important;
    color: var(--green-dark) !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    border: none !important;
    padding: 12px 30px !important;
    width: 100%;
    box-shadow: 0 4px 14px rgba(233,168,76,0.35) !important;
    transition: all 0.2s !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(233,168,76,0.5) !important;
}

/* ---- Divider ---- */
hr {
    border: none;
    border-top: 1.5px solid #d4e8d8;
    margin: 32px 0;
}

/* ---- Footer ---- */
.footer-card {
    background: var(--green-dark);
    border-radius: var(--radius);
    padding: 36px 32px;
    text-align: center;
    margin-top: 40px;
    color: rgba(255,255,255,0.85);
    box-shadow: 0 -4px 20px rgba(26,58,42,0.15);
}

.footer-card .footer-icon { font-size: 2.4rem; margin-bottom: 10px; display: block; }

.footer-card h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #ffffff;
    margin-bottom: 10px;
}

.footer-card p {
    font-size: 0.92rem;
    line-height: 1.75;
    color: rgba(255,255,255,0.7);
    max-width: 500px;
    margin: 0 auto 14px;
}

.footer-card .footer-note {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.4);
    letter-spacing: 0.5px;
}

/* ---- Hide default Streamlit elements ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 780px; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------
# LOAD MODEL FILES
# -----------------------------------------------
@st.cache_resource
def load_model():
    with open("xgb_crop_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, encoder, features

try:
    model, encoder, features = load_model()
except FileNotFoundError:
    st.error("❌ Model files not found. Please upload xgb_crop_model.pkl, label_encoder.pkl, and feature_names.pkl.")
    st.stop()


# -----------------------------------------------
# HERO BANNER
# -----------------------------------------------
st.markdown("""
<div class="hero-banner">
    <span class="hero-icon">🌾</span>
    <div class="hero-title">CropSense</div>
    <div class="hero-subtitle">AI-powered crop recommendation for smarter farming</div>
    <div class="hero-badges">
        <span class="badge">🤖 XGBoost ML</span>
        <span class="badge">🧪 7 Soil & Climate Factors</span>
        <span class="badge">📊 Top-3 Predictions</span>
        <span class="badge">📄 PDF Report</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------
# WELCOME SECTION
# -----------------------------------------------
st.markdown("""
<div class="welcome-card">
    <h3>👋 Welcome to CropSense</h3>
    <p>
        CropSense uses a trained <strong>XGBoost machine learning model</strong> to recommend the most
        suitable crop for your field based on soil nutrients and climate conditions.
        Simply adjust the sliders below to match your farm's profile — and get an instant, data-driven recommendation
        along with confidence scores for the top 3 crops.
    </p>
</div>

<div class="feature-row">
    <div class="feature-pill">
        <span class="pill-icon">🌱</span>
        <span class="pill-label">Soil Nutrients</span>
    </div>
    <div class="feature-pill">
        <span class="pill-icon">🌡️</span>
        <span class="pill-label">Temperature</span>
    </div>
    <div class="feature-pill">
        <span class="pill-icon">💧</span>
        <span class="pill-label">Humidity</span>
    </div>
    <div class="feature-pill">
        <span class="pill-icon">🧪</span>
        <span class="pill-label">Soil pH</span>
    </div>
    <div class="feature-pill">
        <span class="pill-icon">🌧️</span>
        <span class="pill-label">Rainfall</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# -----------------------------------------------
# INPUT SECTION
# -----------------------------------------------
st.markdown("""
<div class="section-header">
    <span class="sh-icon">🔢</span>
    <h2>Enter Soil & Climate Parameters</h2>
</div>
""", unsafe_allow_html=True)

# Soil Nutrients
st.markdown('<div class="slider-card"><h4>🌿 Soil Nutrients</h4>', unsafe_allow_html=True)
N = st.slider("Nitrogen (N) — kg/ha", 0, 140, 50)
P = st.slider("Phosphorus (P) — kg/ha", 0, 140, 40)
K = st.slider("Potassium (K) — kg/ha", 0, 205, 40)
st.markdown('</div>', unsafe_allow_html=True)

# Climate Conditions
st.markdown('<div class="slider-card"><h4>🌤️ Climate Conditions</h4>', unsafe_allow_html=True)
temperature = st.slider("Temperature (°C)", 5.0, 45.0, 25.0)
humidity    = st.slider("Humidity (%)", 10.0, 100.0, 70.0)
rainfall    = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
st.markdown('</div>', unsafe_allow_html=True)

# Soil Property
st.markdown('<div class="slider-card"><h4>🧪 Soil Property</h4>', unsafe_allow_html=True)
ph = st.slider("Soil pH", 3.5, 9.5, 6.5)
st.markdown('</div>', unsafe_allow_html=True)

input_dict = {
    'N': N, 'P': P, 'K': K,
    'temperature': temperature,
    'humidity': humidity,
    'ph': ph,
    'rainfall': rainfall
}


# -----------------------------------------------
# PREDICT BUTTON
# -----------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🌱 Recommend My Crop"):

    user_input = np.array([input_dict[f] for f in features]).reshape(1, -1)
    predicted_class  = model.predict(user_input)
    predicted_proba  = model.predict_proba(user_input)
    crop_name        = encoder.inverse_transform(predicted_class)[0]
    confidence       = predicted_proba[0][predicted_class[0]] * 100

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-main">
        <div class="crop-label">Recommended Crop</div>
        <div class="crop-name">🌾 {crop_name.upper()}</div>
        <div class="confidence-badge">📊 Confidence: {confidence:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Top 3 Crops ----
    st.markdown("""
    <div class="section-header" style="margin-top:28px">
        <span class="sh-icon">🔝</span>
        <h2>Top 3 Crop Recommendations</h2>
    </div>
    """, unsafe_allow_html=True)

    top_3_idx = np.argsort(predicted_proba[0])[-3:][::-1]
    top_crops = [
        (encoder.inverse_transform([i])[0], predicted_proba[0][i] * 100)
        for i in top_3_idx
    ]

    for i, (crop, prob) in enumerate(top_crops, 1):
        medal = ["🥇", "🥈", "🥉"][i - 1]
        st.markdown(f"""
        <div style="background:#fff; border-radius:12px; padding:14px 20px; margin-bottom:10px;
                    display:flex; align-items:center; justify-content:space-between;
                    box-shadow:0 2px 10px rgba(26,58,42,0.07); border-left:4px solid var(--green-light);">
            <span style="font-weight:600; color:var(--green-dark); font-size:1rem;">
                {medal} &nbsp; {crop.title()}
            </span>
            <span style="background:var(--green-light); color:white; padding:4px 14px;
                         border-radius:20px; font-size:0.85rem; font-weight:600;">
                {prob:.2f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Bar chart
    crops, probs = zip(*top_crops)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#f5f0e8')
    ax.set_facecolor('#f5f0e8')
    bars = sns.barplot(x=list(crops), y=list(probs), palette=["#2d6a4f", "#52b788", "#95d5b2"], ax=ax)
    ax.set_ylabel("Confidence (%)", fontsize=10, color="#1a3a2a")
    ax.set_xlabel("Crop", fontsize=10, color="#1a3a2a")
    ax.set_title("Top 3 Crop Confidence Scores", fontsize=12, color="#1a3a2a", fontweight='bold', pad=12)
    ax.tick_params(colors="#1a3a2a")
    for spine in ax.spines.values():
        spine.set_edgecolor('#d4e8d8')
    st.pyplot(fig)

    # ---- PDF Report ----
    def generate_pdf():
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc    = SimpleDocTemplate(temp_file.name, pagesize=A4)
        styles = getSampleStyleSheet()
        story  = []
        story.append(Paragraph("<b>Crop Recommendation Report</b>", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Input Parameters</b>", styles['Heading2']))
        for k, v in input_dict.items():
            story.append(Paragraph(f"{k} : {v}", styles['Normal']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Prediction Result</b>", styles['Heading2']))
        story.append(Paragraph(f"Recommended Crop: <b>{crop_name}</b>", styles['Normal']))
        story.append(Paragraph(f"Confidence: {confidence:.2f}%", styles['Normal']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Top 3 Crops</b>", styles['Heading2']))
        for crop, prob in top_crops:
            story.append(Paragraph(f"{crop}: {prob:.2f}%", styles['Normal']))
        doc.build(story)
        return temp_file.name

    pdf_path = generate_pdf()
    st.markdown("<br>", unsafe_allow_html=True)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download Full Prediction Report (PDF)",
            data=f,
            file_name="crop_recommendation_report.pdf",
            mime="application/pdf"
        )


# -----------------------------------------------
# FOOTER / SIGN-OFF
# -----------------------------------------------
st.markdown("""
<div class="footer-card">
    <span class="footer-icon">🌿</span>
    <h3>Thank You for Using CropSense</h3>
    <p>
        We hope this recommendation helps you make a more informed decision for your next harvest.
        Remember — this tool is a guide powered by data. Always combine it with local agricultural
        expertise and seasonal knowledge for the best results.
    </p>
    <p>
        Happy farming, and may your harvest be plentiful! 🌾
    </p>
    <div class="footer-note">CropSense · XGBoost ML Model · Built with Streamlit</div>
</div>
""", unsafe_allow_html=True)