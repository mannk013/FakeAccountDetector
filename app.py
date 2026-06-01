import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Account Detector",
    page_icon="🤖",
    layout="centered"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .stButton>button {
        background-color: #e74c3c;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #c0392b;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .fake-box {
        background-color: #c0392b;
        color: white;
    }
    .real-box {
        background-color: #27ae60;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ── LOAD & TRAIN MODEL ───────────────────────────────────────
@st.cache_resource
def load_model():
    fake = pd.read_csv('Data/fake_users.csv')
    real = pd.read_csv('Data/real_users.csv')
    fake['is_fake'] = 1
    real['is_fake'] = 0
    df = pd.concat([fake, real], ignore_index=True)
    features = [
        'statuses_count', 'followers_count', 'friends_count',
        'favourites_count', 'listed_count', 'default_profile',
        'default_profile_image', 'geo_enabled',
        'profile_use_background_image', 'profile_background_tile',
        'utc_offset', 'protected', 'verified'
    ]
    df_model = df[features + ['is_fake']].fillna(0)
    X = df_model[features]
    y = df_model['is_fake']
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf, features

model, features = load_model()

# ── HEADER ───────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center'>🤖 Fake Account Detector</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray'>Enter account details to check if it's fake or real</p>",
            unsafe_allow_html=True)
st.divider()

# ── METRICS BAR ──────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Model", "Random Forest")
col2.metric("Accuracy", "91%")
col3.metric("AUC-ROC", "0.955")
st.divider()

# ── INPUT FORM ───────────────────────────────────────────────
st.subheader(" `Account Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Activity**")
    statuses_count = st.number_input(" Number of Posts", min_value=0, value=100)
    favourites_count = st.number_input("❤ Likes Given", min_value=0, value=200)
    listed_count = st.number_input(" Listed Count", min_value=0, value=5)
    utc_offset = st.number_input(" UTC Offset", min_value=-12, max_value=12, value=0)

with col2:
    st.markdown("**Social**")
    followers_count = st.number_input(" Followers", min_value=0, value=500)
    friends_count = st.number_input(" Following", min_value=0, value=300)
    geo_enabled = st.selectbox(" Location Enabled?", [0, 1],
                                format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")
    verified = st.selectbox(" Verified Account?", [0, 1],
                             format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")

st.markdown("**Profile Settings**")
col3, col4, col5 = st.columns(3)
with col3:
    default_profile = st.selectbox("Default Profile?", [0, 1],
                                    format_func=lambda x: "Yes" if x == 1 else "No")
    default_profile_image = st.selectbox("Default Image?", [0, 1],
                                          format_func=lambda x: "Yes" if x == 1 else "No")
with col4:
    profile_use_background_image = st.selectbox("Background Image?", [0, 1],
                                                  format_func=lambda x: "Yes" if x == 1 else "No")
    profile_background_tile = st.selectbox("Background Tile?", [0, 1],
                                            format_func=lambda x: "Yes" if x == 1 else "No")
with col5:
    protected = st.selectbox("Protected?", [0, 1],
                              format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

# ── PREDICTION ───────────────────────────────────────────────
if st.button(" Analyze Account"):
    input_data = pd.DataFrame([[
        statuses_count, followers_count, friends_count,
        favourites_count, listed_count, default_profile,
        default_profile_image, geo_enabled,
        profile_use_background_image, profile_background_tile,
        utc_offset, protected, verified
    ]], columns=features)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    fake_prob = probability[1] * 100
    real_prob = probability[0] * 100

    if prediction == 1:
        st.markdown(f"""
            <div class='result-box fake-box'>
                 FAKE ACCOUNT<br>
                <span style='font-size:16px'>Confidence: {fake_prob:.1f}%</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='result-box real-box'>
                 REAL ACCOUNT<br>
                <span style='font-size:16px'>Confidence: {real_prob:.1f}%</span>
            </div>
        """, unsafe_allow_html=True)

    # Probability chart
    st.markdown("###")
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh([''], [real_prob], color='#27ae60', label=f'Real {real_prob:.1f}%')
    ax.barh([''], [fake_prob], left=[real_prob], color='#e74c3c',
            label=f'Fake {fake_prob:.1f}%')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability %')
    ax.legend(loc='upper right')
    ax.set_title('Prediction Probability')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    st.pyplot(fig)