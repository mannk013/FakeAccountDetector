import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Account Detector",
    page_icon="🤖",
    layout="centered"
)


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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    return rf, features


model, features = load_model()

# ── UI ───────────────────────────────────────────────────────
st.title(" Social Media Fake Account Detector")
st.markdown("Enter account details below to check if it's **fake or real**.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    statuses_count = st.number_input("Number of Posts", min_value=0, value=100)
    followers_count = st.number_input("Followers", min_value=0, value=500)
    friends_count = st.number_input("Following", min_value=0, value=300)
    favourites_count = st.number_input("Likes Given", min_value=0, value=200)
    listed_count = st.number_input("Listed Count", min_value=0, value=5)
    geo_enabled = st.selectbox("Location Enabled?", [0, 1],
                               format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    default_profile = st.selectbox("Default Profile?", [0, 1],
                                   format_func=lambda x: "Yes" if x == 1 else "No")
    default_profile_image = st.selectbox("Default Profile Image?", [0, 1],
                                         format_func=lambda x: "Yes" if x == 1 else "No")
    profile_use_background_image = st.selectbox("Background Image?", [0, 1],
                                                format_func=lambda x: "Yes" if x == 1 else "No")
    profile_background_tile = st.selectbox("Background Tile?", [0, 1],
                                           format_func=lambda x: "Yes" if x == 1 else "No")
    utc_offset = st.number_input("UTC Offset", min_value=-12, max_value=12, value=0)
    protected = st.selectbox("Protected Account?", [0, 1],
                             format_func=lambda x: "Yes" if x == 1 else "No")
    verified = st.selectbox("Verified Account?", [0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

# ── PREDICTION ───────────────────────────────────────────────
if st.button(" Check Account", use_container_width=True):
    input_data = pd.DataFrame([[
        statuses_count, followers_count, friends_count,
        favourites_count, listed_count, default_profile,
        default_profile_image, geo_enabled,
        profile_use_background_image, profile_background_tile,
        utc_offset, protected, verified
    ]], columns=features)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Result:")

    if prediction == 1:
        st.error(f" FAKE ACCOUNT detected!")
        st.metric("Fake probability", f"{probability[1] * 100:.1f}%")
    else:
        st.success(f"REAL ACCOUNT")
        st.metric("Real probability", f"{probability[0] * 100:.1f}%")

    st.progress(float(probability[1]))
    st.caption(f"Fake: {probability[1] * 100:.1f}% | Real: {probability[0] * 100:.1f}%")