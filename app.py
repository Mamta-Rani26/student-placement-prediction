import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓"
)

st.title("🎓 Student Placement Prediction & Recommendation System")

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("placement.csv")

# Drop ID Column
if "College_ID" in df.columns:
    df = df.drop("College_ID", axis=1)

# Encode Internship
df["Internship_Experience"] = LabelEncoder().fit_transform(
    df["Internship_Experience"]
)

# Encode Placement
df["Placement"] = LabelEncoder().fit_transform(
    df["Placement"]
)

# =========================
# FEATURES & TARGET
# =========================

X = df.drop("Placement", axis=1)
y = df["Placement"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# SMOTE
# =========================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# =========================
# RANDOM FOREST MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_smote, y_train_smote)

# =========================
# USER INPUT
# =========================

st.header("Enter Student Details")

iq = st.number_input(
    "IQ Score",
    min_value=50,
    max_value=200,
    value=100
)

prev_sem = st.number_input(
    "Previous Semester Result (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)

academic = st.number_input(
    "Academic Performance (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

internship = st.selectbox(
    "Internship Experience",
    ["No", "Yes"]
)

extra = st.number_input(
    "Extra Curricular Score",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

communication = st.number_input(
    "Communication Skills",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

projects = st.number_input(
    "Projects Completed",
    min_value=0,
    max_value=20,
    value=3
)

# Convert Internship
internship_encoded = 1 if internship == "Yes" else 0

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict Placement"):

    input_data = pd.DataFrame([[
        iq,
        prev_sem,
        cgpa,
        academic,
        internship_encoded,
        extra,
        communication,
        projects
    ]], columns=X.columns)

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1] * 100

    st.subheader(
        f"🎯 Placement Probability: {probability:.2f}%"
    )

    # Placement Status

    if probability >= 80:
        st.success("✅ High Placement Chance")

    elif probability >= 60:
        st.warning("⚠ Moderate Placement Chance")

    else:
        st.error("❌ Low Placement Chance")

    # =====================
    # RECOMMENDATIONS
    # =====================

    st.subheader("📌 Personalized Recommendations")

    recommendations = []

    # CGPA
    if cgpa < 7:
        recommendations.append(
            "Improve CGPA above 7.0."
        )

    elif cgpa < 8:
        recommendations.append(
            "Target CGPA 8+ for better opportunities."
        )

    # Communication
    if communication < 70:
        recommendations.append(
            "Improve communication skills through mock interviews and presentations."
        )

    elif communication < 85:
        recommendations.append(
            "Practice public speaking to further improve communication."
        )

    # Projects
    if projects < 2:
        recommendations.append(
            "Build at least 2-3 real-world projects."
        )

    elif projects < 5:
        recommendations.append(
            "Add advanced AI/ML/Web Development projects."
        )

    # Internship
    if internship_encoded == 0:
        recommendations.append(
            "Gain internship experience."
        )

    # IQ
    if iq < 100:
        recommendations.append(
            "Practice aptitude and reasoning questions."
        )

    # Previous Semester
    if prev_sem < 70:
        recommendations.append(
            "Improve semester marks."
        )

    # Academic Performance
    if academic < 70:
        recommendations.append(
            "Focus on academic consistency."
        )

    # Extra Curricular
    if extra < 50:
        recommendations.append(
            "Participate in hackathons, coding contests and clubs."
        )

    # High Profile Recommendations
    if probability >= 80:
        recommendations.append(
            "Apply for top product-based companies."
        )

        recommendations.append(
            "Strengthen GitHub and LinkedIn profile."
        )

        recommendations.append(
            "Practice DSA and aptitude for higher packages."
        )

    # Show Recommendations

    if len(recommendations) == 0:
        st.success(
            "Excellent Profile! Keep maintaining your performance."
        )

    else:
        for rec in recommendations:
            st.write("✅", rec)

    # =====================
    # PROFILE ANALYSIS
    # =====================

    st.subheader("📊 Profile Analysis")

    if probability >= 80:
        st.success(
            "Your profile is placement-ready and highly competitive."
        )

    elif probability >= 60:
        st.warning(
            "Your profile is good, but improving a few areas can significantly increase placement chances."
        )

    else:
        st.error(
            "Your profile requires improvement in academics, skills and practical exposure."
        )