import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="MedFusion - Disease Predictor", layout="wide")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    sym_df = pd.read_csv("symtoms_df.csv")
    workout_df = pd.read_csv("workout_df.csv")
    diet_df = pd.read_csv("diets.csv")
    med_df = pd.read_csv("medications.csv")
    pre_df = pd.read_csv("precautions_df.csv")
    return sym_df, workout_df, diet_df, med_df, pre_df

symtoms_df, workout_df, diets_df, medications_df, precautions_df = load_data()

# ---------- Normalize Columns ----------
def normalize_cols(df):
    df.columns = df.columns.str.strip().str.lower()
    return df

symtoms_df = normalize_cols(symtoms_df)
workout_df = normalize_cols(workout_df)
diets_df = normalize_cols(diets_df)
medications_df = normalize_cols(medications_df)
precautions_df = normalize_cols(precautions_df)

# --- Auto-detect column names ---
symptom_col = next((c for c in symtoms_df.columns if "symptom" in c.lower()), None)
disease_col = next((c for c in symtoms_df.columns if "disease" in c.lower()), None)

if not symptom_col or not disease_col:
    st.error("❌ 'symtoms_df.csv' must contain columns for symptoms and diseases.")
    st.stop()

# ---------- Title ----------
st.title("🧠 MedFusion - Smart Disease Predictor")
st.write("Select up to 4 symptoms to predict the most likely diseases and get recommendations.")

# ---------- Step 1 ----------
all_symptoms = sorted(symtoms_df[symptom_col].unique())
symptom1 = st.selectbox("🩺 Symptom 1", [""] + all_symptoms)

# ---------- Helper Function ----------
def get_next_symptoms(selected_symptom, prev=[]):
    """Return possible next symptoms based on previous selections."""
    if not selected_symptom:
        return []
    related = symtoms_df[symtoms_df[symptom_col] == selected_symptom][disease_col].unique()
    next_sym = symtoms_df[symtoms_df[disease_col].isin(related)][symptom_col].unique()
    next_sym = [s for s in next_sym if s not in prev]
    return sorted(next_sym)

# ---------- Step 2–4 ----------
symptom2 = symptom3 = symptom4 = ""

if symptom1:
    possible2 = get_next_symptoms(symptom1, [symptom1])
    symptom2 = st.selectbox("🩺 Symptom 2", [""] + possible2)

if symptom2:
    possible3 = get_next_symptoms(symptom2, [symptom1, symptom2])
    symptom3 = st.selectbox("🩺 Symptom 3", [""] + possible3)

if symptom3:
    possible4 = get_next_symptoms(symptom3, [symptom1, symptom2, symptom3])
    symptom4 = st.selectbox("🩺 Symptom 4", [""] + possible4)

selected_symptoms = [s for s in [symptom1, symptom2, symptom3, symptom4] if s]

# ---------- Safe Column Lookup ----------
def safe_lookup(df, disease_name, key_word):
    df = normalize_cols(df)
    disease_col = next((c for c in df.columns if "disease" in c.lower()), None)
    value_col = next((c for c in df.columns if key_word.lower() in c.lower()), None)
    if disease_col and value_col:
        res = df.loc[df[disease_col] == disease_name, value_col]
        return res.values[0] if not res.empty else "N/A"
    return "N/A"

# ---------- Prediction ----------
if st.button("🔍 Predict Disease") and selected_symptoms:
    matching = symtoms_df[symtoms_df[symptom_col].isin(selected_symptoms)]
    counts = matching[disease_col].value_counts()

    if not counts.empty:
        st.subheader("🏥 Top 3 Possible Diseases")
        top3 = counts.head(3)
        for disease, count in top3.items():
            prob = round((count / len(selected_symptoms)) * 100, 2)
            st.markdown(f"**{disease}** — Probability: {prob}%")

            meds = safe_lookup(medications_df, disease, "medication")
            precs = safe_lookup(precautions_df, disease, "precaution")
            diet = safe_lookup(diets_df, disease, "diet")
            work = safe_lookup(workout_df, disease, "workout")

            st.markdown(f"""
            💊 **Medications:** {meds}  
            ⚠️ **Precautions:** {precs}  
            🥗 **Diet:** {diet}  
            🏃 **Workout:** {work}
            """)
    else:
        st.error("No matching diseases found.")
else:
    st.info("Please select at least one symptom to start prediction.")
