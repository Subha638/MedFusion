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

# Normalize column names
symtoms_df.columns = [col.strip().title() for col in symtoms_df.columns]
medications_df.columns = [col.strip().title() for col in medications_df.columns]
precautions_df.columns = [col.strip().title() for col in precautions_df.columns]
diets_df.columns = [col.strip().title() for col in diets_df.columns]
workout_df.columns = [col.strip().title() for col in workout_df.columns]

# ---------- Title ----------
st.title("🧠 MedFusion - Smart Disease Predictor")
st.write("Select up to 4 symptoms to predict the most likely diseases and get recommendations.")

# ---------- Step 1: Symptom 1 ----------
all_symptoms = sorted(symtoms_df['Symptom'].unique())
symptom1 = st.selectbox("🩺 Symptom 1", [""] + all_symptoms)

# ---------- Step 2–4: Dynamic Symptom Filtering ----------
def get_next_symptoms(selected_symptom, prev=[]):
    """Return possible next symptoms based on previous selections."""
    if not selected_symptom:
        return []
    related = symtoms_df[symtoms_df['Symptom'] == selected_symptom]['Disease'].unique()
    next_sym = symtoms_df[symtoms_df['Disease'].isin(related)]['Symptom'].unique()
    next_sym = [s for s in next_sym if s not in prev]
    return sorted(next_sym)

symptom2 = ""
symptom3 = ""
symptom4 = ""

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

# ---------- Prediction Logic ----------
if st.button("🔍 Predict Disease") and selected_symptoms:
    matching_diseases = symtoms_df[symtoms_df["Symptom"].isin(selected_symptoms)]
    disease_counts = matching_diseases["Disease"].value_counts()

    if not disease_counts.empty:
        top3 = disease_counts.head(3)
        st.subheader("🏥 Top 3 Possible Diseases")
        for disease, count in top3.items():
            prob = round((count / len(selected_symptoms)) * 100, 2)
            st.markdown(f"**{disease}** — Probability: {prob}%")

            # --- Fetch Recommendations Safely ---
            def safe_lookup(df, col_disease="Disease", col_value="Medications"):
                col_disease = next((c for c in df.columns if c.lower() == "disease"), None)
                col_value = next((c for c in df.columns if col_value.lower() in c.lower()), None)
                if col_disease and col_value:
                    res = df.loc[df[col_disease] == disease, col_value]
                    return res.values[0] if not res.empty else "N/A"
                return "N/A"

            meds = safe_lookup(medications_df, col_value="Medications")
            precs = safe_lookup(precautions_df, col_value="Precautions")
            diet = safe_lookup(diets_df, col_value="Diet")
            work = safe_lookup(workout_df, col_value="Workout")

            st.markdown(f"""
            💊 **Medications:** {meds}  
            ⚠️ **Precautions:** {precs}  
            🥗 **Diet:** {diet}  
            🏃 **Workout:** {work}
            """)

    else:
        st.error("No matching diseases found for selected symptoms.")
else:
    st.info("Please select at least one symptom to predict.")
