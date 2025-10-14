import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="Disease Prediction Dashboard", layout="centered")

st.title("🩺 Disease Prediction Dashboard")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    symptoms_df = pd.read_csv("symptoms_df.csv")
    medications_df = pd.read_csv("medications_df.csv")
    precautions_df = pd.read_csv("precautions_df.csv")
    workout_df = pd.read_csv("workout_df.csv")
    diets_df = pd.read_csv("diets_df.csv")
    return symptoms_df, medications_df, precautions_df, workout_df, diets_df

symptoms_df, medications_df, precautions_df, workout_df, diets_df = load_data()

# ---------------- SYMPTOM SELECTION LOGIC ----------------
all_symptoms = sorted(symptoms_df['Symptom'].unique())

def get_possible_diseases(selected_symptoms):
    """Filter diseases based on selected symptoms step by step."""
    filtered = symptoms_df.copy()
    for s in selected_symptoms:
        filtered = filtered[filtered['Symptom'] == s]
    return filtered['Disease'].unique().tolist()

st.subheader("Select up to 4 Symptoms")

symptom1 = st.selectbox("Symptom 1", ["Select"] + all_symptoms)

if symptom1 != "Select":
    diseases_after_1 = get_possible_diseases([symptom1])
    symptom2 = st.selectbox("Symptom 2", ["Select"] + diseases_after_1)
else:
    symptom2 = "Select"

if symptom2 != "Select":
    diseases_after_2 = get_possible_diseases([symptom1, symptom2])
    symptom3 = st.selectbox("Symptom 3", ["Select"] + diseases_after_2)
else:
    symptom3 = "Select"

if symptom3 != "Select":
    diseases_after_3 = get_possible_diseases([symptom1, symptom2, symptom3])
    symptom4 = st.selectbox("Symptom 4", ["Select"] + diseases_after_3)
else:
    symptom4 = "Select"

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Disease"):
    selected_symptoms = [s for s in [symptom1, symptom2, symptom3, symptom4] if s != "Select"]

    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        possible_diseases = get_possible_diseases(selected_symptoms)

        if not possible_diseases:
            st.error("No matching diseases found for the selected symptoms.")
        else:
            # Random confidence for display (simulate probabilities)
            top_diseases = random.sample(possible_diseases, min(3, len(possible_diseases)))
            confidences = sorted([round(random.uniform(0.75, 0.99), 2) for _ in top_diseases], reverse=True)

            st.markdown("### 🧠 Top 3 Possible Diseases")
            results = pd.DataFrame({
                "Disease": top_diseases,
                "Probability": confidences
            })
            st.dataframe(results)

            # ---------------- RECOMMENDATIONS ----------------
            st.markdown("## 💊 Recommendations")

            for disease in top_diseases:
                st.markdown(f"### 🦠 {disease}")

                # Medications
                meds = medications_df[medications_df["Disease"] == disease]["Medications"].values
                if len(meds):
                    st.write("**Medications:**", meds[0])
                else:
                    st.write("**Medications:** N/A")

                # Precautions
                prec = precautions_df[precautions_df["Disease"] == disease]["Precautions"].values
                if len(prec):
                    st.write("**Precautions:**", prec[0])
                else:
                    st.write("**Precautions:** N/A")

                # Workout
                work = workout_df[workout_df["Disease"] == disease]["Workout"].values
                if len(work):
                    st.write("**Workout:**", work[0])
                else:
                    st.write("**Workout:** N/A")

                # Diet
                diet = diets_df[diets_df["Disease"] == disease]["Diet"].values
                if len(diet):
                    st.write("**Diet:**", diet[0])
                else:
                    st.write("**Diet:** N/A")

                st.divider()
