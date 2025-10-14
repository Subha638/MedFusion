import streamlit as st
import pandas as pd
import random
import os

st.set_page_config(page_title="🩺 Disease Prediction Dashboard", layout="centered")
st.title("🩺 Disease Prediction Dashboard")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    symtoms_df = pd.read_csv("symtoms_df.csv")
    medications_df = pd.read_csv("medications.csv")
    precautions_df = pd.read_csv("precautions_df.csv")
    workout_df = pd.read_csv("workout_df.csv")
    diets_df = pd.read_csv("diets.csv")

    # Reshape symptom data if needed
    symptom_cols = [col for col in symtoms_df.columns if "symptom" in col.lower()]
    melted_df = symtoms_df.melt(
        id_vars=["Disease"],
        value_vars=symptom_cols,
        var_name="SymptomType",
        value_name="Symptom"
    ).dropna()
    melted_df["Symptom"] = melted_df["Symptom"].str.strip().str.lower()

    return melted_df, medications_df, precautions_df, workout_df, diets_df


symtoms_df, medications_df, precautions_df, workout_df, diets_df = load_data()

# All unique symptoms
all_symptoms = sorted(symtoms_df["Symptom"].unique())

# ---------------- FUNCTIONS ----------------
def diseases_with_symptom(symptoms):
    """Return diseases that have all selected symptoms."""
    df = symtoms_df
    for s in symptoms:
        df = df[df["Disease"].isin(df[df["Symptom"] == s]["Disease"])]
    return df["Disease"].unique().tolist()

def next_possible_symptoms(selected_symptoms):
    """Return next symptoms possible for diseases with selected symptoms."""
    diseases = diseases_with_symptom(selected_symptoms)
    if not diseases:
        return []
    possible = symtoms_df[symtoms_df["Disease"].isin(diseases)]["Symptom"].unique().tolist()
    return sorted([s for s in possible if s not in selected_symptoms])

def predict_disease(selected_symptoms):
    """Simple heuristic: rank by symptom overlap"""
    disease_scores = {}
    for disease in symtoms_df["Disease"].unique():
        disease_symptoms = symtoms_df[symtoms_df["Disease"] == disease]["Symptom"].tolist()
        overlap = len(set(selected_symptoms) & set(disease_symptoms))
        if overlap > 0:
            disease_scores[disease] = overlap / len(disease_symptoms)
    sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_diseases[:3]

# ---------------- STREAMLIT UI ----------------
st.subheader("Select up to 4 Symptoms")

symptom1 = st.selectbox("Symptom 1", ["Select"] + all_symptoms)

if symptom1 != "Select":
    next_symptoms_1 = next_possible_symptoms([symptom1])
    symptom2 = st.selectbox("Symptom 2", ["Select"] + next_symptoms_1)
else:
    symptom2 = "Select"

if symptom2 != "Select":
    next_symptoms_2 = next_possible_symptoms([symptom1, symptom2])
    symptom3 = st.selectbox("Symptom 3", ["Select"] + next_symptoms_2)
else:
    symptom3 = "Select"

if symptom3 != "Select":
    next_symptoms_3 = next_possible_symptoms([symptom1, symptom2, symptom3])
    symptom4 = st.selectbox("Symptom 4", ["Select"] + next_symptoms_3)
else:
    symptom4 = "Select"

# ---------------- PREDICT ----------------
if st.button("🔍 Predict Disease"):
    selected = [s for s in [symptom1, symptom2, symptom3, symptom4] if s != "Select"]
    if not selected:
        st.warning("Please select at least one symptom.")
    else:
        top3 = predict_disease(selected)
        if not top3:
            st.error("No disease matches your selected symptoms.")
        else:
            st.success("Prediction complete!")
            st.markdown("### 🧠 Top 3 Possible Diseases")
            for disease, score in top3:
                st.markdown(f"**{disease}** — Probability: {round(score*100,2)}%")

                # Show recommendations
                st.markdown("**💊 Medications:**")
                meds = medications_df.loc[medications_df["Disease"] == disease, "Medications"]
                st.write(meds.values[0] if not meds.empty else "N/A")

                st.markdown("**⚠️ Precautions:**")
                prec = precautions_df.loc[precautions_df["Disease"] == disease, "Precautions"]
                st.write(prec.values[0] if not prec.empty else "N/A")

                st.markdown("**🏃 Workout:**")
                work = workout_df.loc[workout_df["Disease"] == disease, "Workout"]
                st.write(work.values[0] if not work.empty else "N/A")

                st.markdown("**🥗 Diet:**")
                diet = diets_df.loc[diets_df["Disease"] == disease, "Diet"]
                st.write(diet.values[0] if not diet.empty else "N/A")

                st.divider()
