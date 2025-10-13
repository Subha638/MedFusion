# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load CSV Data
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

@st.cache_data
def load_data():
    symptoms_df = pd.read_csv(os.path.join(BASE_DIR, "symtoms_df.csv"))
    diets_df = pd.read_csv(os.path.join(BASE_DIR, "diets.csv"))
    medications_df = pd.read_csv(os.path.join(BASE_DIR, "medications.csv"))
    precautions_df = pd.read_csv(os.path.join(BASE_DIR, "precautions_df.csv"))
    workout_df = pd.read_csv(os.path.join(BASE_DIR, "workout_df.csv"))
    return symptoms_df, diets_df, medications_df, precautions_df, workout_df

symptoms_df, diets_df, medications_df, precautions_df, workout_df = load_data()

# -----------------------------
# Preprocess Symptoms Data
# -----------------------------
symptom_cols = [col for col in symptoms_df.columns if 'Symptom' in col]
disease_col = 'Disease'

# Drop rows with missing disease
symptoms_df = symptoms_df.dropna(subset=[disease_col])
symptoms_df[disease_col] = symptoms_df[disease_col].astype(str).str.strip()

# Collect all unique symptoms
all_symptoms = set()
for col in symptom_cols:
    if col in symptoms_df.columns:
        all_symptoms.update([str(s).strip() for s in symptoms_df[col].dropna().unique() if str(s).strip()])
all_symptoms = sorted(list(all_symptoms))

# Binary feature matrix
X = pd.DataFrame(0, index=symptoms_df.index, columns=all_symptoms)
for idx, row in symptoms_df.iterrows():
    for col in symptom_cols:
        if col in symptoms_df.columns:
            sym = str(row[col]).strip()
            if sym and sym in all_symptoms:
                X.at[idx, sym] = 1

y = symptoms_df[disease_col]

# Train Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# -----------------------------
# Prediction & Recommendation Functions
# -----------------------------
def predict_disease(user_symptoms):
    input_vector = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for sym in user_symptoms:
        sym_str = str(sym).strip()
        if sym_str in all_symptoms:
            input_vector[sym_str] = 1

    prediction = clf.predict(input_vector)[0]
    probabilities = clf.predict_proba(input_vector)[0]
    top_3 = sorted(zip(clf.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3]
    return prediction, top_3

def get_recommendations(disease):
    # Diet
    diet_row = diets_df[diets_df['Disease'] == disease]
    diet = diet_row['Diet'].iloc[0] if not diet_row.empty else ["No data available"]

    # Medication
    med_row = medications_df[medications_df['Disease'] == disease]
    meds = med_row['Medication'].iloc[0] if not med_row.empty else ["No data available"]

    # Precautions
    prec_rows = precautions_df[precautions_df['Disease'] == disease]
    if not prec_rows.empty:
        prec_cols = [col for col in prec_rows.columns if 'Precaution' in col]
        precautions = prec_rows[prec_cols].values.flatten()
        precautions = [str(p).strip() for p in precautions if pd.notna(p) and str(p).strip() != '']
    else:
        precautions = ["No data available"]

    # Workouts
    workout_rows = workout_df[workout_df['disease'] == disease]
    workouts = [str(w).strip() for w in workout_rows['workout'].tolist()] if not workout_rows.empty else ["No data available"]

    return {
        'Diet': diet,
        'Medications': meds,
        'Precautions': precautions[:4],
        'Workouts': workouts[:5]
    }

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("🩺 Disease Prediction Dashboard")

# Cascading symptom selection
all_symptoms_options = list(all_symptoms)
symptom1 = st.selectbox("Symptom 1", ["Select"] + all_symptoms_options)
symptom2 = st.selectbox("Symptom 2", ["Select"])
symptom3 = st.selectbox("Symptom 3", ["Select"])
symptom4 = st.selectbox("Symptom 4", ["Select"])

# Filter next symptoms based on previous selection
if symptom1 != "Select":
    symptom2_options = X[X[symptom1] == 1].loc[:, all_symptoms].sum()
    symptom2_options = [s for s in symptom2_options[symptom2_options>0].index if s != symptom1]
    symptom2_options = ["Select"] + symptom2_options
    symptom2 = st.selectbox("Symptom 2", symptom2_options)

if symptom2 != "Select" and symptom1 != "Select":
    symptom3_options = X[(X[symptom1]==1) & (X[symptom2]==1)].loc[:, all_symptoms].sum()
    symptom3_options = [s for s in symptom3_options[symptom3_options>0].index if s not in [symptom1, symptom2]]
    symptom3_options = ["Select"] + symptom3_options
    symptom3 = st.selectbox("Symptom 3", symptom3_options)

if symptom3 != "Select" and symptom2 != "Select" and symptom1 != "Select":
    symptom4_options = X[(X[symptom1]==1) & (X[symptom2]==1) & (X[symptom3]==1)].loc[:, all_symptoms].sum()
    symptom4_options = [s for s in symptom4_options[symptom4_options>0].index if s not in [symptom1, symptom2, symptom3]]
    symptom4_options = ["Select"] + symptom4_options
    symptom4 = st.selectbox("Symptom 4", symptom4_options)

# Predict button
if st.button("Predict Disease"):
    selected_symptoms = [s for s in [symptom1, symptom2, symptom3, symptom4] if s != "Select"]
    if len(selected_symptoms) < 1:
        st.warning("Please select at least one symptom.")
    else:
        predicted_disease, top_3 = predict_disease(selected_symptoms)
        st.success(f"Predicted Disease: {predicted_disease}")

        st.subheader("Top 3 Probable Diseases")
        for disease, prob in top_3:
            st.write(f"- {disease}: {prob:.2f}")

        # Recommendations
        recs = get_recommendations(predicted_disease)
        st.subheader("Recommendations")
        st.markdown(f"**Diet:** {recs['Diet']}")
        st.markdown(f"**Medications:** {recs['Medications']}")
        st.markdown(f"**Precautions:** {', '.join(recs['Precautions'])}")
        st.markdown(f"**Workouts:** {', '.join(recs['Workouts'])}")

        # Top 3 probability chart
        probs = [p for d, p in top_3]
        diseases = [d for d, p in top_3]
        fig, ax = plt.subplots()
        ax.barh(diseases, probs)
        ax.set_xlabel("Probability")
        ax.set_title("Top 3 Disease Probabilities")
        st.pyplot(fig)

