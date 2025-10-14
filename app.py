# app.py
import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load CSVs
# ----------------------------
@st.cache_data
def load_data():
    symptoms_df = pd.read_csv("symtoms_df.csv")
    diets_df = pd.read_csv("diets.csv")
    medications_df = pd.read_csv("medications.csv")
    precautions_df = pd.read_csv("precautions_df.csv")
    return symptoms_df, diets_df, medications_df, precautions_df

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    clf = joblib.load("trained_disease_model.joblib")
    # Assuming model was trained with these symptom names
    symptoms_df = pd.read_csv("symtoms_df.csv")
    symptom_cols = [col for col in symptoms_df.columns if "Symptom" in col]

    # Collect all unique symptoms used in training
    all_symptoms = set()
    for col in symptom_cols:
        all_symptoms.update([str(s).strip() for s in symptoms_df[col].dropna().unique()])
    all_symptoms = sorted(list(all_symptoms))

    return clf, all_symptoms, symptom_cols

# ----------------------------
# Predict disease (always show top 3)
# ----------------------------
def predict_disease(user_symptoms, clf, all_symptoms):
    # Build input vector for prediction
    input_vector = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for sym in user_symptoms:
        if sym in all_symptoms:
            input_vector[sym] = 1

    # Predict probabilities for all diseases
    all_probs = clf.predict_proba(input_vector)[0]
    disease_probs = dict(zip(clf.classes_, all_probs))

    # Sort by probability and take top 3
    top_3 = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    predicted_disease = top_3[0][0] if top_3 else "No match found"

    return predicted_disease, top_3

# ----------------------------
# Get recommendations (without workouts)
# ----------------------------
def get_recommendations(disease, diets_df, medications_df, precautions_df):
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
        precautions = [str(p).strip() for p in precautions if p and str(p).strip() != '']
    else:
        precautions = ["No data available"]

    return {
        'Diet': diet,
        'Medications': meds,
        'Precautions': precautions[:4]
    }

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Welcome to MedFusion")
st.markdown("Select symptoms to predict possible disease 🤒 and get recommendations.")

# Load data and trained model
symptoms_df, diets_df, medications_df, precautions_df = load_data()
clf, all_symptoms, symptom_cols = load_model()

# Cascading symptom selection dynamically
selected_symptoms = []
filtered_df = symptoms_df.copy()

for i, col in enumerate(symptom_cols[:4]):  # Max 4 symptoms
    options = sorted(filtered_df[col].dropna().unique())
    symptom = st.selectbox(f"Symptom {i+1}", [None] + options)
    if symptom:
        selected_symptoms.append(symptom)
        filtered_df = filtered_df[filtered_df[col] == symptom]

# Predict button
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        predicted_disease, top_3 = predict_disease(selected_symptoms, clf, all_symptoms)
        st.success(f"Predicted Disease: {predicted_disease}")

        st.subheader("Top 3 probable diseases:")
        for d, p in top_3:
            st.write(f"{d}: {p:.2f}")

        recommendations = get_recommendations(predicted_disease, diets_df, medications_df, precautions_df)
        st.subheader("Diet Recommendations")
        st.write(recommendations['Diet'])
        st.subheader("Medication Recommendations")
        st.write(recommendations['Medications'])
        st.subheader("Precautions")
        st.write(recommendations['Precautions'])
