# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------
# Load CSVs
# ----------------------------
@st.cache_data
def load_data():
    symptoms_df = pd.read_csv("symtoms_df.csv")  # note the typo matches your file
    diets_df = pd.read_csv("diets.csv")
    medications_df = pd.read_csv("medications.csv")
    precautions_df = pd.read_csv("precautions.csv")
    workout_df = pd.read_csv("workout_df.csv")
    return symptoms_df, diets_df, medications_df, precautions_df, workout_df

# ----------------------------
# Train model
# ----------------------------
@st.cache_resource
def train_model(symptoms_df):
    symptom_cols = [col for col in symptoms_df.columns if "Symptom" in col]
    disease_col = "Disease"

    # Clean data
    symptoms_df = symptoms_df.dropna(subset=[disease_col])
    symptoms_df[disease_col] = symptoms_df[disease_col].astype(str).str.strip()

    # Collect all unique symptoms
    all_symptoms = set()
    for col in symptom_cols:
        if col in symptoms_df.columns:
            all_symptoms.update([str(s).strip() for s in symptoms_df[col].dropna().unique() if str(s).strip()])
    all_symptoms = sorted(list(all_symptoms))

    # Create binary feature matrix
    X = pd.DataFrame(0, index=symptoms_df.index, columns=all_symptoms)
    for idx, row in symptoms_df.iterrows():
        for col in symptom_cols:
            if col in symptoms_df.columns:
                sym = str(row[col]).strip()
                if sym and sym in all_symptoms:
                    X.at[idx, sym] = 1
    y = symptoms_df[disease_col]

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    return clf, all_symptoms, symptom_cols, disease_col, X

# ----------------------------
# Prediction
# ----------------------------
def predict_disease(selected_symptoms):
    input_vector = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for sym in selected_symptoms:
        if sym in all_symptoms:
            input_vector[sym] = 1
    prediction = clf.predict(input_vector)[0]
    probabilities = clf.predict_proba(input_vector)[0]
    top_3 = sorted(zip(clf.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3]
    return prediction, top_3

# ----------------------------
# Recommendations
# ----------------------------
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

# ----------------------------
# Load data and train model
# ----------------------------
st.title("Disease Prediction App")
symptoms_df, diets_df, medications_df, precautions_df, workout_df = load_data()
clf, all_symptoms, symptom_cols, disease_col, X = train_model(symptoms_df)

# ----------------------------
# Cascading symptom selection
# ----------------------------
st.subheader("Select Symptoms Sequentially:")

def get_next_symptoms(selected_prev):
    """Return symptoms co-occurring with selected_prev in dataset"""
    if not selected_prev:
        return all_symptoms
    # Filter rows where all previous symptoms are 1
    mask = np.ones(len(X), dtype=bool)
    for s in selected_prev:
        mask &= (X[s] == 1)
    # Find remaining symptoms in these rows
    co_occur = []
    for col in all_symptoms:
        if col not in selected_prev and X.loc[mask, col].sum() > 0:
            co_occur.append(col)
    return co_occur

selected_symptoms = []

sym1_options = get_next_symptoms([])
symptom1 = st.selectbox("Symptom 1", [None]+sym1_options)
if symptom1:
    selected_symptoms.append(symptom1)
    sym2_options = get_next_symptoms(selected_symptoms)
else:
    sym2_options = []

symptom2 = st.selectbox("Symptom 2", [None]+sym2_options)
if symptom2:
    selected_symptoms.append(symptom2)
    sym3_options = get_next_symptoms(selected_symptoms)
else:
    sym3_options = []

symptom3 = st.selectbox("Symptom 3", [None]+sym3_options)
if symptom3:
    selected_symptoms.append(symptom3)
    sym4_options = get_next_symptoms(selected_symptoms)
else:
    sym4_options = []

symptom4 = st.selectbox("Symptom 4", [None]+sym4_options)
if symptom4:
    selected_symptoms.append(symptom4)

# ----------------------------
# Prediction button
# ----------------------------
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        predicted_disease, top_3 = predict_disease(selected_symptoms)
        st.success(f"Predicted Disease: {predicted_disease}")

        st.subheader("Top 3 Probable Diseases:")
        for d, p in top_3:
            st.write(f"{d}: {p:.2f}")

        recommendations = get_recommendations(predicted_disease)
        st.subheader("Diet Recommendations")
        st.write(recommendations['Diet'])
        st.subheader("Medication Recommendations")
        st.write(recommendations['Medications'])
        st.subheader("Precautions")
        st.write(recommendations['Precautions'])
        st.subheader("Workout Tips")
        st.write(recommendations['Workouts'])
