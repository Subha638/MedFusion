# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# Caching function to load data
# ----------------------------
@st.cache_data
def load_data():
    # Load all CSVs from repo (must be committed)
    symptoms_df = pd.read_csv("symptoms_df.csv")
    diets_df = pd.read_csv("diets.csv")
    medications_df = pd.read_csv("medications.csv")
    precautions_df = pd.read_csv("precautions_df.csv")
    workout_df = pd.read_csv("workout_df.csv")
    return symptoms_df, diets_df, medications_df, precautions_df, workout_df

# ----------------------------
# Cache model training
# ----------------------------
@st.cache_resource
def train_model(symptoms_df):
    symptom_cols = [col for col in symptoms_df.columns if "Symptom" in col]
    disease_col = "Disease"

    # Clean data
    symptoms_df = symptoms_df.dropna(subset=[disease_col])
    symptoms_df[disease_col] = symptoms_df[disease_col].astype(str).str.strip()

    # Collect unique symptoms
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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    return clf, all_symptoms, symptom_cols, disease_col, X

# ----------------------------
# Load data and train model
# ----------------------------
st.title("Disease Prediction App")
st.markdown("Select symptoms to predict possible disease and get recommendations.")

symptoms_df, diets_df, medications_df, precautions_df, workout_df = load_data()
clf, all_symptoms, symptom_cols, disease_col, X = train_model(symptoms_df)

# ----------------------------
# Prediction function
# ----------------------------
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

# ----------------------------
# Recommendations function
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
# Streamlit cascading symptom selection
# ----------------------------
selected_symptoms = []
col1, col2 = st.columns(2)
with col1:
    symptom1 = st.selectbox("Symptom 1", [None]+all_symptoms)
with col2:
    symptom2 = st.selectbox("Symptom 2", [None]+all_symptoms)

symptom3 = st.selectbox("Symptom 3", [None]+all_symptoms)
symptom4 = st.selectbox("Symptom 4", [None]+all_symptoms)

for s in [symptom1, symptom2, symptom3, symptom4]:
    if s is not None:
        selected_symptoms.append(s)

if st.button("Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")
    else:
        predicted_disease, top_3 = predict_disease(selected_symptoms)
        st.success(f"Predicted Disease: {predicted_disease}")

        st.subheader("Top 3 probable diseases:")
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
