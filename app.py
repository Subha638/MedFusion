# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------
# Caching function to load data
# ----------------------------
@st.cache_data
def load_data():
    symptoms_df = pd.read_csv("symtoms_df.csv")
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
    filtered_df = symptoms_df.copy()
    for i, sym in enumerate(user_symptoms):
        if i < len(symptom_cols):
            filtered_df = filtered_df[filtered_df[symptom_cols[i]] == sym]

    # Build input vector for prediction
    input_vector = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for sym in user_symptoms:
        sym_str = str(sym).strip()
        if sym_str in all_symptoms:
            input_vector[sym_str] = 1

    # Predict probabilities only for diseases in filtered_df
    possible_diseases = filtered_df['Disease'].unique()
    all_probs = clf.predict_proba(input_vector)[0]
    disease_probs = dict(zip(clf.classes_, all_probs))
    disease_probs_filtered = {d: disease_probs[d] for d in possible_diseases if d in disease_probs}
    top_3 = sorted(disease_probs_filtered.items(), key=lambda x: x[1], reverse=True)[:3]
    predicted_disease = top_3[0][0] if top_3 else "No match found"
    return predicted_disease, top_3

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
# Cascading symptom selection
# ----------------------------
selected_symptoms = []

# Symptom 1
symptom1 = st.selectbox("Symptom 1", [None] + sorted(symptoms_df[symptom_cols[0]].dropna().unique()))
if symptom1:
    selected_symptoms.append(symptom1)

# Symptom 2
if symptom1:
    filtered_df = symptoms_df[symptoms_df[symptom_cols[0]] == symptom1]
    symptom2_options = []
    if len(symptom_cols) > 1:
        for col in symptom_cols[1:]:
            symptom2_options.extend(filtered_df[col].dropna().unique())
    symptom2_options = sorted(list(set(symptom2_options)))
    symptom2 = st.selectbox("Symptom 2", [None] + symptom2_options)
    if symptom2:
        selected_symptoms.append(symptom2)
else:
    symptom2 = None

# Symptom 3
if selected_symptoms:
    filtered_df = symptoms_df.copy()
    for i, sym in enumerate(selected_symptoms):
        if i < len(symptom_cols):
            filtered_df = filtered_df[filtered_df[symptom_cols[i]] == sym]
    symptom3_options = []
    if len(symptom_cols) > 2:
        for col in symptom_cols[2:]:
            symptom3_options.extend(filtered_df[col].dropna().unique())
    symptom3_options = sorted(list(set(symptom3_options)))
    symptom3 = st.selectbox("Symptom 3", [None] + symptom3_options)
    if symptom3:
        selected_symptoms.append(symptom3)
else:
    symptom3 = None

# Symptom 4
if selected_symptoms:
    filtered_df = symptoms_df.copy()
    for i, sym in enumerate(selected_symptoms):
        if i < len(symptom_cols):
            filtered_df = filtered_df[filtered_df[symptom_cols[i]] == sym]
    symptom4_options = []
    if len(symptom_cols) > 3:
        for col in symptom_cols[3:]:
            symptom4_options.extend(filtered_df[col].dropna().unique())
    symptom4_options = sorted(list(set(symptom4_options)))
    symptom4 = st.selectbox("Symptom 4", [None] + symptom4_options)
    if symptom4:
        selected_symptoms.append(symptom4)
else:
    symptom4 = None

# ----------------------------
# Predict button
# ----------------------------
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
