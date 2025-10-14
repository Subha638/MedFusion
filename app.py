# app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------
# Load CSVs
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
# Train model
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
        all_symptoms.update([str(s).strip() for s in symptoms_df[col].dropna().unique()])
    all_symptoms = sorted(list(all_symptoms))

    # Binary feature matrix
    X = pd.DataFrame(0, index=symptoms_df.index, columns=all_symptoms)
    for idx, row in symptoms_df.iterrows():
        for col in symptom_cols:
            sym = str(row[col]).strip()
            if sym in all_symptoms:
                X.at[idx, sym] = 1
    y = symptoms_df[disease_col]

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    return clf, all_symptoms, symptom_cols

# ----------------------------
# Predict disease (always show top 3)
# ----------------------------
def predict_disease(user_symptoms, symptoms_df, clf, all_symptoms, symptom_cols):
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
# Get recommendations
# ----------------------------
def get_recommendations(disease, diets_df, medications_df, precautions_df, workout_df):
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
# Streamlit UI
# ----------------------------
st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>🩺 MedFusion</h1>
<hr>
<p style='text-align: center; font-size:18px; color:#566573;'><em>“Predict, Prevent, and Personalize your Health Care”</em></p>
""", unsafe_allow_html=True)

st.title("Welcome to MedFusion")
st.markdown("
st.markdown("Select symptoms to predict possible disease and get recommendations.")

# Load data and train model
symptoms_df, diets_df, medications_df, precautions_df, workout_df = load_data()
clf, all_symptoms, symptom_cols = train_model(symptoms_df)

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
        predicted_disease, top_3 = predict_disease(selected_symptoms, symptoms_df, clf, all_symptoms, symptom_cols)
        st.success(f"Predicted Disease: {predicted_disease}")

        st.subheader("Top 3 probable diseases:")
        for d, p in top_3:
            st.write(f"{d}: {p:.2f}")

        recommendations = get_recommendations(predicted_disease, diets_df, medications_df, precautions_df, workout_df)
        st.subheader("Diet Recommendations")
        st.write(recommendations['Diet'])
        st.subheader("Medication Recommendations")
        st.write(recommendations['Medications'])
        st.subheader("Precautions")
        st.write(recommendations['Precautions'])
        st.subheader("Workout Tips")
        st.write(recommendations['Workouts'])

