# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Disease Prediction App", layout="wide")

# --------------------------
# Load Data
# --------------------------
@st.cache_data
def load_data():
    symptoms_df = pd.read_csv("symtoms_df.csv")
    diets_df = pd.read_csv("diets.csv")
    medications_df = pd.read_csv("medications.csv")
    precautions_df = pd.read_csv("precautions.csv")
    workout_df = pd.read_csv("workout_df.csv")
    return symptoms_df, diets_df, medications_df, precautions_df, workout_df

symptoms_df, diets_df, medications_df, precautions_df, workout_df = load_data()

# --------------------------
# Preprocess Data
# --------------------------
symptom_cols = [col for col in symptoms_df.columns if 'Symptom' in col]
disease_col = 'Disease'

# Drop rows where Disease is missing
symptoms_df = symptoms_df.dropna(subset=[disease_col])
symptoms_df[disease_col] = symptoms_df[disease_col].astype(str).str.strip()

# Filter classes with at least 2 samples
class_counts = symptoms_df[disease_col].value_counts()
valid_mask = symptoms_df[disease_col].isin(class_counts[class_counts >= 2].index)
symptoms_df = symptoms_df[valid_mask]

# Collect unique symptoms
all_symptoms = sorted({str(s).strip() for col in symptom_cols for s in symptoms_df[col].dropna() if str(s).strip()})

# Create binary feature matrix
X = pd.DataFrame(0, index=symptoms_df.index, columns=all_symptoms)
for idx, row in symptoms_df.iterrows():
    for col in symptom_cols:
        sym = str(row[col]).strip()
        if sym in all_symptoms:
            X.at[idx, sym] = 1
y = symptoms_df[disease_col]

# Train Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# --------------------------
# Prediction Functions
# --------------------------
def predict_disease(selected_symptoms):
    input_vec = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for s in selected_symptoms:
        if s in all_symptoms:
            input_vec[s] = 1
    pred = clf.predict(input_vec)[0]
    probs = clf.predict_proba(input_vec)[0]
    top_3 = sorted(zip(clf.classes_, probs), key=lambda x: x[1], reverse=True)[:3]
    return pred, top_3

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
        prec_cols = [c for c in prec_rows.columns if 'Precaution' in c]
        precautions = [str(p).strip() for p in prec_rows[prec_cols].values.flatten() if str(p).strip()]
    else:
        precautions = ["No data available"]

    # Workouts
    workout_rows = workout_df[workout_df['disease'] == disease]
    workouts = [str(w).strip() for w in workout_rows['workout'].tolist()] if not workout_rows.empty else ["No data available"]

    return {'Diet': diet, 'Medications': meds, 'Precautions': precautions[:4], 'Workouts': workouts[:5]}

# --------------------------
# Streamlit UI
# --------------------------
st.title("🩺 Disease Prediction App")

# Cascading symptom dropdowns
symptom1 = st.selectbox("Select Symptom 1", [None]+all_symptoms)
symptom2 = st.selectbox("Select Symptom 2", [None], disabled=True)
symptom3 = st.selectbox("Select Symptom 3", [None], disabled=True)
symptom4 = st.selectbox("Select Symptom 4", [None], disabled=True)

def update_options():
    global symptom2, symptom3, symptom4
    if symptom1:
        # Diseases with symptom1
        rows = X[X[symptom1]==1].index
        co_symptoms = [s for s in all_symptoms if s!=symptom1 and X.loc[rows,s].sum()>0]
        symptom2.options = [None]+co_symptoms
        symptom2.disabled = False
    else:
        symptom2.options = [None]
        symptom2.disabled = True

update_options()

if st.button("Predict Disease"):
    selected = [symptom1, symptom2, symptom3, symptom4]
    selected = [s for s in selected if s]
    if not selected:
        st.warning("Please select at least one symptom!")
    else:
        pred, top_3 = predict_disease(selected)
        st.subheader(f"Predicted Disease: {pred}")
        st.write("Top 3 Probable Diseases:")
        for d, p in top_3:
            st.write(f"- {d}: {p:.2f}")

        rec = get_recommendations(pred)
        st.subheader("Diet Recommendations")
        st.write(rec['Diet'])
        st.subheader("Medication Recommendations")
        st.write(rec['Medications'])
        st.subheader("Precautions")
        st.write(rec['Precautions'])
        st.subheader("Workouts")
        st.write(rec['Workouts'])

        # Optional bar chart
        fig, ax = plt.subplots()
        ax.barh([d for d,_ in top_3], [p for _,p in top_3])
        ax.set_xlabel("Probability")
        ax.set_title("Top 3 Disease Probabilities")
        st.pyplot(fig)
