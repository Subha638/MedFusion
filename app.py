import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Disease Prediction App", layout="wide")
st.title("🩺 Disease Prediction and Recommendation System")

# ----------------------------
# LOAD MODEL AND DATA
# ----------------------------
@st.cache_resource
def load_model():
    clf = joblib.load("MLFinPro.joblib")
    return clf

@st.cache_data
def load_data():
    symptoms_df = pd.read_csv("symptoms_df.csv")
    diets_df = pd.read_csv("diets.csv")
    medications_df = pd.read_csv("medications.csv")
    precautions_df = pd.read_csv("precautions_df.csv")
    workout_df = pd.read_csv("workout_df.csv")
    description_df = pd.read_csv("description.csv")
    return symptoms_df, diets_df, medications_df, precautions_df, workout_df, description_df

clf = load_model()
symptoms_df, diets_df, medications_df, precautions_df, workout_df, description_df = load_data()

# ----------------------------
# PREPARE SYMPTOMS DATA
# ----------------------------
symptom_cols = [c for c in symptoms_df.columns if "Symptom" in c]
all_symptoms = sorted(list(set(
    s.strip() for col in symptom_cols for s in symptoms_df[col].dropna().astype(str)
)))

# ----------------------------
# PREDICT DISEASE FUNCTION
# ----------------------------
def predict_disease(symptoms):
    input_vector = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for s in symptoms:
        s = s.strip()
        if s in all_symptoms:
            input_vector[s] = 1

    prediction = clf.predict(input_vector)[0]
    probs = clf.predict_proba(input_vector)[0]
    top_3 = sorted(zip(clf.classes_, probs), key=lambda x: x[1], reverse=True)[:3]
    return prediction, top_3

# ----------------------------
# GET RECOMMENDATIONS
# ----------------------------
def get_recommendations(disease):
    disease = disease.strip()

    diet = diets_df.loc[diets_df["Disease"] == disease, "Diet"]
    diet = eval(diet.iloc[0]) if not diet.empty else ["No data"]

    meds = medications_df.loc[medications_df["Disease"] == disease, "Medication"]
    meds = eval(meds.iloc[0]) if not meds.empty else ["No data"]

    prec = precautions_df.loc[precautions_df["Disease"] == disease]
    if not prec.empty:
        prec_cols = [c for c in prec.columns if "Precaution" in c]
        precautions = [str(x) for x in prec[prec_cols].values.flatten() if pd.notna(x)]
    else:
        precautions = ["No data"]

    workout = workout_df.loc[workout_df["disease"] == disease, "workout"]
    workout = workout.tolist() if not workout.empty else ["No data"]

    desc = description_df.loc[description_df["Disease"] == disease, "Description"]
    desc = desc.iloc[0] if not desc.empty else "No description available"

    return {
        "Diet": diet,
        "Medications": meds,
        "Precautions": precautions,
        "Workouts": workout,
        "Description": desc
    }

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.sidebar.header("🧩 Select Symptoms")
symptom1 = st.sidebar.selectbox("Symptom 1", ["None"] + all_symptoms)
symptom2 = st.sidebar.selectbox("Symptom 2", ["None"] + all_symptoms)
symptom3 = st.sidebar.selectbox("Symptom 3", ["None"] + all_symptoms)
symptom4 = st.sidebar.selectbox("Symptom 4", ["None"] + all_symptoms)

if st.sidebar.button("🔍 Predict Disease"):
    selected = [s for s in [symptom1, symptom2, symptom3, symptom4] if s != "None"]
    if not selected:
        st.warning("Please select at least one symptom.")
    else:
        with st.spinner("Predicting disease..."):
            disease, top3 = predict_disease(selected)
            st.success(f"**Predicted Disease:** {disease}")
            
            st.subheader("Top 3 Possible Diseases")
            for d, p in top3:
                st.write(f"🔹 {d}: {p:.2f}")

            rec = get_recommendations(disease)
            st.markdown(f"### 🩸 Description")
            st.write(rec['Description'])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🍽️ Diet")
                st.write(rec["Diet"])
                st.markdown("### 💊 Medications")
                st.write(rec["Medications"])
            with col2:
                st.markdown("### ⚠️ Precautions")
                st.write(rec["Precautions"])
                st.markdown("### 🏋️ Workouts")
                st.write(rec["Workouts"])
else:
    st.info("Select symptoms and click **Predict Disease** to start.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ by Subhalaxmi Behera")
