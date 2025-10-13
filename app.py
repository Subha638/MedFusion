import streamlit as st
import pandas as pd

st.set_page_config(page_title="Disease Prediction Dashboard", layout="centered")
st.title("🩺 Disease Prediction Dashboard")

# ----------------- LOAD CSV -----------------
@st.cache_data
def load_csv(file_name):
    try:
        df = pd.read_csv(file_name)
        df.columns = [c.strip().title() for c in df.columns]
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
        return df
    except FileNotFoundError:
        st.warning(f"{file_name} not found in app folder.")
        return None

# Load all CSV files
symtoms_df = load_csv("symtoms_df.csv")  # Symptoms
workout_df = load_csv("workout_df.csv")  # Workout
precautions_df = load_csv("precautions_df.csv")  # Precautions
diets_df = load_csv("diets.csv")  # Diet
medications_df = load_csv("medications.csv")  # Medications

if symtoms_df is not None:
    symptom_cols = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
    all_symptoms = pd.unique(symtoms_df[symptom_cols].values.ravel())
    all_symptoms = sorted([s for s in all_symptoms if pd.notna(s)])

    st.subheader("Select Symptoms")

    def filter_symptoms(selected_symptoms):
        if not selected_symptoms:
            return all_symptoms
        filtered_df = symtoms_df.copy()
        for s in selected_symptoms:
            filtered_df = filtered_df[
                (filtered_df['Symptom_1'] == s) |
                (filtered_df['Symptom_2'] == s) |
                (filtered_df['Symptom_3'] == s) |
                (filtered_df['Symptom_4'] == s)
            ]
        remaining_symptoms = pd.unique(filtered_df[symptom_cols].values.ravel())
        remaining_symptoms = [x for x in remaining_symptoms if pd.notna(x) and x not in selected_symptoms]
        return sorted(remaining_symptoms)

    symptom1 = st.selectbox("Symptom 1", ["Select"] + all_symptoms)
    symptom2 = st.selectbox("Symptom 2", ["Select"] + (filter_symptoms([symptom1]) if symptom1 != "Select" else []))
    symptom3 = st.selectbox("Symptom 3", ["Select"] + (filter_symptoms([symptom1, symptom2]) if symptom2 != "Select" else []))
    symptom4 = st.selectbox("Symptom 4", ["Select"] + (filter_symptoms([symptom1, symptom2, symptom3]) if symptom3 != "Select" else []))

    if st.button("Predict Disease"):
        selected = [s for s in [symptom1, symptom2, symptom3, symptom4] if s != "Select"]
        st.write("You selected:", selected)

        diseases_df = symtoms_df.copy()
        for s in selected:
            diseases_df = diseases_df[
                (diseases_df['Symptom_1'] == s) |
                (diseases_df['Symptom_2'] == s) |
                (diseases_df['Symptom_3'] == s) |
                (diseases_df['Symptom_4'] == s)
            ]

        possible_diseases = diseases_df['Disease'].unique() if not diseases_df.empty else ["No disease found"]
        st.success(f"Possible diseases: {', '.join(possible_diseases)}")

        # ----------------- RECOMMENDATIONS -----------------
        for disease in possible_diseases:
            st.subheader(f"{disease}")

            # Function to safely get recommendation text
            def get_rec_text(df, col):
                if df is not None:
                    rec = df[df['Disease'].str.lower() == disease.lower()]
                    if not rec.empty and col in rec.columns:
                        return rec[col].values[0]
                return "N/A"

            st.markdown(f"**Medications:** {get_rec_text(medications_df, 'Medications')}")
            st.markdown(f"**Precautions:** {get_rec_text(precautions_df, 'Precautions')}")
            st.markdown(f"**Diet:** {get_rec_text(diets_df, 'Diet')}")
            st.markdown(f"**Workout:** {get_rec_text(workout_df, 'Workout')}")

else:
    st.warning("Please make sure all required CSV files are in the app folder.")
