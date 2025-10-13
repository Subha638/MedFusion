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
        # Strip spaces for string columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
        return df
    except FileNotFoundError:
        st.warning(f"{file_name} not found in app folder.")
        return None

# Load all CSV files
symtoms_df = load_csv("symtoms_df.csv")       # Symptoms
medications_df = load_csv("medications.csv") # Medications
precautions_df = load_csv("precautions_df.csv")
diets_df = load_csv("diets.csv")
workout_df = load_csv("workout_df.csv")

# ----------------- NORMALIZE DISEASE NAMES -----------------
def normalize_disease(df):
    if df is not None and 'Disease' in df.columns:
        df['Disease'] = df['Disease'].str.lower().str.strip()
    return df

symtoms_df = normalize_disease(symtoms_df)
medications_df = normalize_disease(medications_df)
precautions_df = normalize_disease(precautions_df)
diets_df = normalize_disease(diets_df)
workout_df = normalize_disease(workout_df)

if symtoms_df is not None:
    # ----------------- SYMPTOM LIST -----------------
    symptom_cols = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
    all_symptoms = pd.unique(symtoms_df[symptom_cols].values.ravel())
    all_symptoms = sorted([s for s in all_symptoms if pd.notna(s)])

    st.subheader("Select Symptoms")

    # ----------------- FILTER SYMPTOMS -----------------
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

    # ----------------- SYMPTOM SELECTION -----------------
    symptom1 = st.selectbox("Symptom 1", ["Select"] + all_symptoms)
    symptom2 = st.selectbox("Symptom 2", ["Select"] + (filter_symptoms([symptom1]) if symptom1 != "Select" else []))
    symptom3 = st.selectbox("Symptom 3", ["Select"] + (filter_symptoms([symptom1, symptom2]) if symptom2 != "Select" else []))
    symptom4 = st.selectbox("Symptom 4", ["Select"] + (filter_symptoms([symptom1, symptom2, symptom3]) if symptom3 != "Select" else []))

    # ----------------- PREDICTION -----------------
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
        st.success(f"Possible diseases: {', '.join([d.title() for d in possible_diseases])}")

        # ----------------- RECOMMENDATIONS -----------------
        for disease in possible_diseases:
            st.subheader(f"{disease.title()}")

            def get_rec_list(df, col):
                """Return recommendations as list if possible, else N/A"""
                if df is not None:
                    rec = df[df['Disease'] == disease]
                    if not rec.empty and col in rec.columns:
                        # Split by comma if multiple items
                        val = rec[col].values[0]
                        if ',' in val:
                            return [i.strip() for i in val.split(',')]
                        else:
                            return [val]
                return ["N/A"]

            medications = get_rec_list(medications_df, 'Medications')
            precautions = get_rec_list(precautions_df, 'Precautions')
            diet = get_rec_list(diets_df, 'Diet')
            workout = get_rec_list(workout_df, 'Workout')

            st.markdown(f"**Medications:** {', '.join(medications)}")
            st.markdown(f"**Precautions:** {', '.join(precautions)}")
            st.markdown(f"**Diet:** {', '.join(diet)}")
            st.markdown(f"**Workout:** {', '.join(workout)}")

else:
    st.warning("Please make sure all required CSV files are in the app folder.")
