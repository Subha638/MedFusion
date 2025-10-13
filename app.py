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

# Load CSV files
symtoms_df = load_csv("symtoms_df.csv")
medications_df = load_csv("medications.csv")
precautions_df = load_csv("precautions_df.csv")
diets_df = load_csv("diets.csv")
workout_df = load_csv("workout_df.csv")

# ----------------- NORMALIZE DISEASE NAMES -----------------
def normalize_disease(df):
    if df is not None and 'Disease' in df.columns:
        df['Disease'] = df['Disease'].astype(str).str.strip().str.lower()
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

        # Filter diseases based on selected symptoms
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
        def get_recommendations(df, disease, column_name):
            """Return list of recommendations from CSV or ['N/A']"""
            if df is not None and column_name in df.columns:
                rec = df[df['Disease'] == disease.lower()]
                if not rec.empty:
                    val = rec[column_name].values[0]
                    # Convert comma-separated strings into list
                    if ',' in val:
                        return [v.strip() for v in val.split(',')]
                    else:
                        return [val.strip()]
            return ["N/A"]

        for disease in possible_diseases:
            st.subheader(disease.title())
            medications = get_recommendations(medications_df, disease, 'Medications')
            precautions = get_recommendations(precautions_df, disease, 'Precautions')
            diet = get_recommendations(diets_df, disease, 'Diet')
            workout = get_recommendations(workout_df, disease, 'Workout')

            # Display as bullet points
            st.markdown("**Medications:**")
            for item in medications:
                st.markdown(f"- {item}")
            st.markdown("**Precautions:**")
            for item in precautions:
                st.markdown(f"- {item}")
            st.markdown("**Diet:**")
            for item in diet:
                st.markdown(f"- {item}")
            st.markdown("**Workout:**")
            for item in workout:
                st.markdown(f"- {item}")

else:
    st.warning("Please make sure all required CSV files are in the app folder.")
