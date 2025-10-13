import streamlit as st
import pandas as pd

st.set_page_config(page_title="Disease Prediction Dashboard", layout="centered")
st.title("🩺 Disease Prediction Dashboard")

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_csv(file_name):
    try:
        df = pd.read_csv(file_name)
        df.columns = [c.strip().title() for c in df.columns]
        return df
    except FileNotFoundError:
        return None

symtoms_df = load_csv("symtoms_df.csv")  # Disease & Symptoms
workout_df = load_csv("workout_df.csv")  # Disease & Workout
precautions_df = load_csv("precautions_df.csv")  # Disease & Precautions
diets_df = load_csv("diets.csv")  # Disease & Diet
description_df = load_csv("description.csv")  # Disease & Description

if symtoms_df is not None:
    # ----------------- SYMPTOM LIST -----------------
    symptom_cols = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
    all_symptoms = pd.unique(symtoms_df[symptom_cols].values.ravel())
    all_symptoms = [s for s in all_symptoms if pd.notna(s)]
    all_symptoms = sorted(all_symptoms)

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
    symptom2 = st.selectbox(
        "Symptom 2",
        ["Select"] + (filter_symptoms([symptom1]) if symptom1 != "Select" else [])
    )
    symptom3 = st.selectbox(
        "Symptom 3",
        ["Select"] + (filter_symptoms([symptom1, symptom2]) if symptom2 != "Select" else [])
    )
    symptom4 = st.selectbox(
        "Symptom 4",
        ["Select"] + (filter_symptoms([symptom1, symptom2, symptom3]) if symptom3 != "Select" else [])
    )

    # ----------------- PREDICTION -----------------
    if st.button("Predict Disease"):
        selected = [s for s in [symptom1, symptom2, symptom3, symptom4] if s != "Select"]
        st.write("You selected:", selected)

        # Filter diseases containing any selected symptom
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

            # Description
            if description_df is not None:
                desc = description_df[description_df['Disease'].str.strip().str.lower() == disease.strip().lower()]
                if not desc.empty:
                    st.markdown(f"**Description:** {desc['Description'].values[0]}")

            # Precautions
            if precautions_df is not None:
                prec = precautions_df[precautions_df['Disease'].str.strip().str.lower() == disease.strip().lower()]
                if not prec.empty:
                    st.markdown(f"**Precautions:** {prec['Precautions'].values[0]}")

            # Diet
            if diets_df is not None:
                diet = diets_df[diets_df['Disease'].str.strip().str.lower() == disease.strip().lower()]
                if not diet.empty:
                    st.markdown(f"**Diet:** {diet['Diet'].values[0]}")

            # Workout
            if workout_df is not None:
                workout = workout_df[workout_df['Disease'].str.strip().str.lower() == disease.strip().lower()]
                if not workout.empty:
                    st.markdown(f"**Workout:** {workout['Workout'].values[0]}")

else:
    st.warning("Please make sure all required CSV files are in the app folder.")
