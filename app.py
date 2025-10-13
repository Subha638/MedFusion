import streamlit as st
import pandas as pd

st.set_page_config(page_title="Disease Prediction Dashboard", layout="centered")
st.title("🩺 Disease Prediction Dashboard")

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("symtoms_df.csv")  # your actual CSV file
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        required_cols = ['Disease', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"CSV must contain '{col}' column.")
                return None
        return df
    except FileNotFoundError:
        st.error("File 'symtoms_df.csv' not found. Please upload it to the app folder.")
        return None

symtoms_df = load_data()

if symtoms_df is not None:
    # ----------------- SYMPTOM LIST -----------------
    symptom_cols = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
    all_symptoms = pd.unique(symtoms_df[symptom_cols].values.ravel())  # flatten all symptom columns
    all_symptoms = [s for s in all_symptoms if pd.notna(s)]  # remove NaN
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
        # Remaining symptoms
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

else:
    st.warning("Please make sure 'symtoms_df.csv' is in the app folder.")
