import streamlit as st
import pandas as pd

st.set_page_config(page_title="Disease Prediction Dashboard", layout="centered")

st.title("🩺 Disease Prediction Dashboard")

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    symptoms_df = pd.read_csv("symtoms_df.csv")  # your CSV with symptoms & diseases
    return symptoms_df

symptoms_df = load_data()

# Get list of all symptoms
all_symptoms = sorted(symptoms_df['Symptom'].unique())

# ----------------- SELECT SYMPTOMS -----------------
st.subheader("Select Symptoms")

# Function to filter symptoms based on previous selections
def filter_symptoms(selected_symptoms):
    if not selected_symptoms:
        return all_symptoms
    # Filter dataset where all previous symptoms are present
    filtered_df = symptoms_df
    for i, s in enumerate(selected_symptoms):
        filtered_df = filtered_df[filtered_df['Symptom'] == s]
    # Return unique symptoms that are next to selected ones
    next_symptoms = symptoms_df['Symptom'].unique()
    for s in selected_symptoms:
        next_symptoms = [x for x in next_symptoms if x != s]
    return sorted(next_symptoms)

# Symptom 1
symptom1 = st.selectbox("Symptom 1", ["Select"] + all_symptoms)

# Symptom 2
if symptom1 != "Select":
    symptom2_options = filter_symptoms([symptom1])
    symptom2 = st.selectbox("Symptom 2", ["Select"] + symptom2_options)
else:
    symptom2 = "Select"

# Symptom 3
if symptom2 != "Select":
    symptom3_options = filter_symptoms([symptom1, symptom2])
    symptom3 = st.selectbox("Symptom 3", ["Select"] + symptom3_options)
else:
    symptom3 = "Select"

# Symptom 4
if symptom3 != "Select":
    symptom4_options = filter_symptoms([symptom1, symptom2, symptom3])
    symptom4 = st.selectbox("Symptom 4", ["Select"] + symptom4_options)
else:
    symptom4 = "Select"

# ----------------- PREDICTION -----------------
if st.button("Predict Disease"):
    selected = [s for s in [symptom1, symptom2, symptom3, symptom4] if s != "Select"]
    st.write("You selected:", selected)
    
    # Filter dataset for matching diseases
    diseases = symptoms_df
    for s in selected:
        diseases = diseases[diseases['Symptom'] == s]
    
    # Show possible diseases
    possible_diseases = diseases['Disease'].unique() if not diseases.empty else ["No disease found"]
    st.success(f"Possible diseases: {', '.join(possible_diseases)}")
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Disease Prediction Dashboard", layout="centered")

st.title("🩺 Disease Prediction Dashboard")

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    symptoms_df = pd.read_csv("symptoms_df.csv")  # your CSV with symptoms & diseases
    return symptoms_df

symptoms_df = load_data()

# Get list of all symptoms
all_symptoms = sorted(symptoms_df['Symptom'].unique())

# ----------------- SELECT SYMPTOMS -----------------
st.subheader("Select Symptoms")

# Function to filter symptoms based on previous selections
def filter_symptoms(selected_symptoms):
    if not selected_symptoms:
        return all_symptoms
    # Filter dataset where all previous symptoms are present
    filtered_df = symptoms_df
    for i, s in enumerate(selected_symptoms):
        filtered_df = filtered_df[filtered_df['Symptom'] == s]
    # Return unique symptoms that are next to selected ones
    next_symptoms = symptoms_df['Symptom'].unique()
    for s in selected_symptoms:
        next_symptoms = [x for x in next_symptoms if x != s]
    return sorted(next_symptoms)

# Symptom 1
symptom1 = st.selectbox("Symptom 1", ["Select"] + all_symptoms)

# Symptom 2
if symptom1 != "Select":
    symptom2_options = filter_symptoms([symptom1])
    symptom2 = st.selectbox("Symptom 2", ["Select"] + symptom2_options)
else:
    symptom2 = "Select"

# Symptom 3
if symptom2 != "Select":
    symptom3_options = filter_symptoms([symptom1, symptom2])
    symptom3 = st.selectbox("Symptom 3", ["Select"] + symptom3_options)
else:
    symptom3 = "Select"

# Symptom 4
if symptom3 != "Select":
    symptom4_options = filter_symptoms([symptom1, symptom2, symptom3])
    symptom4 = st.selectbox("Symptom 4", ["Select"] + symptom4_options)
else:
    symptom4 = "Select"

# ----------------- PREDICTION -----------------
if st.button("Predict Disease"):
    selected = [s for s in [symptom1, symptom2, symptom3, symptom4] if s != "Select"]
    st.write("You selected:", selected)
    
    # Filter dataset for matching diseases
    diseases = symptoms_df
    for s in selected:
        diseases = diseases[diseases['Symptom'] == s]
    
    # Show possible diseases
    possible_diseases = diseases['Disease'].unique() if not diseases.empty else ["No disease found"]
    st.success(f"Possible diseases: {', '.join(possible_diseases)}")

