import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------- DATA LOADING ----------
@st.cache_data
def load_data():
    symptoms_df = pd.read_csv("symptoms_df.csv")
    diets_df = pd.read_csv("diets_df.csv")
    medications_df = pd.read_csv("medications_df.csv")
    precautions_df = pd.read_csv("precautions_df.csv")
    workout_df = pd.read_csv("workout_df.csv")
    return symptoms_df, diets_df, medications_df, precautions_df, workout_df

symptoms_df, diets_df, medications_df, precautions_df, workout_df = load_data()

# ---------- MODEL TRAINING ----------
symptom_cols = [col for col in symptoms_df.columns if 'Symptom' in col]
all_symptoms = sorted(list(set(sum([symptoms_df[c].dropna().astype(str).tolist() for c in symptom_cols], []))))

# Binary encoding
X = pd.DataFrame(0, index=symptoms_df.index, columns=all_symptoms)
for idx, row in symptoms_df.iterrows():
    for col in symptom_cols:
        val = str(row[col]).strip()
        if val in all_symptoms:
            X.at[idx, val] = 1

y = symptoms_df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ---------- HELPER FUNCTIONS ----------
def predict_disease(symptoms):
    input_vector = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for s in symptoms:
        s = str(s).strip()
        if s in all_symptoms:
            input_vector[s] = 1
    pred = clf.predict(input_vector)[0]
    probs = clf.predict_proba(input_vector)[0]
    top_3 = sorted(zip(clf.classes_, probs), key=lambda x: x[1], reverse=True)[:3]
    return pred, top_3

def get_recommendations(disease):
    diet = diets_df[diets_df['Disease'] == disease]['Diet'].tolist() or ["No data available"]
    meds = medications_df[medications_df['Disease'] == disease]['Medication'].tolist() or ["No data available"]
    pre = precautions_df[precautions_df['Disease'] == disease].dropna(axis=1).values.flatten().tolist() or ["No data available"]
    work = workout_df[workout_df['disease'] == disease]['workout'].tolist() or ["No data available"]
    return {"Diet": diet, "Medications": meds, "Precautions": pre[:4], "Workouts": work[:4]}

# ---------- CHATBOT-LIKE UI ----------
st.title("💬 Health AI Chatbot")
st.write("Describe your symptoms or select them manually to get predictions and advice.")

chat = st.chat_input("Enter symptoms (comma separated) or click below options:")
if "messages" not in st.session_state:
    st.session_state.messages = []

if chat:
    st.session_state.messages.append({"role": "user", "content": chat})
    symptoms = [s.strip().lower() for s in chat.split(",")]
    pred, top3 = predict_disease(symptoms)
    rec = get_recommendations(pred)

    msg = f"🎯 **Predicted Disease:** {pred}\n\n"
    msg += "**Top Predictions:**\n" + "\n".join([f"- {d}: {round(p,2)}" for d,p in top3])
    st.session_state.messages.append({"role": "assistant", "content": msg})

# display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Optionally show dropdown selection
with st.expander("Or select symptoms manually"):
    s1 = st.selectbox("Symptom 1", ["None"] + all_symptoms)
    s2 = st.selectbox("Symptom 2", ["None"] + all_symptoms)
    s3 = st.selectbox("Symptom 3", ["None"] + all_symptoms)
    s4 = st.selectbox("Symptom 4", ["None"] + all_symptoms)
    if st.button("Predict from Dropdown"):
        syms = [s for s in [s1,s2,s3,s4] if s != "None"]
        pred, top3 = predict_disease(syms)
        st.success(f"🎯 Disease: {pred}")
        st.bar_chart(pd.DataFrame(top3, columns=["Disease","Probability"]).set_index("Disease"))
        st.json(get_recommendations(pred))
