import streamlit as st
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# ----------------- Helper functions -----------------
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    tokens = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

st.title("📊 Task Priority Prediction Dashboard")

# Load saved model, scaler, and vectorizer
try:
    with open("bestmodel.sav", "rb") as f:
        best_model = pickle.load(f)
    with open("scaler1.sav", "rb") as f:
        scaler = pickle.load(f)
    # st.success("✅ Model, Scaler, and TF-IDF Vectorizer loaded successfully.")
except Exception as e:
    st.error(f"Error loading saved model: {e}")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    tokens = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# ----------------- User Input Prediction -----------------
st.header("Predict Priority for a New Task")

task_input = st.text_area("Enter task description")
status = st.selectbox("Status", ['Completed', 'In Progress' ,'Pending'])
category = st.selectbox("Category", ['Deployment' ,'Design' ,'Development', 'Documentation', 'Testing'])
estimated_hours = st.number_input("Estimated Hours", min_value=0.0, step=1.0)

if st.button("Predict Priority"):
    if task_input:
        processed_text = preprocess_text(task_input)

        # Encode categorical values manually (ensure same encoding as training)
        status_map = {"Completed": 0, "In Progress": 1, "Pending": 2}
        category_map = {"Deployment": 0, "Design": 1, "Development": 2, "Documentation": 3, "Testing": 4}
        status_encoded = status_map.get(status, 0)
        category_encoded = category_map.get(category, 0)
        task_input_encoded = len(processed_text.split())

        # Combine all numeric inputs in same order used during model training
        numeric_features = np.array([[estimated_hours, status_encoded, category_encoded,task_input_encoded]])
        numeric_scaled = scaler.transform(numeric_features)

        # Predict
        pred = best_model.predict(numeric_scaled)
        priority_map = {0: "high", 1: "Low", 2: "Medium"}
        pred_label = priority_map.get(int(pred[0]), "Unknown")

        # Display result
        # st.subheader("Prediction Result")
        # st.write(f"🎯 **Predicted Priority:** {pred_label}")
        if pred_label == "High":
            st.error(f"⚠️ Predicted Priority: {pred_label}")
        elif pred_label == "Medium":
            st.warning(f"⚠️ Predicted Priority: {pred_label}")
        else:
            st.success(f"✅ Predicted Priority: {pred_label}")
    else:
        st.error("Please enter a task description.")
