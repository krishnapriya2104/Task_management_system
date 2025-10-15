import streamlit as st
import pickle
from scipy.sparse import hstack

# -----------------------------
# Load Trained Models & Encoders
# -----------------------------
task_model = pickle.load(open('task_classifier_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
le_category = pickle.load(open('le_category.pkl', 'rb'))

priority_model = pickle.load(open('priority_model.pkl', 'rb'))
le_priority = pickle.load(open('le_priority.pkl', 'rb'))

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="AI Task Management System", page_icon="🤖", layout="centered")

st.title("🧠 AI-Powered Task Management System")
st.write("Predict both **Task Category** and **Priority** using AI.")

# -----------------------------
# User Inputs
# -----------------------------
task_description = st.text_area("✍️ Enter Task Description:", placeholder="e.g., test task handle bug")
estimated_hours = st.number_input("⏱️ Estimated Hours:", min_value=1, max_value=100, value=8)

# Predict Button
if st.button("Predict"):
    if task_description.strip() == "":
        st.warning("Please enter a task description.")
    else:
        # ------------------------------------------------------
        # 1️⃣ Predict Category using SVC
        # ------------------------------------------------------
        X_desc = vectorizer.transform([task_description])
        category_pred = task_model.predict(X_desc)
        predicted_category = le_category.inverse_transform(category_pred)[0]

        # ------------------------------------------------------
        # 2️⃣ Predict Priority using Random Forest
        # ------------------------------------------------------
        # Encode category for priority model
        category_encoded = le_category.transform([predicted_category])[0]
        import numpy as np
        X_num = np.array([[estimated_hours, category_encoded]])

        # Combine text + numeric features for priority model
        X_combined = hstack((X_desc, X_num))
        priority_pred = priority_model.predict(X_combined)
        predicted_priority = le_priority.inverse_transform(priority_pred)[0]

        # ------------------------------------------------------
        # Display Results
        # ------------------------------------------------------
        st.subheader("🎯 AI Predictions")
        st.success(f"**Task Category:** {predicted_category}")
        st.success(f"**Task Priority:** {predicted_priority}")


