🧠 AI Powered Task Management System

An AI-based Task Management System that intelligently predicts the priority level of a task using Machine Learning and Natural Language Processing (NLP).
This helps teams focus on critical tasks and manage workloads efficiently.


🚀 Features

Predict task priority using AI

Clean and user-friendly Streamlit interface

Uses TF-IDF for text vectorization

Encodes categorical values with Label Encoding

Trained on a synthetic dataset generated using Faker

🧠 Technologies Used

Python

Streamlit

Scikit-learn

Pandas & NumPy

NLTK / TF-IDF Vectorizer

Pickle (for model saving)

⚙️ How It Works

1)A synthetic dataset is generated using Faker.

2)Task descriptions are vectorized using TF-IDF.

3)Categorical columns (like category, status) are label encoded.

4)ML models such as SVC,Naive Bayes and Random Forest are trained.

5)The best model is saved and used in the Streamlit web app for real-time predictions.

📂 Folder Structure
📦 ai-powered-task-management-system
├── app.py                     # Streamlit main app
├── task_classifier_model.pkl  # Trained ML model
├── vectorizer.pkl             # TF-IDF vectorizer
├── le_category.pkl            # Label encoder for category
├── le_status.pkl              # Label encoder for status
├── scaler.pkl                 # Scaler for numeric features
├── synthetic_dataset.csv      # Generated dataset
├── requirements.txt           # Required libraries
└── README.md                  # Project documentation