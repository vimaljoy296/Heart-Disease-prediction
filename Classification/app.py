# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv("C:/Supervised_Learning/Classification/heart.csv")

# Title of the app
st.title("Heart Disease Prediction Web App")

# Sidebar for user input to select model
st.sidebar.header("Select Model and Input Data")
model_option = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Random Forest", "K-Nearest Neighbors"])

# Create the feature and target arrays
X = data.drop(columns=['target'])
y = data['target']

# Train-test split and normalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# User Input Fields in the sidebar for prediction (Example: Age, Sex, Blood Pressure, etc.)
st.sidebar.header("Input Features")

def user_input():
    # Input for all features (or adjust based on dataset columns)
    age = st.sidebar.slider("Age", 20, 80, 50)
    sex = st.sidebar.selectbox("Sex", [0, 1])  # Assuming 0 = Female, 1 = Male
    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])  # Adjust according to your dataset
    trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.sidebar.slider("Serum Cholestoral", 100, 600, 250)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])  # 0 = False, 1 = True
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 200, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])  # 0 = No, 1 = Yes
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

    # Create a dictionary with the user's input values
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    # Convert the dictionary into a DataFrame
    user_df = pd.DataFrame(user_data, index=[0])
    return user_df

# Get the user's input
input_data = user_input()

# Make the input features the same scale as the training data
scaled_input = scaler.transform(input_data)

# Display the input data
st.subheader("User Input Data:")
st.write(input_data)

# Model Training and Evaluation
if model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=200)
elif model_option == "Random Forest":
    model = RandomForestClassifier(random_state=42)
else:
    model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

# Make predictions using the user's input
prediction = model.predict(scaled_input)

# Display Prediction Result
if prediction == 1:
    st.subheader("Prediction: Heart Disease Detected")
else:
    st.subheader("Prediction: No Heart Disease Detected")

# Display Accuracy and Evaluation Metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader(f"Accuracy of {model_option}: {accuracy:.2f}")
st.subheader("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.subheader("Classification Report:")
st.write(classification_report(y_test, y_pred))

# Run the app: Use the following command in terminal:
# streamlit run app.py
