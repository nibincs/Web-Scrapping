import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Title
st.title("Email Spam Classifier")

# User input
message = st.text_area("Enter Email or Message")

# Prediction button
if st.button("Predict"):

    # Convert text to vector
    data = cv.transform([message]).toarray()

    # Prediction
    prediction = model.predict(data)

    # Output
    if prediction[0] == 1:
        st.error("Spam Email")
    else:
        st.success("Not Spam (Ham)")