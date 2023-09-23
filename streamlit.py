import streamlit as st
import pickle
with open('model_1.pkl', 'rb') as f:
    bagging_classifier = pickle.load(f)
# Load the saved Bagging Classifier model
# Load the saved Bagging Classifier model

# Define the Streamlit app
st.title("URL Phishing Detection")

# User input for URL features
st.write("Enter the following features:")
length_url = st.number_input("Length of URL")
length_hostname = st.number_input("Length of Hostname")
# Add more feature inputs as needed based on your dataset

if st.button("Check"):
    # Create a feature vector based on the user inputs
    feature_vector = [[length_url, length_hostname]]  # Add other features as needed
    
    # Make a prediction using the Bagging Classifier
    prediction = bagging_classifier.predict(feature_vector)

    # Define the labels for prediction
    labels = {0: 'Legitimate', 1: 'Phishing'}

    # Display the prediction result
    result = labels[prediction[0]]
    st.write(f"The entered URL is: {result}")
