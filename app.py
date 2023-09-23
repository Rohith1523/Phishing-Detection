import streamlit as st
import pickle
import numpy as np
from urllib.parse import urlparse  # For URL parsing

# Load the saved Bagging Classifier model
with open('model_1.pkl', 'rb') as f:
    bagging_classifier = pickle.load(f)

# Define the Streamlit app
st.title("URL Phishing Detection")

# User input for URL
user_url = st.text_input("Enter a URL")

if st.button("Check"):
    if not user_url:
        st.warning("Please enter a URL.")
    else:
        # Feature extraction from the URL
        parsed_url = urlparse(user_url)
        length_url = len(user_url)
        length_hostname = len(parsed_url.hostname)

        # You should add code here to extract and preprocess the remaining 86 features
        # from your dataset and append them to the feature_vector.
        # For example:
        # feature_vector = [[length_url, length_hostname, feature3, feature4, ...]]

        # Ensure the feature vector has 88 features
        while len(feature_vector[0]) < 88:
            feature_vector[0].append(0.0)  # You should replace 0.0 with the actual values

        # Convert the feature vector to a NumPy array
        feature_vector = np.array(feature_vector)

        # Make a prediction using the Bagging Classifier
        prediction = bagging_classifier.predict(feature_vector)

        # Define the labels for prediction
        labels = {0: 'Legitimate', 1: 'Phishing'}

        # Display the prediction result
        result = labels[prediction[0]]
        st.write(f"The entered URL is: {result}")
