import streamlit as st
from model import train_model, predict, save_model, load_model, label_encoder

# App Title
st.title("BERT Multi-Class Text Classification")

# File Upload or Input Text
input_text = st.text_area("Enter Text for Classification:")

if st.button("Predict"):
    if input_text:
        # Predict using the trained model
        model = load_model()  # Load the trained model (or train it if needed)
        prediction = predict([input_text])

        # Display the result
        predicted_label = label_encoder.inverse_transform(prediction)
        st.write(f"Predicted Category: {predicted_label[0]}")
    else:
        st.error("Please enter some text for prediction.")
