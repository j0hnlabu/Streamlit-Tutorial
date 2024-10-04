import streamlit as st
import openai as OpenAI
import os

# Set your OpenAI API key
OpenAI.api_key = os.getenv("OPEN_AI_API_KEY") 

# Title of the app
st.title("Generative AI with OpenAI")

# Add a header
st.header("Generate Text with OpenAI's GPT-3")

# Input text for user prompt
user_input = st.text_input("Enter your prompt:")

# Temperature slider for creativity
temperature = st.slider('Temperature (creativity level)', 0.0, 1.0, 0.7)

# Generate AI response when user submits prompt
if st.button("Generate"):
    if user_input:
        st.write("Generating response...")

        try:
            # Call OpenAI API to generate text
            response = OpenAI.Completion.create(
                engine="gpt-3.5-turbo",  
                prompt=user_input,
                max_tokens=150,
                temperature=temperature,
            )

            # Display the generated text
            generated_text = response.choices[0].text.strip()
            st.subheader("Generated Text:")
            st.write(generated_text)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt.")


