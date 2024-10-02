import os 
import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
my_secret = st.secrets['OPEN_AI_API_KEY']
client = OpenAI(api_key=my_secret)

# Story generator function
def story_gen(prompt):
    system_prompt = """
    You are a world renowned children's storyteller with 50 years of experience. 
    You will be given a concept to generate a story suitable for ages 5-7 years old.
    """
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=1.3,
        max_tokens=300  # Increased for longer stories
    )
    return response.choices[0].message.content

# Cover prompt generator function
def cover_gen(story):
    system_prompt = """
    You will be given a children's story. Generate a prompt for cover art that is 
    suitable and shows off the story themes. The prompt will be sent to DALL-E 2.
    """
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": story}
        ],
        temperature=1.3,
        max_tokens=1000
    )
    return response.choices[0].message.content

# Image generator function
def image_gen(prompt):
    response = client.images.generate(
        model='dall-e-2',
        prompt=prompt,
        size='256x256',
        n=1,
    )
    return response.data[0].url

# Streamlit UI
st.title("Storybook Generator for Kids")
st.divider()

prompt = st.text_input("Enter your story concept:")

if st.button("Generate Storybook"):
    with st.spinner("Generating your storybook..."):
        story = story_gen(prompt)
        cover_prompt = cover_gen(story)
        image_url = image_gen(cover_prompt)
    st.image(image_url, caption="Story Cover")
    st.write("### Your Story:")
    st.write(story)
    st.write("### Cover Art Prompt:")
    st.write(cover_prompt)