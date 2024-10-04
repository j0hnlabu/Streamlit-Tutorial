import os
import json
import PyPDF2
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import matplotlib.pyplot as plt
from PIL import Image
import io

# Initialize OpenAI client
my_secret = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=my_secret)

# Configure Google AI
my_secret = os.environ['GOOGLE_API_KEY']
genai.configure(api_key=my_secret)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

st.markdown("""
<style>
    .stApp {
        background-color: #664229;
    }

      .stTab:hover {
        background-color: #FFFB3;  
      }
      .stTab.stTab-active {
          background-color: #009FB7;  
          color: white;
      }
    
    .stButton>button {
        background-color: #DCAB6B;
        color: white;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {

        background-color: #0D1317;
    }
    
    .stHeader {
        background-color: #009FB7;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def get_ev_info(model, image=None):
    system_prompt = """
    You are an automotive expert specializing in electric vehicles (EVs). 
    Provide detailed information about the requested EV model, including:
    1. Engine performance (in hp or kW)
    2. Battery capacity (in kWh)
    3. Energy efficiency (in km/kWh)
    4. Estimated cost per kWh in Malaysian Ringgit (MYR)

    Present the information in a structured format, with each piece of information on a new line.
    For cost per kWh in MYR and energy efficiency in km/kWh, provide an exact value. Do not provide answers in range.
    If there is no estimated cost per kWh in MYR, provided an average of cost per kWh in MYR. Apply the same concept to energy efficiency (in km/kWh).

    """

    user_prompt = f"Provide information for the electric vehicle model: {model}"

    if image:
        image_prompt = "Here's an image of the vehicle. Use it to provide any additional relevant information."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + "\n" + image_prompt},
            {"role": "user", "content": {"type": "image_url", "image_url": image}}
        ]
        response = client.chat.completions.create(
            model='gpt-4-vision-preview',
            messages=messages,
            max_tokens=300
        )
    else:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

    return response.choices[0].message.content

def parse_ev_info(info_text):
  lines = info_text.split('\n')
  parsed_info = {}
  for line in lines:
      if ':' in line:
          key, value = line.split(':', 1)
          parsed_info[key.strip()] = value.strip()

  # Try to extract energy efficiency, default to None if not found or not parseable
  try:
      energy_efficiency = float(parsed_info.get('Energy efficiency', '0').split()[0])
  except ValueError:
      energy_efficiency = None

  parsed_info['Energy efficiency'] = energy_efficiency
  return parsed_info

import pandas as pd
import plotly.express as px

def calculate_savings(km_per_kwh, cost_per_kwh, distance_per_day, years=5):
    km_per_year = distance_per_day * 365

    # Petrol calculations
    avg_km_per_liter = 12  # Assumption for an average car
    liters_per_year = km_per_year / avg_km_per_liter
    petrol_cost_per_liter = 2.05  # MYR
    petrol_cost_per_year = liters_per_year * petrol_cost_per_liter

    # Electricity calculations
    kwh_per_year = km_per_year / km_per_kwh
    electricity_cost_per_year = kwh_per_year * cost_per_kwh

    # Create DataFrame for multiple years
    data = []
    for year in range(1, years + 1):
        data.append({
            'Year': year,
            'Petrol Cost': petrol_cost_per_year * year,
            'Electricity Cost': electricity_cost_per_year * year,
            'Savings': (petrol_cost_per_year - electricity_cost_per_year) * year
        })

    return pd.DataFrame(data)

def create_savings_chart(df):
    fig = px.bar(df, x='Year', y=['Petrol Cost', 'Electricity Cost'],
                 title='Comparison of Petrol vs Electricity Costs Over Years',
                 labels={'value': 'Cost (MYR)', 'variable': 'Type'},
                 barmode='group',
                 color_discrete_map={'Petrol Cost': 'red', 'Electricity Cost': 'blue'})

    # Add text annotations for savings
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row['Year'],
            y=max(row['Petrol Cost'], row['Electricity Cost']),
            text=f"Savings: RM {row['Savings']:.2f}",
            showarrow=False,
            yshift=10
        )

    return fig

def extract_text_from_pdf(pdf_file):
  text = ""
  try:
      pdf_reader = PyPDF2.PdfReader(pdf_file)
      for page in pdf_reader.pages:
          text += page.extract_text() + "\n"
  except Exception as e:
      st.error(f"Error reading PDF: {str(e)}. Please ensure the PDF is not encrypted and try again.")
      return None
  return text

# Import Google AI client (Google Cloud Natural Language API, for example)
import os
import re
import google.generativeai as genai

# Configure the genai library
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

def extract_qa_pairs(text):
    # Attempt to find question-answer pairs in the text
    qa_pairs = {}
    lines = text.split('\n')
    current_question = None
    current_answer = []

    for line in lines:
        if re.match(r'^\d+\.?\s*', line):  # Looks like a numbered question
            if current_question:
                qa_pairs[current_question] = ' '.join(current_answer).strip()
            current_question = line.strip()
            current_answer = []
        elif current_question:
            current_answer.append(line.strip())

    if current_question:
        qa_pairs[current_question] = ' '.join(current_answer).strip()

    return qa_pairs

def analyze_manual(manual_text):
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
    You are an AI assistant specializing in analyzing EV owner's manuals. 
    Based on the provided manual text, generate 5 common questions that users might ask about the vehicle.
    Also, provide brief answers to these questions based on the information in the manual.
    Format your response as follows:

    1. [Question 1]
    [Answer 1]

    2. [Question 2]
    [Answer 2]

    ... and so on for 5 questions and answers.

    Manual text:
    {manual_text[:4000]}
    """

    response = model.generate_content(prompt)

    try:
        # First, attempt to parse as JSON
        qa_pairs = json.loads(response.text)
    except json.JSONDecodeError:
        # If JSON parsing fails, attempt to extract Q&A pairs from text
        qa_pairs = extract_qa_pairs(response.text)

    if not qa_pairs:
        return {"Error": "Failed to extract question-answer pairs from the model's response. Please try again."}

    return qa_pairs

def answer_question(manual_text, question):
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
    You are an AI assistant specializing in answering questions about EVs based on their owner's manuals. 
    Provide a concise and accurate answer to the user's question using the information from the manual.
    If the information is not available in the manual, politely state that and provide a general response if possible.

    Manual: {manual_text[:4000]}

    Question: {question}
    """

    response = model.generate_content(prompt)
    return response.text

# Streamlit UI

st.title("🚗 EVision")


tab1, tab2, tab3 = st.tabs(["📚 EV Information", "💰 Fuel Savings Calculator", "🖥 Manual Analysis"])

with tab1:
    st.header("EV Information")
    model = st.text_input("Search for an EV Model:")

    if st.button("Get EV Information"):
        with st.spinner("Fetching EV information..."):
            ev_info_text = get_ev_info(model)
            ev_info = parse_ev_info(ev_info_text)

        st.write(f"### {model} Information:")
        ev_info_df = pd.DataFrame(list(ev_info.items()), columns=['Aspect', 'Details'])
        st.table(ev_info_df)

with tab2:
    st.header("EV Fuel Saving Calculator")

    km_per_kwh = st.number_input("Enter energy efficiency (KM/kWh):", min_value=0.1, value=6.0, step=0.1)
    cost_per_kwh = st.number_input("Enter electricity cost (RM/kWh):", min_value=0.01, value=0.571, step=0.001)
    distance_per_day = st.number_input("Enter distance traveled per day (km):", min_value=0.1, value=50.0, step=0.1)
    years = st.slider("Number of years for projection:", min_value=1, max_value=10, value=5)

    if st.button("Calculate Savings"):
        savings_df = calculate_savings(km_per_kwh, cost_per_kwh, distance_per_day, years)

        st.subheader(f"Savings Over {years} Years")
        st.dataframe(savings_df)

        st.subheader("Savings Comparison Chart")
        fig = create_savings_chart(savings_df)
        st.plotly_chart(fig)



with tab3:
    st.header("Owner's Manual Analysis")
    uploaded_file = st.file_uploader("Upload your EV owner's manual (PDF)", type="pdf")

    if uploaded_file is not None:
        # Generate a unique key for the current file
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"

        # Check if this is a new file
        if "current_file_key" not in st.session_state or st.session_state.current_file_key != file_key:
            st.session_state.current_file_key = file_key
            # Clear previous analysis results
            if "manual_text" in st.session_state:
                del st.session_state.manual_text
            if "qa_pairs" in st.session_state:
                del st.session_state.qa_pairs

        with st.spinner("Analyzing the manual..."):
            if "manual_text" not in st.session_state:
                manual_text = extract_text_from_pdf(uploaded_file)
                if manual_text is not None:
                    st.session_state.manual_text = manual_text
                else:
                    st.error("Failed to extract text from the PDF. Please try a different file or check if the PDF is encrypted.")
                    st.stop()

            if "qa_pairs" not in st.session_state:
                st.session_state.qa_pairs = analyze_manual(st.session_state.manual_text)

            if "Error" in st.session_state.qa_pairs:
                st.error(st.session_state.qa_pairs["Error"])
            else:
                # Display common questions
                st.subheader("Common Questions")
                for i, (question, answer) in enumerate(st.session_state.qa_pairs.items()):
                    if st.button(question, key=f"q_{i}"):
                        st.session_state.selected_question = question

                # Display the selected answer
                if 'selected_question' in st.session_state:
                    st.write("Answer:")
                    st.write(st.session_state.qa_pairs[st.session_state.selected_question])

            # Ask a custom question
            st.subheader("Ask Your Own Question")
            user_question = st.text_input("Enter your question about the manual:")

            if st.button("Get Answer", key="custom_question"):
                if user_question:
                    answer = answer_question(st.session_state.manual_text, user_question)
                    st.write("Answer:")
                    st.write(answer)
                else:
                    st.error("Please enter a question.")

st.divider()
st.write("🔍Need more information? Explore the different tabs for EV details, savings calculations, and manual analysis!")