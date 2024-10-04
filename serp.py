import os
import PyPDF2
import streamlit as st
from openai import OpenAI
from streamlit_extras.app_logo import add_logo
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container

# Initialize OpenAI client
my_secret = os.environ['OPEN_AI_API_KEY']
client = OpenAI(api_key=my_secret)

st.set_page_config(
    page_title="EV Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #22577A;
    }
    .stButton>button {
        background-color: #38A3A5;
        color: white;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
       
        background-color: #FFFFFF;
    }
    .stHeader {
        background-color: #22577A;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_ev_info(model):
    system_prompt = """
    You are an automotive expert specializing in electric vehicles (EVs). 
    Provide detailed information about the requested EV model, including:
    1. Engine performance (in hp or kW)
    2. Battery capacity (in kWh)
    3. Energy efficiency (in km/kWh)
    4. Estimated cost per kWh in Malaysian Ringgit (MYR)

    Present the information in a structured format, with each piece of information on a new line.
    If the exact information is not available, provide a reasonable estimate based on similar models.
    """

    user_prompt = f"Provide information for the electric vehicle model: {model}"

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

    try:
        energy_efficiency = float(parsed_info.get('Energy efficiency', '0').split()[0])
    except ValueError:
        energy_efficiency = None

    parsed_info['Energy efficiency'] = energy_efficiency
    return parsed_info

def calculate_savings(km_per_kwh, cost_per_kwh, distance_per_day):
    km_per_year = distance_per_day * 365

    # Petrol calculations
    avg_km_per_liter = 12  # Assumption for an average car
    liters_per_year = km_per_year / avg_km_per_liter
    petrol_cost_per_liter = 2.05  # MYR
    petrol_cost_per_year = liters_per_year * petrol_cost_per_liter

    try:
        if km_per_kwh <= 0:
            raise ValueError("Energy efficiency (KM/kWh) must be greater than zero")

        kwh_per_year = km_per_year / km_per_kwh
        electricity_cost_per_year = kwh_per_year * cost_per_kwh

        results = {
            "Petrol": {
                "Distance per year": f"{km_per_year:.2f} km",
                "Average fuel efficiency": f"{avg_km_per_liter} km/liter",
                "Liters used per year": f"{liters_per_year:.2f} liters",
                "Petrol cost per year": f"RM {petrol_cost_per_year:.2f}"
            },
            "Electricity": {
                "Distance per year": f"{km_per_year:.2f} km",
                "Energy efficiency": f"{km_per_kwh} km/kWh",
                "kWh used per year": f"{kwh_per_year:.2f} kWh",
                "Electricity cost per year": f"RM {electricity_cost_per_year:.2f}"
            },
            "Savings": {
                "Annual savings by using EV": f"RM {petrol_cost_per_year - electricity_cost_per_year:.2f}"
            }
        }
    except (ValueError, ZeroDivisionError) as e:
        results = {
            "Error": f"Unable to calculate savings: {str(e)}. Please check your inputs and try again."
        }

    return results

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

def analyze_manual(manual_text):
    system_prompt = """
    You are an AI assistant specializing in analyzing EV owner's manuals. 
    Based on the provided manual text, generate 5 common questions that users might ask about the vehicle.
    Also, provide brief answers to these questions based on the information in the manual.
    Format your response as a JSON object with questions as keys and answers as values.
    """

    user_prompt = f"Analyze the following EV owner's manual and generate questions and answers:\n\n{manual_text[:4000]}"

    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return eval(response.choices[0].message.content)

def answer_question(manual_text, question):
    system_prompt = """
    You are an AI assistant specializing in answering questions about EVs based on their owner's manuals. 
    Provide a concise and accurate answer to the user's question using the information from the manual.
    If the information is not available in the manual, politely state that and provide a general response if possible.
    """

    user_prompt = f"Manual: {manual_text[:4000]}\n\nQuestion: {question}"

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

# Streamlit UI
st.title("🚗 EV Assistant")


st.markdown("""

<style>
    .centered-container {
        display: flex;
        justify-content: center;  
        align-items: center;      
        height: 53vh;           
        position: relative;       
    }
    .rounded-image {
        border-radius: 15px;  
        overflow: hidden;     
    }
</style>
    """, unsafe_allow_html=True)
st.markdown('<div class="centered-container"><div class="rounded-image"><img src="https://plus.unsplash.com/premium_photo-1682141678972-4965e2599c90?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTc5fHxlbGVjdHJpYyUyMGNhcnxlbnwwfHwwfHx8MA%3D%3D" width="500%"></div></div>', unsafe_allow_html=True)



colored_header(
    label="Your Smart Electric Vehicle Companion",
    description="Get information, calculate savings, and analyze manuals for your EV",
    color_name = "light-blue-70"
)

tab1, tab2, tab3 = st.tabs(["📊 EV Information", "💰 Fuel Savings Calculator", "📘 Manual Analysis"])

with tab1:
    colored_header(
        label="EV Information",
        description="Search for details about specific EV models",
        color_name="light-blue-70",
    )


    with stylable_container(
        key="ev_info_container",
        css_styles="""
            {
                background-color: #22577A;
                border-radius: 10px;
                
            }
        """
    ):
        model = st.text_input("🔍 Search for an EV Model:")

        if st.button("Get EV Information"):
            with st.spinner("Fetching EV information..."):
                ev_info_text = get_ev_info(model)
                ev_info = parse_ev_info(ev_info_text)

            st.write(f"### 🚙 {model} Information:")
            for key, value in ev_info.items():
                st.write(f"**{key}:** {value}")

with tab2:
    colored_header(
        label="EV Fuel Saving Calculator",
        description="Calculate potential savings by switching to an EV",
        color_name="light-blue-70",
    )


    with stylable_container(
        key="calculator_container",
        css_styles="""
            {
                background-color: #22577A;
                border-radius: 10px;
                padding: 20px;
            }
        """
    ):
        col1, col2, col3 = st.columns(3)
        with col1:
            km_per_kwh = st.number_input("⚡ Energy efficiency (KM/kWh):", min_value=0.1, value=6.0, step=0.1)
        with col2:
            cost_per_kwh = st.number_input("💲 Electricity cost (RM/kWh):", min_value=0.01, value=0.571, step=0.001)
        with col3:
            distance_per_day = st.number_input("🛣️ Distance per day (km):", min_value=0.1, value=50.0, step=0.1)

        if st.button("Calculate Savings"):
            results = calculate_savings(km_per_kwh, cost_per_kwh, distance_per_day)

            if "Error" in results:
                st.error(results["Error"])
            else:
                for category, data in results.items():
                    st.subheader(f"{'🐖💰' if category == 'Savings' else ''} {category}")
                    for key, value in data.items():
                        st.write(f"**{key}:** {value}")


st.divider()
st.write("🔍 Need more information? Explore the different tabs for EV details, savings calculations, and manual analysis!")