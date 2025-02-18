import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load("models/trained_model.joblib")

# Configure Streamlit page
st.set_page_config(page_title="California Dream Home Price Predictor", layout="wide")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* Darken the background image with a black overlay */
    .stApp {
        background: linear-gradient(
            rgba(0, 0, 0, 0.5), 
            rgba(0, 0, 0, 0.5)
        ), 
        url("https://images.unsplash.com/photo-1523217582562-09d0def993a6?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
        background-position: center;
    }

    /* Main overlay container for all content */
    .overlay {
        background-color: rgba(0, 0, 0, 0.5);
        padding: 2rem;
        border-radius: 0.5rem;
        max-width: 900px;
        margin: 2rem auto;
        color: #fff;  /* White text for contrast on dark background */
    }

    /* Headings and text: white for visibility */
    h1, h2, h3, h4, h5, h6, label, p, span {
        color: #fff !important;
        font-family: 'Arial', sans-serif;
    }

    /* Make buttons stand out */
    .stButton > button {
        background-color: #FFD700 !important;  /* Gold */
        color: #000 !important;
        border-radius: 0.3rem;
        font-weight: bold;
        font-size: 1rem;
        padding: 0.6rem 1.2rem;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Start of main overlay
st.markdown(
    """
    <div class='overlay'>
    <p>Dreaming of the perfect home in sunny California? Whether you're eyeing a cozy 
    bungalow by the beach or a hillside retreat with a view, our machine learning predictive model 
    helps you estimate what <strong>your</strong> ideal dream place might cost.</p>
    """,
    unsafe_allow_html=True,
)

# Title and some filler text
st.title("California Dream Home Price Predictor")
st.write(
    """
    Enter the key details below to see an estimated price—then get ready to start packing!
    """
)

# Split inputs into two columns
col1, col2 = st.columns(2)

with col1:
    medinc = st.number_input(
        "Median Income (in tens of thousands)", 2.0, 12.0, 8.0, 0.1
    )
    house_age = st.number_input("House Age (years)", 0, 100, 30)
    ave_rooms = st.number_input("Average Rooms", 1.0, 10.0, 5.0, 0.1)
    ave_bedrms = st.number_input("Average Bedrooms", 0.5, 5.0, 2.0, 0.1)

with col2:
    population = st.number_input("Population", 100, 5000, 1500)
    ave_occup = st.number_input("Average Occupancy", 1.0, 5.0, 3.0, 0.1)
    latitude = st.number_input("Latitude", 32.0, 42.0, 37.5, 0.1)
    longitude = st.number_input("Longitude", -124.5, -114.0, -122.5, 0.1)


# Prediction helper
def predict_realistic_price(features):
    prediction = model.predict(features.reshape(1, -1))[0]
    return prediction * 100000  # Convert to actual dollars


# Prediction button
if st.button("Predict Price"):
    features = np.array(
        [
            medinc,
            house_age,
            ave_rooms,
            ave_bedrms,
            population,
            ave_occup,
            latitude,
            longitude,
        ]
    )
    predicted_price = predict_realistic_price(features)
    st.success(f"Your California dream home is estimated at: ${predicted_price:,.2f}")

st.write(
    "Note: This predictor uses **median income** in tens of thousands of dollars. "
    "The predicted price is in **actual dollars**."
)

# Close main overlay
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("California Dreaming")
st.sidebar.write(
    "'The California dream is a global dream. It's big. It's bold.' – Gavin Newsom"
)
st.sidebar.write(
    "'California is a place in which a boom mentality and a sense of Chekhovian loss "
    "meet in uneasy suspension.' – Joan Didion"
)
