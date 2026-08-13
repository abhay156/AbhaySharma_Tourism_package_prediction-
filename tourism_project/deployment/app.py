
import os
import streamlit as st
import pandas as pd
import joblib


# Load the model committed by the pipeline
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_model.pkl"
)

model = joblib.load(model_path)


# Streamlit page configuration
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="wide"
)


st.title("✈️ Wellness Tourism Package Predictor")

st.write(
    "Enter the customer's details below to predict "
    "whether they are likely to purchase the "
    "Wellness Tourism Package."
)


# Customer Information
st.header("Customer Details")

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Free Lancer",
            "Small Business",
            "Large Business"
        ]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single", "Divorced"]
    )

    designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )


with col2:

    type_of_contact = st.selectbox(
        "Type of Contact",
        ["Company Invited", "Self Inquiry"]
    )

    product_pitched = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Standard",
            "Deluxe",
            "Super Deluxe",
            "King"
        ]
    )

    number_of_person_visiting = st.number_input(
        "Number of People Visiting",
        min_value=1,
        max_value=20,
        value=2
    )

    number_of_children = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=10,
        value=0
    )

    number_of_trips = st.number_input(
        "Number of Trips",
        min_value=0.0,
        max_value=50.0,
        value=3.0
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=20000.0
    )


# Customer Interaction Details
st.header("Customer Interaction Details")

col3, col4 = st.columns(2)


with col3:

    duration_of_pitch = st.number_input(
        "Duration of Pitch",
        min_value=1.0,
        max_value=120.0,
        value=15.0
    )

    number_of_followups = st.number_input(
        "Number of Followups",
        min_value=0.0,
        max_value=20.0,
        value=3.0
    )

    pitch_satisfaction = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )


with col4:

    preferred_property_star = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

    passport = st.selectbox(
        "Passport",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    own_car = st.selectbox(
        "Own Car",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


# Prediction
if st.button(
    "Predict Package Purchase",
    type="primary"
):

    input_data = pd.DataFrame({

        "Age": [age],

        "TypeofContact": [type_of_contact],

        "CityTier": [city_tier],

        "DurationOfPitch": [
            duration_of_pitch
        ],

        "Occupation": [occupation],

        "Gender": [gender],

        "NumberOfPersonVisiting": [
            number_of_person_visiting
        ],

        "NumberOfFollowups": [
            number_of_followups
        ],

        "ProductPitched": [
            product_pitched
        ],

        "PreferredPropertyStar": [
            preferred_property_star
        ],

        "MaritalStatus": [
            marital_status
        ],

        "NumberOfTrips": [
            number_of_trips
        ],

        "Passport": [passport],

        "PitchSatisfactionScore": [
            pitch_satisfaction
        ],

        "OwnCar": [own_car],

        "NumberOfChildrenVisiting": [
            number_of_children
        ],

        "Designation": [designation],

        "MonthlyIncome": [
            monthly_income
        ]
    })


    # Make prediction using the same threshold used during evaluation
    probability = model.predict_proba(input_data)[0][1]

    classification_threshold = 0.45

    prediction = int(
        probability >= classification_threshold
    )


    # Display result
    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            "The customer is likely to purchase "
            "the Wellness Tourism Package."
        )

    else:

        st.info(
            "The customer is unlikely to purchase "
            "the Wellness Tourism Package."
        )


    st.metric(
        "Purchase Probability",
        f"{probability:.2%}"
    )
