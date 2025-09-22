# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image  # For adding images if needed

# Set page configuration
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load your saved model and scaler
@st.cache_resource  # This caches the load so it only happens once
def load_model():
    with open('real_estate_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Load the model and scaler
model, scaler = load_model()

# App title and description
st.title("🏠 Real Estate Price Predictor")
st.write("""
Predict house prices per unit area based on key property features.
This model uses Linear Regression and considers three main factors that influence property values.
""")

# Create input fields in the sidebar
st.sidebar.header("Input Property Features")

# Input fields for the three features
distance_to_mrt = st.sidebar.number_input(
    "Distance to Nearest MRT Station (meters)",
    min_value=0.0,
    max_value=5000.0,
    value=500.0,  # Default value
    help="Enter distance in meters"
)

num_convenience_stores = st.sidebar.number_input(
    "Number of Convenience Stores Nearby",
    min_value=0,
    max_value=20,
    value=5,  # Default value
    help="Number of convenience stores in the vicinity"
)

distance_to_center = st.sidebar.number_input(
    "Distance to City Center (km)",
    min_value=0.0,
    max_value=20.0,
    value=0.05,  # Default value
    help="Distance to Taipei City Center in kilometers"
)

# Create a button for prediction
predict_button = st.sidebar.button("Predict Price", type="primary")

# Main content area
if predict_button:
    # Create input array in the exact same order as during training
    input_data = np.array([[distance_to_mrt, num_convenience_stores, distance_to_center]])
    
    # Scale the input using the same scaler from training
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Display results
    st.success("### Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Price per Unit Area", f"${prediction[0]:.2f}")
        
        # Show input summary
        st.write("**Input Summary:**")
        st.write(f"- Distance to MRT: {distance_to_mrt} meters")
        st.write(f"- Convenience Stores: {num_convenience_stores}")
        st.write(f"- Distance to Center: {distance_to_center} km")
    
    with col2:
        # Show feature importance insights
        st.write("**How Features Affect Price:**")
        st.write("📍 **Closer to MRT** → Higher Price")
        st.write("🏪 **More Stores** → Higher Price")  
        st.write("🏙️ **Closer to Center** → Higher Price")

# Add some explanations and tips
st.sidebar.markdown("---")
st.sidebar.info("""
**💡 Tips for Accurate Predictions:**
- MRT distance greatly influences price
- More convenience stores increase value
- City center proximity adds premium
""")

# Footer
st.markdown("---")
st.caption("""
This predictive model was built using Linear Regression on real estate data.
The model considers three key location-based features that most significantly impact property values.
""")