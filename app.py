# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load your saved model and scaler
@st.cache_resource
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

distance_to_mrt = st.sidebar.number_input(
    "Distance to Nearest MRT Station (meters)",
    min_value=0.0,
    max_value=5000.0,
    value=500.0
)

num_convenience_stores = st.sidebar.number_input(
    "Number of Convenience Stores Nearby",
    min_value=0,
    max_value=20,
    value=5
)

distance_to_center = st.sidebar.number_input(
    "Distance to City Center (km)",
    min_value=0.0,
    max_value=20.0,
    value=0.05
)

predict_button = st.sidebar.button("Predict Price", type="primary")

if predict_button:
    input_data = np.array([[distance_to_mrt, num_convenience_stores, distance_to_center]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    st.success("### Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Price per Unit Area", f"${prediction[0]:.2f}")
        st.write("**Input Summary:**")
        st.write(f"- Distance to MRT: {distance_to_mrt} meters")
        st.write(f"- Convenience Stores: {num_convenience_stores}")
        st.write(f"- Distance to Center: {distance_to_center} km")
    
    with col2:
        st.write("**How Features Affect Price:**")
        st.write("📍 **Closer to MRT** → Higher Price")
        st.write("🏪 **More Stores** → Higher Price")  
        st.write("🏙️ **Closer to Center** → Higher Price")

    # SIMPLE FEATURE IMPORTANCE USING STREAMLIT NATIVE CHART
    st.markdown("---")
    st.subheader("📊 Feature Importance Analysis")
    
    features = ['Distance to MRT', 'Convenience Stores', 'Distance to Center']
    coefficients = model.coef_
    
    # Create a simple bar chart using Streamlit's native bar_chart
    importance_df = pd.DataFrame({
        'Feature': features,
        'Impact on Price': coefficients
    })
    
    st.bar_chart(importance_df.set_index('Feature'))
    
    # Show the exact coefficients
    st.write("**Exact Impact Values:**")
    for feature, coef in zip(features, coefficients):
        impact = "Increases" if coef > 0 else "Decreases"
        st.write(f"- **{feature}**: {impact} price by {abs(coef):.2f} units per change")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**💡 Tips for Accurate Predictions:**
- MRT distance greatly influences price
- More convenience stores increase value
- City center proximity adds premium
""")

st.markdown("---")
st.caption("""
This predictive model was built using Linear Regression on real estate data.
The model considers three key location-based features that most significantly impact property values.
""")

# Ultra-compact contact links
st.markdown("---")
st.subheader("🌐 Connect")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""<a href="mailto:uzohhillary@gmail.com" target="_blank" style="text-decoration: none;">
    <div style="text-align: center; padding: 5px;">
    <span style="color: #EA4335; font-size: 24px;">📨</span><br>
    <small style="color: #333;">Email</small>
    </div></a>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<a href="https://github.com/Uzo-Hill" target="_blank" style="text-decoration: none;">
    <div style="text-align: center; padding: 5px;">
    <span style="color: #333; font-size: 24px;">💻</span><br>
    <small style="color: #333;">GitHub</small>
    </div></a>""", unsafe_allow_html=True)

with col3:
    st.markdown("""<a href="http://www.linkedin.com/in/hillaryuzoh" target="_blank" style="text-decoration: none;">
    <div style="text-align: center; padding: 5px;">
    <span style="color: #0077B5; font-size: 24px;">👔</span><br>
    <small style="color: #333;">LinkedIn</small>
    </div></a>""", unsafe_allow_html=True)

with col4:
    st.markdown("""<a href="https://x.com/UzohHillary" target="_blank" style="text-decoration: none;">
    <div style="text-align: center; padding: 5px;">
    <span style="color: #000; font-size: 24px;">𝕏</span><br>
    <small style="color: #333;">X</small>
    </div></a>""", unsafe_allow_html=True)

st.markdown("""<div style="text-align: center; margin: 5px 0;">
<small style="color: #666;">uzohhillary@gmail.com</small>
</div>""", unsafe_allow_html=True)


