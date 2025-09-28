# Real-Estate-Price-Prediction-and-Deployment-Using-Streamlit
Real Estate Price Prediction Model - End-to-end ML project featuring data analysis, feature engineering, model training (Linear Regression), and Streamlit deployment. Predicts property prices using MRT proximity, convenience stores, and distance to city center. Demonstrates full ML pipeline from data to production-ready application.

---
## Primary Objectives
To build, evaluate, and deploy a predictive model that accurately forecasts house prices per unit area using property characteristics, and to create a user-friendly web application for real-time predictions.

---

## 📂 Dataset
- Source: [Statso Website](https://amanxai.com/2023/12/11/real-estate-price-prediction-using-python/)
- Contains 414 entries with features: Transaction date, House age, Distance to MRT, Convenience stores, Latitude, Longitude, House price per unit area.


*![Sample Dataset Screenshot](https://github.com/Uzo-Hill/Real-Estate-Price-Prediction-and-Deployment-Using-Streamlit/blob/main/raw_data.PNG)*

---

## 🛠️ Tech Stack & Tools Used

- Python 3.9+ - Primary programming language

- Jupyterlab - Interactive development and analysis

- **Data Processing & Analysis** - Pandas, Numpy.

- **Machine Learning**: Scikit-learn, Machine learning models - LinearRegression, RandomForestRegressor, GradientBoostingRegressor


- Pickle - Model serialization and storage

- Data Visualization: Matplotlib, Seaborn, plotly, Streamlit Charts - Native charting in web app

### Web Application & Deployment
- Streamlit : Web application framework

- Streamlit Community Cloud : Free deployment platform

- GitHub: Version control and repository hosting

---

## Data Preprocessing & Feature Engineering

### Key preprocessing steps
- Handled zero prices and validated data quality
- Created 'Distance_to_center' using Taipei City Hall coordinates
- Dropped raw coordinates to avoid multicollinearity
- Final feature set: Distance to MRT, Convenience stores, Distance to center


```python
# Create distance to city center (using approximate Taiwan's Taipei city center, coordinates - (25.03770° N, 121.56456° E) )

city_center_lat, city_center_lon = 25.0330, 121.5654
df['Distance_to_center'] = np.sqrt((df['Latitude'] - city_center_lat)**2 + 
                                  (df['Longitude'] - city_center_lon)**2)

```
---

## Exploratory Data Analysis
*![Relationships between features](https://github.com/Uzo-Hill/Real-Estate-Price-Prediction-and-Deployment-Using-Streamlit/blob/main/Relationships_Features.PNG)*



*![Correlation Analysis](https://github.com/Uzo-Hill/Real-Estate-Price-Prediction-and-Deployment-Using-Streamlit/blob/main/Corr_Matrix.PNG)*


*![Distribution](https://github.com/Uzo-Hill/Real-Estate-Price-Prediction-and-Deployment-Using-Streamlit/blob/main/Distribution.PNG)*

---

## Model Training & Evaluation

```python
# Initialize models

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Linear Regression': LinearRegression()
}

# Train and evaluate models

results = {}
for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    # Store results
    results[name] = {
        'model': model,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"\n{name} Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
```


- **Random Forest Performance:**
  - MAE: 10.1093
  - RMSE: 11.8040
  - R²: 0.2296
  - Cross-validation R²: 0.1622 (±0.0801)

- **Gradient Boosting Performance:**
  - MAE: 10.5972
  - RMSE: 12.5106
  - R²: 0.1346
  - Cross-validation R²: 0.1095 (±0.0694)

- **Linear Regression Performance:**
  - MAE: 9.8528
  - RMSE: 11.4104
  - R²: 0.2802
  - Cross-validation R²: 0.3608 (±0.0201)


---

## Feature Importance (Linear Regression)

*![features importance](https://github.com/Uzo-Hill/Real-Estate-Price-Prediction-and-Deployment-Using-Streamlit/blob/main/Feature_Importance.PNG)*




*![predictedvsActual](https://github.com/Uzo-Hill/Real-Estate-Price-Prediction-and-Deployment-Using-Streamlit/blob/main/Actual%20vs%20Pred.PNG)*



---
## Saving the Best Trained Model - Linear Regression Model:

```python
# Save the best model and scaler
print("\nSaving the best model and scaler...")
with open('real_estate_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")

```
---

## 🚀 Deployment

### Streamlit Web Application
- Framework: Streamlit

- Deployment: Streamlit Community Cloud

- Integration: GitHub continuous deployment

- Features: Real-time predictions with feature importance visualization

### Application Features
- Interactive input sliders for property features

- Real-time price predictions

- Feature importance analysis

- Model performance metrics

- Professional contact footer


### Web app URL: https://uzoh-real-estate-price-prediction-and-deployment-app.streamlit.app/



*![web app](https://github.com/Uzo-Hill/Real-Estate-Price-Prediction-and-Deployment-Using-Streamlit/blob/main/web_app.PNG)*


---

## 📈 Key Insights
- MRT Proximity is Paramount: Properties closer to transit stations command premium prices

- Local Amenities Matter: Convenience stores significantly impact property values

- City Center Less Critical: Neighborhood amenities outweigh central location importance

- Investment Strategy: Focus on properties within 500m of MRT with 5+ convenience stores

## 🎯 Business Impact
Price Estimation: Provides baseline property valuation

Investment Guidance: Identifies key value drivers for real estate investment

Market Analysis: Reveals location preference patterns in the data

---

## ⚠️ Limitations & Challenges
Model Performance (R² = 28%)
The moderate performance is attributed to:

1. **Limited Feature Set**:

  - Missing property characteristics (size, rooms, condition)

  - No neighborhood quality metrics

  - Limited amenity data

2. **Data Constraints**:

  - Small dataset (414 records)

  - Limited geographic diversity

---

### 🔮 Future Improvements
Model Enhancement
- Collect additional features (square footage, room counts)

- Include neighborhood characteristics (schools, crime rates)

- Implement advanced ensemble methods


---
## 🏗️ Project Structure

```text
real-estate-prediction/
├── app.py                          # Streamlit application
├── real_estate_model.pkl          # Trained model
├── scaler.pkl                     # Feature scaler
├── requirements.txt               # Dependencies
├── Real_Estate.csv               # Original dataset
└── README.md                     # Project documentation

```

---
## Installation & Usage
```bash
# Clone repository
git clone https://github.com/Uzo-Hill/real-estate-price-prediction.git

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

---
👨‍💻 Author
Hillary Uzoh

📧 Email: uzohhillary@gmail.com

💼 LinkedIn: http://www.linkedin.com/in/hillaryuzoh

𝕏 Twitter: https://x.com/UzohHillary

📄 License
This project is open source and available under the MIT License.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check issues page.
---
⭐ Star this repo if you find it helpful!

















