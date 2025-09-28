#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Real Estate Price Prediction and Deployment

# ## Introduction

# This project aims to develop a machine learning model to predict house prices per unit area based on various property features. The model will help real estate professionals, investors, and homeowners estimate property values more accurately by considering factors like location, age, proximity to transportation, and nearby amenities.

# ## Primary Objective

# To build, evaluate, and deploy a predictive model that accurately forecasts house prices per unit area using property characteristics, and to create a user-friendly web application for real-time predictions.

# In[ ]:





# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# ## Data Loading and Exploration

# In[2]:


df = pd.read_csv(r"C:\Users\DELL\Desktop\Data Analytics Projects\Real Estate Prediction\Real_Estate.csv")


# In[3]:


df.head()


# In[ ]:





# In[4]:


print(f"Dataset shape: {df.shape}")
print()
print("\nFirst 5 rows:")
print()
print(df.head())
print()
print("\nDataset info:")
print()
print(df.info())
print()
print("\nDescriptive statistics:")
print()
print(df.describe())


# In[ ]:





# ## Data Cleaning and Preprocessing

# ### Data Quality Checks

# In[5]:


# data quality checks

print("\nCleaning and preprocessing data...")

# Check for missing values
print("Missing values:")
print(df.isnull().sum())


# In[ ]:





# In[6]:


# Handle zero prices (assuming they might be errors or missing data)

df = df[df['House price of unit area'] > 0]


# In[ ]:





# In[7]:


# Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")


# In[ ]:





# In[8]:


# Check data types and convert if necessary
df['Transaction date'] = pd.to_datetime(df['Transaction date'], errors='coerce')


# In[ ]:





# In[9]:


# Handle any outliers using IQR method for price

Q1 = df['House price of unit area'].quantile(0.25)
Q3 = df['House price of unit area'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR



outlier_count = df[(df['House price of unit area'] < lower_bound) | (df['House price of unit area'] > upper_bound)].shape[0]
print(f"Price outliers detected: {outlier_count}")

if outlier_count == 0:
    print("✓ No statistical outliers found - data appears well-distributed")
    print(f"Price range: {df['House price of unit area'].min():.2f} - {df['House price of unit area'].max():.2f}")
    print(f"IQR bounds: {lower_bound:.2f} - {upper_bound:.2f}")
else:
    print(f"Consider reviewing {outlier_count} outlier(s)")


# Data Characteristics:
# 
# - Wide price diversity: From very affordable (0.37) to premium properties (65.57)
# - No artificial caps: Natural market distribution

# In[ ]:





# ## Feature Engineering

# In[10]:


# Create distance to city center (using approximate Taiwan's Taipei city center, coordinates - (25.03770° N, 121.56456° E) )

city_center_lat, city_center_lon = 25.0330, 121.5654
df['Distance_to_center'] = np.sqrt((df['Latitude'] - city_center_lat)**2 + 
                                  (df['Longitude'] - city_center_lon)**2)


# In[11]:


df.head()


# we engineered a new feature, Distance_to_city_center, to create a more directly interpretable and powerful predictor for our model. In real estate, proximity to the urban core is a primary driver of value, and this feature would explicitly captures that relationship.

# In[ ]:





# In[12]:


# Dropping the original latitude and longitude to avoid multicollinearity

df = df.drop(columns=['Latitude', 'Longitude'])


# In[19]:


df.head()  # confirming the drop


# ## Exploratory Data Analysis

# ### Correlation matrix

# In[13]:


# Correlation Matrix (excluding 'Transaction date')
plt.figure(figsize=(10, 8))

# Select numeric columns and exclude 'Transaction date'
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['Transaction date'], errors='ignore')

correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix (Numeric Features)')
plt.tight_layout()
plt.show()


# - MRT Proximity Drives Price: Homes closer to transit stations command higher prices (strong -0.49 correlation).
# 
# - Engineered Feature Works: Distance_to_center effectively replaces raw coordinates, simplifying the geographic signal.
# 
# - Age Impact Varies: The house's age category influences price differently, which may challenge a simple linear model.

# In[ ]:





# ### Distribution of target variable

# In[14]:


# Distribution of target variable

plt.figure(figsize=(10, 6))
sns.histplot(df['House price of unit area'], kde=True)
plt.title('Distribution of House Price per Unit Area')
plt.xlabel('Price per Unit Area')
plt.show()


# In[ ]:





# In[15]:


# Relationship between features and target

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.scatterplot(x='Distance to the nearest MRT station', y='House price of unit area', data=df, ax=axes[0,0])
axes[0,0].set_title('Price vs Distance to MRT')

sns.boxplot(x='Number of convenience stores', y='House price of unit area', data=df, ax=axes[0,1])
axes[0,1].set_title('Price vs Number of Convenience Stores')

sns.scatterplot(x='House age', y='House price of unit area', data=df, ax=axes[1,0])
axes[1,0].set_title('Price vs House Age')

sns.scatterplot(x='Distance_to_center', y='House price of unit area', data=df, ax=axes[1,1])
axes[1,1].set_title('Price vs Distance to City Center')
plt.tight_layout()
plt.show()


# - **_Price vs Distance to MRT_**: Clear negative correlation - properties closer to MRT stations (0-500m) command significantly higher prices
#   
# - **_Price vs Number of Convenience Stores_**: More stores nearby correlates with higher prices, suggesting amenities add value.
# 
#   
# - **_Price vs House Age_**: Weak correlation with high variability
# 
# - **_Price vs Distance to City Center_**: Moderate negative relationship where properties closer to Taipei's city center tend to have higher values, though the relationship is less pronounced than MRT proximity.

# In[ ]:





# ## Prepare data for modeling

# In[16]:


# Selecting features and target variable
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Distance_to_center']
target = 'House price of unit area'

X = df[features]
y = df[target]


# ### Data Splitting and Scaling

# In[17]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:





# In[18]:


# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


# In[ ]:





# ## Model Training and Evaluation

# In[19]:


# Initialize models

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Linear Regression': LinearRegression()
}


# In[ ]:





# In[20]:


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


# In[21]:


# Select best model based on R² score

best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name}")


# Linear Regression performed best (R²=0.28) as the linear relationships between our features and price were more reliable than the complex patterns other models tried to fit on this small dataset.

# In[ ]:





# In[ ]:





# In[22]:


# Feature importance for Linear Regression
if best_model_name == 'Linear Regression':
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(best_model.coef_)  # Use absolute coefficients as importance
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature')
    plt.title('Feature Importance (Linear Regression)')
    plt.show()

    print("Feature Importance:")
    print(feature_importance)

    # Show actual coefficients (with direction)
    coefficients_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': best_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)

    print("\nActual Coefficients (with direction):")
    print(coefficients_df)


# - **MRT Proximity is Paramount (Importance: 7.55)**: For every unit of distance closer to the MRT, property price increases significantly. Being near a station is the single biggest value driver.
# 
# - **Convenience Stores Add Value (Importance: 5.18)**: More nearby convenience stores directly increase property value, highlighting the importance of immediate amenities.
# 
# - **City Center Distance Less Critical (Importance: 0.56)** - Distance to city center has minimal impact compared to MRT and convenience factors, suggesting neighborhood-level amenities outweigh central location.

# Investment Strategy: Focus on properties within 500m of MRT stations in areas with 5+ convenience stores for maximum value potential, as these two factors account for ~95% of the model's predictive power.

# In[ ]:





# In[23]:


# Visualize Actual vs Predicted Values for Linear Regression
plt.figure(figsize=(8, 6))

# Get predictions
y_pred = best_model.predict(X_test_scaled)

# Create scatter plot
plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predictions')

# Add perfect prediction line (y=x)
max_val = max(max(y_test), max(y_pred)) + 5
min_val = min(min(y_test), min(y_pred)) - 5
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices (Linear Regression)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print performance metrics for reference
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")


# - Predictions generally follow the ideal line, showing the model captures the main price trends correctly.
# 
# - The model consistently underestimates the actual price for the most expensive houses (top-right dots below the line), suggesting it misses factors that drive premium prices.
# 
# - For most mid-priced homes, predictions are relatively close to actual values, indicating good reliability for typical properties.

# In[ ]:





# ## Saving the Trained Model

# In[24]:


# Save the best model and scaler
print("\nSaving the best model and scaler...")
with open('real_estate_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")


# In[ ]:





# # 🏠 Real Estate Price Prediction and Deployment Project
# 
# ## 📌 Project Summary
# This end-to-end machine learning project focused on predicting **house price per unit area** using real estate data from Taiwan. The dataset contained 414 records with features such as house age, distance to the nearest MRT station, number of convenience stores, and property location (latitude & longitude).
# 
# The key steps included:
# 1. **Data Exploration & Cleaning**  
#    - Removed zero-priced records and checked for missing values.  
#    - Converted `Transaction date` to datetime format.  
#    - Detected and confirmed no extreme outliers in price distribution.  
# 
# 2. **Feature Engineering**  
#    - Created a new feature: **Distance_to_center** (distance from Taipei city center).  
#    - Dropped raw latitude and longitude to avoid multicollinearity.  
#    - Final feature set:  
#      - Distance to the nearest MRT station  
#      - Number of convenience stores  
#      - Distance to city center  
# 
# 3. **Exploratory Data Analysis (EDA)**  
#    - Found a **strong negative correlation** (-0.49) between price and distance to MRT stations.  
#    - Number of convenience stores showed a **positive correlation** (0.27) with price.  
#    - Distance to the city center had a weaker effect compared to MRT proximity and stores.  
# 
# 4. **Modeling & Evaluation**  
#    - Trained multiple models: **Linear Regression, Random Forest, Gradient Boosting**.  
#    - **Linear Regression performed best** with:  
#      - R² Score: **0.28** (28%)  
#      - RMSE: **11.41**  
#      - MAE: **9.85**  
#    - Feature importance (Linear Regression):  
#      - Distance to MRT: **highest impact**  
#      - Convenience stores: **moderate impact**  
#      - Distance to center: **minor impact**  
# 
#    📊 *Key insight*: MRT proximity and nearby stores account for ~95% of predictive power.  
# 
# 5. **Model Saving & Deployment**  
#    - Best model (**Linear Regression**) and scaler were saved as `.pkl` files.  
#    - Built an interactive **Streamlit web app** for real-time predictions.  
#    - Integrated with **GitHub + Streamlit Community Cloud** for free deployment.  
#    - 🌍 Live app: [Real Estate Price Predictor](https://uzoh-real-estate-price-prediction-and-deployment-app.streamlit.app/)  
# 
# ---
# 
# ## ⚠️ Challenges & Limitations
# - **Model Performance (R² = 28%)**:  
#   The model explains only 28% of the variance in house prices. The moderate R² score can be attributed to:
# 
# 1. Limited Feature Set: Only 3 location-based features were available, missing critical property characteristics like:
# 
#     - Square footage and number of rooms
# 
#     - Property condition and interior quality
# 
#     - School district ratings and neighborhood demographics
# 
# 2. Dataset Size: With 414 initial entries (306 training samples), the model had limited data to capture complex real estate patterns
#   
# ---
# 
# ## Key Results & Business Insights
#    - MRT proximity emerged as the dominant price driver, explaining most of the predictable variance
# 
#    - Local amenities (convenience stores) proved more important than city center proximity
# 
#    - The model provides reliable predictions for mid-range properties but underestimates premium properties#
# 
# ---
# 
# ## ✅ Recommendations
# 
# **For Model Improvement**
# 
# - **Feature Expansion:** Collect data on property size, room counts, condition, and age
# 
# - **Additional Data:** Include school ratings, crime rates, and neighborhood characteristics
# 
# - **Advanced Modeling:** Experiment with ensemble methods on expanded feature sets
# 
# **For Business Application**
# - **Investment Strategy:** Focus on properties within 500m of MRT stations with multiple convenience stores
# 
# - **Price Validation:** Use as a baseline estimator while acknowledging limitations for premium properties
# 
# - **Continuous Learning:** Implement model retraining with new transaction data
# 
# 
# ---
# 
# 📍 **Conclusion**  
# Despite limitations, this project successfully demonstrated the full **end-to-end ML workflow**: from data preparation and feature engineering to model training, evaluation, saving, and deployment via **Streamlit + GitHub**. The deployed app provides an easy-to-use tool for property price estimation and can be improved further with richer data and advanced modeling techniques.
# 

# 
# ---
# *Project developed by Hillary Uzoh | [Web App](https://uzoh-real-estate-price-prediction-and-deployment-app.streamlit.app/) | [GitHub](https://github.com/Uzo-Hill)*
# 

# In[ ]:




