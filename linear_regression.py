# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# Load the dataset
df = pd.read_csv('house_price.csv')

# Basic data exploration
print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns)

# Preprocessing

# Fill missing values only for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Encode binary categorical columns (yes/no -> 1/0)
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# One-Hot Encode 'furnishingstatus'
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Feature Engineering: Create a new feature 'total_rooms'
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# Visualization: Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

# Features (X) and Target (y)
X = df.drop('price', axis=1)
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")

# Print model coefficients
print("\nModel Parameters:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficients:")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print(feature_importance)

# Plotting: Simple regression line (for 'area' feature)
try:
    plt.scatter(X_test['area'], y_test, color='blue', label='Actual')
    plt.scatter(X_test['area'], y_pred, color='red', label='Predicted')
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.title('Simple Linear Regression: Area vs Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('regression_line.png')
    plt.show()
except Exception as e:
    print("\nSkipping plot because multiple features are used or error:", e)

    