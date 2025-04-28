# 🏡 Linear Regression - House Price Prediction

## 📌 Objective
The main aim of this project was to implement both simple and multiple linear regression to predict house prices and understand how various features impact the pricing.

---

## 🛠️ Tools and Libraries Used
- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

---

## 🚀 Steps I Followed

1. **Dataset Loading:**  
   Loaded the dataset using Pandas to explore its structure.

2. **Data Preprocessing:**  
   - Filled missing values for numeric columns.
   - Encoded binary categorical variables into numerical format.
   - One-hot encoded the 'furnishingstatus' column.

3. **Feature Engineering:**  
   - Created a new feature called `total_rooms` by combining `bedrooms` and `bathrooms` after researching that total living space impacts house prices significantly.

4. **Exploratory Data Analysis:**  
   - Plotted a correlation heatmap to understand relationships among features.

5. **Model Training:**  
   - Built a Linear Regression model using scikit-learn.
   - Used 80% of the data for training and 20% for testing.

6. **Model Evaluation:**  
   - Evaluated the model with MAE, MSE, RMSE, and R² metrics.
   - Analyzed feature importance by checking the magnitude and direction of coefficients.

7. **Visualization:**  
   - Plotted the regression line for 'Area' vs 'Price' to visualize how the prediction works.

---

## 📈 Results

| Metric | Value |
|:------:|:-----:|
| MAE    | 970043.40 |
| MSE    | 1754318687330.65 |
| RMSE   | 1324506.96 |
| R² Score | 0.65 |

- **Intercept:** 260032.36
- **Top Features Impacting Price:**
  - area
  - bathrooms
  - total_rooms
---

## 🔎 Key Learnings and Interpretation

- **Feature Engineering** helped slightly improve the model accuracy.
- **Total Rooms** showed a positive correlation with price.
- **Correlation heatmaps** provided insights about multicollinearity.
- Coefficients interpretation helped understand which features increased or decreased price predictions.
  
---

## 🧠 Interview Question Answers

1. **Assumptions of Linear Regression:** Linearity, Independence, Homoscedasticity, No Multicollinearity, Normality of Errors.
2. **Coefficient Interpretation:** Represents how much the target variable changes with a unit change in the feature.
3. **R² score:** Represents the proportion of variance explained by the model.
4. **MSE vs MAE:** MSE penalizes larger errors more than MAE.
5. **Detecting Multicollinearity:** Using Correlation Matrix or Variance Inflation Factor (VIF).
6. **Simple vs Multiple Regression:** One feature vs multiple features.
7. **Can Linear Regression classify?** No, it predicts continuous values. Use Logistic Regression for classification tasks.
8. **What if assumptions are violated?** The model might give biased or unreliable results.

---

## 📂 Files Included

- `linear_regression.py`
- `house_price.csv`
- `regression_line.png`
- `correlation_heatmap.png`
- `README.md`

---

## ✅ Conclusion
Working on this project made me realize the importance of not just building models, but also properly understanding the data before modeling.  
By adding feature engineering and visualization steps, I got deeper insights into the dataset, which improved the model's performance and made the project richer.

---
