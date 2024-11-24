# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices
<H3>NAME: KEERTHANA S</H3>
<H3>REGISTER NO.: 212223240070</H3>
<H3>EX. NO.3</H3>
<H3>DATE: 02.09.24</H3>

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load the Dataset:**:  
  Import and read the car price dataset.

2. **Preprocess Data:** 
   Remove unnecessary columns and convert categorical data to numeric.
   
3. **Define Features and Target:**
    Split data into predictor variables (X) and target variable (y).
   
4. **Split Data:**
   Divide data into training and testing sets.
   
5. **Train the Model:**
    Fit a multiple linear regression model on the training data.
   
6.**Evaluate the Model:**
   Use cross-validation to check model performance.
   
7.**Make Predictions:**
    Predict car prices for the test data.
    
8.**Visualize Results:**
    Plot actual vs predicted prices to assess accuracy.
   
## Program:
```py
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv")

# Preprocessing: Drop unnecessary columns and encode categorical variables
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Features and target
X = data.drop('price', axis=1)
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Cross-validation scores
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation Mean Score:", cv_scores.mean())

# Model parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predictions
predictions = model.predict(X_test)

# Plot actual vs predicted prices
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/bf4c9b32-7b8a-4e72-903c-ec5bda3f6245)


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
