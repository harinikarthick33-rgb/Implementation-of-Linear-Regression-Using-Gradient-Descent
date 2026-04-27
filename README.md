# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
Load and Prepare Data

Import required libraries and load the dataset using pandas. Clean column names and separate input features (Size, Bedrooms) and target variables (Price, Occupants).

Scale the Features

Apply feature scaling using StandardScaler to normalize the input data, which improves the performance of SGD.
Initialize the Models

Create two SGD Regressor models—one for predicting price and another for predicting occupants—with suitable parameters.
Train the Models

Fit both models using the scaled input data and their respective target values.
Predict Output

Take user input (house size and bedrooms), scale it using the same scaler, and use both trained models to predict and display the price and number of occupants.

## Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Startup.csv")

# Select one feature (R&D Spend) and target (Profit)
X = data['R&D Spend'].values
y = data['Profit'].values

# Normalize (important for gradient descent)
X = (X - X.mean()) / X.std()

# Initialize parameters
m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

# Gradient Descent
for i in range(epochs):
    y_pred = m * X + b
    
    # Gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)

# Predictions for plotting
y_pred = m * X + b

# Plot
plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show()
```

/*
Program to implement the linear regression using gradient descent.
Developed by: Harini V K
RegisterNumber: 212225220036
*/


## Output:
<img width="850" height="622" alt="Screenshot 2026-04-27 142335" src="https://github.com/user-attachments/assets/87c865ce-50dc-486c-90f3-4e9ac64e33b3" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
