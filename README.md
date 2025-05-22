# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries: 'numpy' for numerical operations, 'pandas' for data manipulation, and 'StandardScale'r from sklearn for feature scaling.
2. Create a function linear_regression that takes input features x1, target variable y, a learning rate, and the number of iterations.
3. Add a column of ones to x1 to account for the intercept term in the linear regression model.
4. Initialize the parameter vector theta to zeros.
5. Use gradient descent to update theta iteratively based on the predictions and errors.
6. Read the dataset '50_Startups.csv' into a pandas DataFrame. The 'header=None' argument indicates that the CSV file does not have a header row.
7. Extract features 'x' (all columns except the last two) and target variable 'y' (the last column).
8. Convert 'x' to a float type.
9. Scale the features 'x1' and target variable 'y' using 'StandardScaler'. This standardizes the data to have a mean of 0 and a standard deviation of 1.
10. Call the 'linear_regression' function with the scaled features and target variable to obtain the optimized parameters 'theta'.
11. Create a new data point (e.g., [165349.2, 136897.8, 471784.1]) and reshape it for scaling.
12. Scale the new data point using the same scaler.
13. Use the learned parameters 'theta' to make a prediction for the new scaled data point.
14. Reshape the prediction and inverse transform it to get the predicted value in the original scale.
## Program:

```
Program to implement the linear regression using gradient descent.
Developed by: D.Varshini
RegisterNumber:  212223230234

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1, y, learning_rate=0.01, num_iters=100):
    x = np.c_[np.ones(len(x1)), x1]
    theta = np.zeros(x.shape[1]).reshape(-1,1)
    for i in range(num_iters):
        predictions = (x).dot(theta).reshape(-1,1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(x1)) * x.T.dot(errors)
    return theta
data = pd.read_csv('50_Startups.csv',header=None)
print("Name: D.Varshini\nReg.no: 212223230234")
print(data.head())
x = (data.iloc[1:, :-2].values)
print(x)
x1=x.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_scaled = scaler.fit_transform(x)
y1_scaled = scaler.fit_transform(y)
print(x1_scaled)
print(y1_scaled)
theta = linear_regression(x1_scaled, y1_scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```


## Output:
![image](https://github.com/user-attachments/assets/262dc6da-b7db-4fb4-957f-a81cb3a93db9)

![image](https://github.com/user-attachments/assets/ca6b71f3-c653-4b0a-ac09-e56941a34dfd)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
