# This program calculates multiple regression
# a sample file "FuelConsumptionCo2" with some data is used
# This program was created by: Emely Alfaro
# Program 1: Regression Model
# Acknowledgements: Elaheh Jamali (class partner who did the homework with me
#                   Giorgi Lomia. Our TA who answered questions about the libraries and the linear regression model.
#                   jagadeesh-gajula/Linear-Regression-from-scratch. this public repo was used for debugging.

# Libraries used: this program uses numpy, matplotlib, pandas and matplotlib.patches for match calculations, manipulating big data
# and mpl_toolkits.mplot3d for plotting the 3D model.


import numpy as np  # library supporting large, multi-dimensional arrays and matrices.
import pandas as pd  # library to take data and creates a Python object with rows and columns
import matplotlib.pyplot as plot  # library for embedding plots
from mpl_toolkits.mplot3d import Axes3D  # library for 3D model

data = pd.read_csv('FuelConsumptionCo2.csv')
print(data.shape)
print(data.head())

CO2EMISSIONS = data['CO2EMISSIONS'].values # setting the dependent variable
FUELCONSUMPTION_CITY = data['FUELCONSUMPTION_CITY'].values # setting an independent variable
FUELCONSUMPTION_HWY = data['FUELCONSUMPTION_HWY'].values # setting another dependent variable

# Plotting a scatter plot from our data
figure = plot.figure()
axes = Axes3D(figure)
axes.scatter(CO2EMISSIONS, FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY, color='#CE7D7E')
plot.show()

#formulas to caclulate x and y
m = len(CO2EMISSIONS) # using the length of our dependent variable
x0 = np.ones(m)
X = np.array([x0, CO2EMISSIONS, FUELCONSUMPTION_CITY]).T
#  using the matrix with 0s to start it up
B = np.array([0, 0, 0])
Y = np.array(FUELCONSUMPTION_HWY)
alpha = 0.0001 # setting learning rate for multiple regression


# defining the cost function
def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)

    return J
initialCost = cost_function(X, Y, B)
print("This is the initial cost:", initialCost)

# Gradient Descent formula
def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)

    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # calculating the difference between hipotesis and values
        loss = h - Y
        # calculating gradient
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # calculating the new cost value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost

    return B, cost_history


# 100000 Iterations
newB, cost_history = gradient_descent(X, Y, B, alpha, 100000)

# New Values of B
print("this is the new value of B", newB)

# Final Cost of new B
print("This is the final value of B", cost_history[-1])


# calculating the root mean square error)
def RMSE(Y, Y_prediction):
    rmse = np.sqrt(sum((Y - Y_prediction) ** 2) / len(Y))
    return rmse


# Calculating the coefficient of determination for the exercise.
def r2_score(Y, Y_prediction):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_prediction) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


Y_prediction = X.dot(newB)

print("This is Root Mean Square Error:", RMSE(Y, Y_prediction))
print("This is the coefficient of determination:", r2_score(Y, Y_prediction))
