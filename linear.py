# This program calculates linear regression
# a sample file "Salary.csv" with some data is used
# This program was created by: Emely Alfaro
# Program 1: Regression Model
# Acknowledgements: Elaheh Jamali (class partner who did the homework with me
# Libraries used: this program uses numpy, matplotlib, pandas and matplotlib.patches for match calculations, manipulating big data
# and plotting the linear regression. I did use sklearn.modelselection only for splitting the data

import numpy
import matplotlib.pyplot as plot
import pandas as pd

# opening the csv file
data = pd.read_csv('Salary.csv')
print(data.shape)
print(data.head())

# separating x and y. (dependent and independent variables using .values
X = data['YearsExperience'].values
Y = data['Salary'].values

# using the mean to calculate the  value of x and y
mean_x = numpy.mean(X)
mean_y = numpy.mean(Y)

# getting m by calculating the length of x
m = len(X)

# calculating our b0 and b1 to be used in the regression formula
numerator = 0
denominator = 0
for i in range(m):
    numerator += (X[i] - mean_x) * (Y[i] - mean_y)
    denominator += (X[i] - mean_x) ** 2
b1 = numerator / denominator
b0 = mean_y - (b1 * mean_x)
print("The value of the B1 is:", b1, "and the value of b0 is:", b0) #printing the values we just calculated
# setting the plot limitations
max_x = numpy.max(X) + 5
min_x = numpy.min(X) - 5
x = numpy.linspace(min_x, max_x, 1000)
y = b0 + b1 * x # regression line formula

# setting up the plot format. using scatter points
plot.plot(x, y, color='#040707', label='Regression Line')
plot.scatter(X, Y, c='#FFA500', label='Scatter Plot')

plot.xlabel('YearsExperience')
plot.ylabel('Salary')
plot.legend()
plot.show()

# time to calculate RMSE
RMSE = 0
for i in range(m):
    y_prediction = b0 + b1 * X[i]
    RMSE += (Y[i] - y_prediction) ** 2
RMSE = numpy.sqrt(RMSE/m)
print("The Root Mean Square Error is", RMSE)

# formula for the coefficient of determination
SST = 0
SSR = 0
for i in range(m):
    y_prediction = b0 + b1 * X[i]
    SST += (Y[i] - mean_y) ** 2
    SSR += (Y[i] - y_prediction) ** 2
r2 = 1 - (SSR/SST)
print("The coefficient of determination is", r2)
