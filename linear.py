# This program calculates linear regression
# a sample file "Salary.csv" with some data is used
# This program was created by: Emely Alfaro"
# Acknowledgements: Elaheh Jamali (class partner who did the homework with me
#                    Lalith Bharadwaj (Her public Github repo served as guide for this program)
# Libraries used: this program uses numpy, matplotlib, pandas and matplotlib.patches for match calculations, manipulating big data
# and plotting the linear regression. I did use sklearn.modelselection only for splitting the data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches

# finding the Slope
def Slope(a,b):
    """
    "this function finds the slope of the line"
    :param a: variable a
    :param b: variable b
    :return: the slope of the line
    """
    n = len(a)
    two_sum = np.sum(a*b)
    sumX = np.sum(a)
    sumY = np.sum(b)
    sumX_2 = np.sum(a**2)
    slope = (n*two_sum-sumX*sumY)/(n*sumX_2-(sumX)**2)
    return slope


def Intercept(a,b):
    """
    Finding Intercept of linear regression line
    :param a: variable a
    :param b: variable b
    :return: the intercept of the line
    """

    intercept = np.mean(b)-Slope(a, b)*np.mean(a)
    return intercept


def Predictions(slope,x_input,intercept):
    """
    making predictions
    :param slope: slope of the line
    :param x_input: x input
    :param intercept: intercept of the line
    :return:
    """
    predict=slope*x_input + intercept
    return predict

def R_squared(predicted_values,test_values):
    """
    getting rsquared
    :param predicted_values: predicted values
    :param test_values: test values we use (y)
    :return:
    """
    f=predicted_values
    y=test_values
    print(f,'\n\n',y)
    #sum of squares
    ss_total=np.sum((y-np.mean(y))**2)
    ss_res=np.sum((y-f)**2)
    #R-squared formula
    R_2=1-(ss_res/ss_total)
    return R_2

def correlation_coeff(predicted_values,test_values):
    """
    getting the correlations
    :param predicted_values: predicted values for the correlations
    :param test_values: test values to use
    :return:
    """
    a=predicted_values
    b=test_values
    n=len(a)
    two_sum=np.sum(a*b)
    sumX=np.sum(a)
    sumY=np.sum(b)
    sumX_2=np.sum(a**2)
    sumY_2=np.sum(b**2)
    score=(n*two_sum-sumX*sumY)/np.sqrt((n*sumX_2-(sumX)**2)*(n*sumY_2-(sumY)**2))
    return score

def Covariance(X,Y):
    """
    finding covariance for our x and y
    :param X: variable x to predict
    :param Y: variable y
    :return: cov - the covariance
    """
    a=X
    b=Y
    n=len(a)
    two_sum=np.sum(a*b)
    cov=two_sum/n-np.mean(a)*np.mean(b)
    return cov

# this line imports the data in csv format
dataset=pd.read_csv('Salary.csv')
# this csv file contains random data about salaries and years of experience.
# we are trying to see the regression between salaries and years of experience

# splitting data using libraries
array = dataset.values
X = array[:,0]
print(X.shape)
Y = array[:,1]
print(Y.shape)

# printing covariance
print(Covariance(X,Y))

# spliting test and train data
test_size = 0.10
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size= test_size, random_state=seed)

# this code prints the intercepts of the data
intercept=Intercept(X_train,Y_train)
slope=Slope(X_train,Y_train)
print("this is the intercept", intercept, "and the slope" , slope)
predictions = Predictions(slope=slope, x_input=X_validation, intercept=intercept)
print("these are the predictions", predictions)
print("these are the r squared", R_squared(predicted_values=predictions,test_values=Y_validation))
print("this is the coefficient of determination", correlation_coeff(test_values=Y_validation,predicted_values=predictions))

# calculating linear regression with the equation
y=slope*X+intercept

# plotting linear regression
plt.scatter(X,Y,marker='^',color='k',alpha=0.55)
plt.plot(X,y,color='R',linewidth=2)
red_patch = mpatches.Patch(color='red', label='Regression Line')
plt.legend(loc=0,handles=[red_patch])
plt.title('Linear Regression Model')
plt.tight_layout(pad=2)
plt.grid(False)
plt.show()













