# This program calculates multiple linear regression
# a sample file "FuelCosumptionCo2" with some data is used
# This program was created by: Emely Alfaro"
# Acknowledgements: Elaheh Jamali (class partner who did the homework with me
#                    harkirat (used this public Github repo served as guide for this program)
# Libraries used: this program uses numpy, matplotlib, pandas and matplotlib.patches for match calculations, manipulating big data
# and plotting the linear regression. I did use sklearn.metrics for calcuting r2 only


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

#opening file csv to begin
df = pd.read_csv('FuelConsumptionCo2.csv')

# separating variables. first independent ones
indepen = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']]

# selecting the one and only dependent variable that will be co2 emissions
depen = df[['CO2EMISSIONS']]

#plotting various plot
for i in indepen:
    plt.scatter(indepen[[i]],depen,color='blue')
    plt.xlabel(i)
    plt.ylabel('CO2 Emissions')
    plt.show()

# setting up a random seed
np.random.seed(2)
# distribuing dataset using random too
split = np.random.rand(len(df))<0.8
train_data_X = indepen[split]
test_data_X = indepen[~split]
train_data_Y = depen[split]
test_data_Y = depen[~split]


#shape will me 5xm_train , m_train = no of training examples
train_data_X = train_data_X.values.T
#shape will be 1xm_train
train_data_Y = train_data_Y.values.T
#shape will be 5xm_test, m_test = no of test examples
test_data_X = test_data_X.values.T
#shape will be 1xm_test
test_data_Y = test_data_Y.values.T

weights = np.zeros((train_data_X.shape[0],1))
bias = 0

indepen_mean = np.mean(train_data_X,axis=1).reshape(train_data_X.shape[0],1)
depen_mean = np.mean(train_data_Y,axis=1).reshape(1,1)
indepen_std = np.std(train_data_X,axis=1).reshape(train_data_X.shape[0],1)
depen_std = np.std(train_data_Y,axis=1).reshape(1,1)

train_data_X = (train_data_X-indepen_mean)/indepen_std
train_data_Y = (train_data_Y-depen_mean)/depen_std
test_data_X = (test_data_X-indepen_mean)/indepen_std
test_data_Y = (test_data_Y-depen_mean)/depen_std

def costFunction(data,labels,parameters,bias):
    """
    used to calculate cost/loss of the model.

    :param data: from fuel comsumption
    :param labels:
    :param parameters:
    :param bias:
    :return: the cost
    """
    predicted = np.dot(parameters.T,data) + bias
    cost = (1/(2*labels.shape[1]))*np.sum(np.power(predicted-labels,2))
    return cost

def learningModel(data,labels,parameters,b,learning_rate=0.001,num_iterations=1000):
    """
    earningModel function is used to learn weights and biases related to model
    Input parameters: data to be trained on, correct labels, initial weights and bias, learning rate and number of iterations
    return learned parameters
    :param data:
    :param labels:
    :param parameters:
    :param b:
    :param learning_rate:
    :param num_iterations:
    :return: the parameters and the bias
    """
    cost = []
    for i in range(num_iterations):
        predicted = np.dot(parameters.T,data) + bias
        parameters = parameters - (learning_rate/labels.shape[1])*(np.dot(data,(predicted-labels).T))
        b = b - (learning_rate/labels.shape[1])*(np.sum(predicted-labels))
        cost.append(costFunction(data,labels,parameters,b))

    plt.plot(range(num_iterations),cost)
    return {
            'parameters':parameters,
            'bias':b}

print('Testing Learning Rate on Training Data')
# pre set learning rate in this program. will vary between 10^-5 to 10^-1
learning = -4 * np.random.rand(11)-1
learning = np.power(10,learning)

# used for collecting r2score and parameters of each testing so to choose afterwards
modelSelection=[]
for i in learning:
    parameters = learningModel(train_data_X,train_data_Y,weights,bias,i)
    train_cost = costFunction(train_data_X,train_data_Y,parameters['parameters'],parameters['bias'])
    test_cost = costFunction(test_data_X,test_data_Y,parameters['parameters'],parameters['bias'])
    print('For Learning rate',i)
    print('Training Cost',train_cost)
    print('Test Cost',test_cost)
    print('R2 Score')
    r2 = r2_score(test_data_Y.T,(np.dot(parameters['parameters'].T,test_data_X)+parameters['bias']).T)
    print(r2)
    print('')
    modelSelection.append([parameters,r2])
# model selection on basis of r2_score
max_r2 = 0
for i,model in enumerate(modelSelection):
    if model[1]>max_r2:
        max_r2 = model[1]
        index = i

# final Model selection
print('Final Model Selection')
print('Learning rate: ',learning[index])
print('Corresponding R2_score: ',modelSelection[index][1])
print('Corresponding Parameters: ',modelSelection[index][0])
print('Test cost: ',costFunction(test_data_X,test_data_Y,modelSelection[index][0]['parameters'],modelSelection[index][0]['bias']))



