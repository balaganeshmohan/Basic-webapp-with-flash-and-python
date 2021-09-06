import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets



iris = datasets.load_iris()           #loading iris data
data = pd.DataFrame(iris.data, columns=iris.feature_names)    #to pandas dataframe
data['target'] = pd.Series(iris.target)
encodings = {0: 'Setosa', 1: 'Versicolor', 2:'Viriginica'}    #integer encoding for traget strings
data = data.replace(['Setosa', 'Versicolor' , 'Virginica'], [0, 1, 2])  #replacing values with the dictionary

X = data.iloc[:, 0:-1]     #predictor variables
y = data.iloc[:,-1]        #target variable


clf = LogisticRegression(max_iter=1000)      #classifier
clf.fit(X,y)       #fitting model

'''method to predict unseen data with the trained model
input: 4 dimensions of the petal data in integer or float form
output: predicted class
'''


def predict(a, b, c, d):
    arr = np.array([a, b, c, d])
    arr = arr.astype(np.float64)
    query = arr.reshape(1, -1)
    prediction = encodings[clf.predict(query)[0]]  
    return prediction

