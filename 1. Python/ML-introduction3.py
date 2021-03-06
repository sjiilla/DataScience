# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:53:29 2017

@author: Sreenivas.J
"""
#DecissionTree and Predict methods are very important in this example. This is ther real starting/building of ML
#Here we will be playing with more columns. However DecisionTreeClassifier algorithm works only on numeric/continuous data/columns
#Henceforth we need to assign  non numerical columns to dummy columns
#This technique is called one-hot encoding
#import os
import pandas as pd
from sklearn import tree
import io
import pydot #if we need to use any external .exe files.... Here we are using dot.exe

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("D:/Data Science/Data/")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#Transformation of non numneric cloumns
#There is an exception with the pclass. Though it's conincidentally is a number but it's a classification bit not nnumber.
#titanic_train1 = titanic_train[['Pclass', 'Sex', 'Embarked', 'Fare']]

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
print(type(titanic_train1))
titanic_train1.shape
titanic_train1.info()
titanic_train1.describe
titanic_train1.head(6) #Get me top 6 records

#now the drop non numerical columns where we will not be applying logic. Something like we will not apply logic on names, passengerID ticket id etc...
X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'],1) #As a best practive use Capital X bcoz many columns will be there
y_train = titanic_train['Survived'] #As a best practice use lower case y_.. as there will be only one output/Target column.

#Another issue is Fare column data is missing for one.
#Now we have to better guess and add the values for missing.
#And then build.

#Build the decision tree model
dt = tree.DecisionTreeClassifier() #By default DecisionTreeClassifier algorithm works only on numeric/continuous data/columns.
#.fit builds the model. In this case the model building is using Decission Treee Algorithm
dt.fit(X_train,y_train) #Builds a decision tree classifier from the training set (X, y).
type(dt)

#visualize the decciion tree
dot_data = io.StringIO()  

tree.export_graphviz(dt, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("decission_tree3.pdf")

#predict the outcome using decision tree
titanic_test = pd.read_csv("test.csv")
titanic_test.info() #Found that one row has Fare = null. Instead of dropping this column, let's take the mean of it.
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

#Now apply same get_dummies and drop columns on test data as well like above we did for train data
titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
titanic_test['Survived'] = dt.predict(X_test)
titanic_test.to_csv("Submission.csv", columns=['PassengerId', 'Survived'], index=False)