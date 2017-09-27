import os
import pandas as pd
import seaborn as sns
import numpy as np

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("D:/Data Science/Data")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#explore bivariate relationships: categorical vs categorical 
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Sex'])
#margins=True gives sub total and total across cross-tab
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Pclass'], margins=True)

sns.factorplot(x="Survived", data=titanic_train, kind="count", size=6)
#hue is for further classification plotting, In this case Plot survivied for each sex.
sns.factorplot(x="Sex", hue="Survived", data=titanic_train, kind="count", size=6) 
sns.factorplot(x="Pclass", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Embarked", hue="Survived", data=titanic_train, kind="count", size=6)

#explore bivariate relationships: categorical vs continuous 
sns.factorplot(x="Fare", row="Survived", data=titanic_train, kind="box", size=6)
#.map is a inline function like a for loop
#Survived Vs Fare
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.kdeplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.distplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.boxplot, "Fare").add_legend()

#explore bivariate relationships: continuous vs continuous 
#Estimate a covariance matrix, given data and weights.
np.cov(titanic_train['SibSp'], titanic_train['Parch'])
#correlation coefficient matrix
np.corrcoef(titanic_train['SibSp'], titanic_train['Parch'])
sns.jointplot(x="SibSp", y="Parch", data=titanic_train)