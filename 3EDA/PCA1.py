import numpy as np
from sklearn import decomposition
import pandas as pd

#X1, X2, X3 are features/columns
df1= pd.DataFrame({
        'X1':[10,2,8,9,12],
        'X2':[20,50,1,20,22],
        'X3':[10,2,7,10,11]})

pca = decomposition.PCA(n_components=3)

#Creation of Eigen Vectors
pca.fit(df1)
print(df1)

#How the points are projected against new dimensions
df1_pca = pca.transform(df1)
print(df1_pca)

#variance of data along original axes
print(np.var(df1.X1))
print(np.var(df1.X2))
print(np.var(df1.X3))
#Sum of Variance
np.var(df1.X1) + np.var(df1.X2) + np.var(df1.X3)

#understand how much variance captured by each principal component
print(pca.explained_variance_)

#variance of data along principal component axes
np.sum(pca.explained_variance_)


#Percentage of variance
print(pca.explained_variance_ratio_)
#With this we find where to stop as soon as we reach close to 1.
print(pca.explained_variance_ratio_.cumsum()) 

#show the principal components
pca.components_[0]
pca.components_[1]
pca.components_[2]

#specify number of required dimensions as n_components
pca = decomposition.PCA(n_components=2)
pca.fit(df1)
pca.explained_variance_
pca.components_[0]
pca.components_[1]
df1_pca = pca.transform(df1)