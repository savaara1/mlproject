import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

#removing first column
df = pd.read_csv("Book1.csv")
first_column = df.columns[0]
df = df.drop([first_column], axis=1)
print(df)

#Removing No/Low Variance Features
df7 = df.to_numpy()
print(df7)
selector = VarianceThreshold( threshold=0.0)
df8 = selector.fit_transform(df7)
print(df8)


#Removing Highly Correlated Features
df2 = pd.DataFrame(df8)
cor_matrix = df2.corr().abs()
#print(cor_matrix)
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1) .astype(bool))
#print(upper_tri)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > .95)]
print(); print(to_drop)
df3 = df2.drop(df2.columns.intersection(to_drop), axis=1)
print(); print(df3.head())

#converting final df to csv
df3.to_csv(("Book2.csv"),header=False, index = False)
