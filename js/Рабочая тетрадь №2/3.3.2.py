import pandas as pd
from sklearn import preprocessing

url = 'https://raw.githubusercontent.com/akmand/datasets/master/iris.csv'

MinMax_Scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
Z_Scale = preprocessing.StandardScaler()

dataFrame = pd.read_csv(url)
dataFrame[['sepal_length_cm']] = MinMax_Scale.fit_transform(dataFrame[['sepal_length_cm']])
dataFrame[['sepal_width_cm']] = Z_Scale.fit_transform(dataFrame[['sepal_width_cm']])

dataFrame.head(5)