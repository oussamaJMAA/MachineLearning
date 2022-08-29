import pandas as pd
import numpy as np

desired_width = 320
pd.set_option('display.width', desired_width)
# import the  dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :4].values
y = dataset.iloc[:, -1].values
# Encode categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_country = LabelEncoder()
X[:, 3] = labelencoder_country.fit_transform(X[:, 3])
# creating dummy variables
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[:, 1:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)

print(lm.coef_)
