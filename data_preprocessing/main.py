import numpy as np
import pandas as pd

desired_width = 320
pd.set_option('display.width', desired_width)
# import the  dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :3].values
y = dataset.iloc[:, -1]

# Taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_country = LabelEncoder()
X[:, 0] = labelencoder_country.fit_transform(X[:, 0])
# creating dummy variables
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# X = X[:, 1]
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
# Split the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

