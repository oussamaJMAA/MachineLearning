import matplotlib.pyplot as plt
import pandas as pd

desired_width = 320
pd.set_option('display.width', desired_width)
# import the  dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values
X = X.reshape(-1, 1)
y = dataset.iloc[:, 1].values
# splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
# Fitting the model  to the training set
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)
# Predict the test set result
y_pred = lm.predict(X_test)
# Visualising the training results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, lm.predict(X_train), color='blue')
plt.title("Salary vs Experience ( Training set)")
plt.xlabel('Yeas of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, lm.predict(X_train), color='blue')
plt.title("Salary vs Experience ( Test set)")
plt.xlabel('Yeas of Experience')
plt.ylabel('Salary')
plt.show()
