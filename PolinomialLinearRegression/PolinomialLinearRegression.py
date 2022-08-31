import pandas as pd
import matplotlib.pyplot as plt

desired_width = 320
pd.set_option('display.width', desired_width)
# import the  dataset
dataset = pd.read_csv('Position_Salaries.csv')
print(dataset)
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

print(X)
print(y)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

print(X.shape,y.shape)
lr.fit(X, y)


from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
# Visualising the LinearRegression results
plt.scatter(X, y, color="red")
plt.plot(X, lr.predict(X), color="blue")
plt.ticklabel_format(style='plain')
plt.show()

# Visualising the PolinomialRegression results
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg_2.predict(X_poly), color="blue")
plt.ticklabel_format(style='plain')
plt.show()

#Predict a new result with Linear Regression


