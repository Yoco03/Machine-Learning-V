import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
X_train = [[5], [10], [15], [20], [24]] #diamters of pizzas
y_train = [[27], [39], [43], [50], [70]] #prices of pizzas

# Testing set
X_test = [[2], [4], [10], [18]] #Packs of coffee
y_test = [[28], [32], [45], [58]] #Price of coffee in Rand

# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 100, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=3)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Sale of coffee per pack')
plt.xlabel('Packs of coffee')
plt.ylabel('Price of coffee in Rand')
plt.axis([0, 25, 0, 80])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show()

#Data (control)
print (X_train)
print (X_train_quadratic)
print (X_test)
print (X_test_quadratic)
