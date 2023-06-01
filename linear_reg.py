"""
push what I did before to github
"""

import matplotlib.pyplot as plt # visualization module in python. #seaborn is another one
import numpy as np # numerical calculations, creates vectors
from sklearn.linear_model import LinearRegression # module for ML algos
from sklearn import datasets
from sklearn.model_selection import train_test_split


# loading & exploring the data
diabetes = datasets.load_diabetes()
print(type(diabetes))

print(diabetes.data.shape) # looks at the shape of the entire dataset
print(diabetes.target.shape) # looks at the shape ofthe target data (1 metric per row)
print(diabetes.feature_names) # prints out the different columns


X_train, X_test, y_train, y_test = train_test_split(diabetes.data,diabetes.target, test_size=0.2, random_state=1)

print(X_train)

# 1 Set up the model (instantiating)
model = LinearRegression()

# 2 fitting the data
model.fit(X_train, y_train)

# 3 evaluating
print(model.score(X_test, y_test)) 
#print(model.predict(X_test))

# plot the data
y_pred = model.predict(X_test)
plt.plot(y_test,y_pred,".")


x = np.linspace(0,330,100)
print(x)

y = x

plt.plot(x,y)
plt.show()

"""
"""


