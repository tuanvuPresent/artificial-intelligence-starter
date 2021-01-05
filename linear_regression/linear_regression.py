import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# -------------------------------  LinearRegression  -----------------------------------------
x1 = np.array([[5], [15], [25], [35], [45], [55]])
y1 = np.array([5, 20, 14, 32, 22, 38])
model = LinearRegression().fit(x1, y1)
print('predict', model.predict(x1))
print('coefficient', model.score(x1, y1))
print('coef', model.coef_)
print()

x2 = np.array([[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]])
y2 = np.array([4, 5, 20, 14, 32, 22, 38, 43])
model = LinearRegression().fit(x2, y2)
print('predict', model.predict(x2))
print('coefficient', model.score(x2, y2))
print()

# ------------------------------- PolynomialFeatures ------------------------------------------
x = np.array([[5], [15], [25], [35], [45], [55], [65], [75]])
y = np.array([5, 20, 14, 32, 22, 38, 29, 45])
x_ = PolynomialFeatures(degree=5, include_bias=True).fit_transform(x)
model = LinearRegression().fit(x_, y)
print('predict', model.predict(x_))
print('coefficient', model.score(x_, y))
