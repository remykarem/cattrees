import numpy as np
from trees_and_forests import DecisionTreeRegressor

X = np.array([
    [1, 101, 103, 3, 5, 107, 109, 7, 8],
    [1, 101, 103, 3, 5, 107, 109, 7, 8],
    [0,   0,   0, 0, 0,   0,   0, 1, 1]]).T
y = np.array([13, 30, 22, 31, 16, 19, 18, 27, 15])

rgr = DecisionTreeRegressor()
rgr.fit(X,y)
rgr.predict(X)
