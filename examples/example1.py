import numpy as np
from trees_and_forests import DecisionTreeRegressor, GradientBoostingRegressor

X = np.array([
    [1, 101, 103, 3, 5, 107, 109, 7, 8],
    [1, 101, 103, 3, 5, 107, 109, 7, 8],
    [0,   0,   0, 0, 0,   0,   0, 1, 1]]).T
y = np.array([13, 67, 66, 14, 14, 64, 63, 16, 15])

rgr = GradientBoostingRegressor(5)
rgr.fit(X,y)
rgr.predict(X)