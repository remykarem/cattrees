# Trees and forests from scratch

## Usage

Prepare data. Here there are 3 features: the first 2 are numerical and the last is nominal.

```python
import numpy as np
X = np.array([
    [1, 101, 103, 3, 5, 107, 109, 7, 8],
    [1, 101, 103, 3, 5, 107, 109, 7, 8],
    [0,   0,   0, 0, 0,   0,   0, 1, 1]]).T
y = np.array([0, 1, 1, 0, 0, 1, 1, 2, 2])
```

Import module

```python
from decision_tree import DecisionTreeClassifier
```

Initialise and fit data

```python
clf = DecisionTreeClassifier()
clf.fit(X,y)
```

Inference

```python
clf.predict(np.array([[1,1,0]]))
```
