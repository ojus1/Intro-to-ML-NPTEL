import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge


X = pd.read_csv("LR_X.csv")
Y = pd.read_csv("LR_Y.csv")
model = Ridge(alpha = 10)
model.fit(X,Y)
print(model.coef_)
print(model.intercept_)
