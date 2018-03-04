import pandas as pd
import numpy as np
from sklearn.svm import SVC


dfX = pd.read_csv("irisX.csv")
dfY = pd.read_csv("irisY.csv")

dfX = dfX.drop(columns = ["C","D"])
dfX = dfX.drop([0])

X_train = dfX.drop(dfX.index[list(range(100,149))])
Y_train = dfY.drop(dfY.index[list(range(100,149))])

X_test = dfX.drop(dfX.index[list(range(0,99))])
Y_test = dfY.drop(dfY.index[list(range(0,99))])

model = SVC(gamma = 0.5, C = 1)
model.fit(X_train,Y_train)
print(model.score(X_test,Y_test))
print(sum(model.n_support_))
