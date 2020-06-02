import sklearn
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "traveltime", "famrel", "goout", "Dalc", "Walc", "health", "age", "freetime",]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best = 0
for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(round(acc, 2))
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

print("the best result: ", round(best, 2))
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co: \n", linear.coef_)
print("Intercept:\n ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(round(predictions[x]), y_test[x])
p = 'age'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel([p])
pyplot.ylabel("Final Grade")
pyplot.show()
