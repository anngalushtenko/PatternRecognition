# load the iris dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# training the model on training set
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X, y)
y_pred_mnb = mnb.predict(X_test)
# Complement Naive Bayes
cnb = ComplementNB()
cnb.fit(X, y)
y_pred_cnb = cnb.predict(X_test)


# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics

print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred_gnb) * 100)
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred_mnb) * 100)
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred_cnb) * 100)