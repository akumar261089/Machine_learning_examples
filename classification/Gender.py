# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('SUV.csv')
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_decision_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_decision_tree.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
classifier_KNeighborsClassifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_KNeighborsClassifier.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
classifier_LogisticRegression = LogisticRegression(random_state = 0)
classifier_LogisticRegression.fit(X_train, y_train)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier_SVC = SVC(kernel = 'linear', random_state = 0)
classifier_SVC.fit(X_train, y_train)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier_svc_kernel = SVC(kernel = 'rbf', random_state = 0)
classifier_svc_kernel.fit(X_train, y_train)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_naive = GaussianNB()
classifier_naive.fit(X_train, y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RandomForestClassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RandomForestClassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_decision_tree = classifier_decision_tree.predict(X_test)
y_pred_KNeighborsClassifier = classifier_KNeighborsClassifier.predict(X_test)
y_pred_LogisticRegression = classifier_LogisticRegression.predict(X_test)
y_pred_SVC = classifier_SVC.predict(X_test)

y_pred_svc_kernel = classifier_svc_kernel.predict(X_test)
y_pred_naive = classifier_naive.predict(X_test)
y_pred_RandomForestClassifier = classifier_RandomForestClassifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)
cm_KNeighborsClassifier = confusion_matrix(y_test, y_pred_KNeighborsClassifier)
cm_LogisticRegression = confusion_matrix(y_test, y_pred_LogisticRegression)
cm_SVC = confusion_matrix(y_test, y_pred_SVC)
cm_svc_kernel = confusion_matrix(y_test, y_pred_svc_kernel)
cm_naive = confusion_matrix(y_test, y_pred_naive)
cm_RandomForestClassifier = confusion_matrix(y_test, y_pred_RandomForestClassifier)
#Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test,  y_pred_decision_tree)
accuracy_score(y_test,  y_pred_decision_tree, normalize=False)

accuracy_score(y_test,  y_pred_KNeighborsClassifier)
accuracy_score(y_test,  y_pred_KNeighborsClassifier, normalize=False)

accuracy_score(y_test,  y_pred_LogisticRegression)
accuracy_score(y_test,  y_pred_LogisticRegression, normalize=False)


accuracy_score(y_test,  y_pred_SVC)
accuracy_score(y_test,  y_pred_SVC, normalize=False)

accuracy_score(y_test,  y_pred_svc_kernel)
accuracy_score(y_test,  y_pred_svc_kernel, normalize=False)


accuracy_score(y_test,  y_pred_naive)
accuracy_score(y_test,  y_pred_naive, normalize=False)

accuracy_score(y_test,  y_pred_RandomForestClassifier)
accuracy_score(y_test,  y_pred_RandomForestClassifier, normalize=False)