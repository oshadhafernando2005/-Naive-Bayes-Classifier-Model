import numpy as np
import pandas as pd
import plotly.express as px
dataset = pd.read_csv('/content/BC_Data.csv')
dataset.head()
dataset.info()
fig = px.scatter(dataset, x="radius_mean",y="texture_mean", color = "diagnosis",
width=800, height=800)
fig.show()
dataset = dataset.drop(["id"], axis = 1)
dataset = dataset.drop(["Unnamed: 32"], axis = 1)
X = dataset.drop(["diagnosis"], axis = 1)
y = dataset['diagnosis']
# Perform Minimum - Maximum Normalization:
X1 = (X - np.min(X)) / (np.max(X) - np.min(X))
from sklearn.model_selection import train_test_split
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size = 0.3,
random_state = 42)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X1_train, y_train)
y_pred=nb.predict(X1_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
nb_cm = confusion_matrix(y_test, y_pred, labels = nb.classes_)
nb_cm = ConfusionMatrixDisplay(nb_cm, display_labels = nb.classes_)
nb_cm.plot()
from sklearn.metrics import RocCurveDisplay
nb_roc = RocCurveDisplay.from_estimator(nb, X1_test, y_test)
