#Train a logistic regression classifier to predict whether a flower is iris virginica or not
import numpy as np
from sklearn import datasets
from  sklearn.linear_model import  LogisticRegression
import matplotlib.pyplot as plt


#import dataset
iris= datasets.load_iris()
X= iris["data"][:,3:] #slicing only 3rd column to extract just one feature
Y= (iris["target"]==2).astype(np.int)
#print(Y)
#print(X)
#Trian the logistic regression classifier
model= LogisticRegression()
model.fit(X,Y)

pred=model.predict(([[1.6]]))
#print(pred)
x_new= np.linspace(0,3,1000).reshape(-1,1)
y_prob= model.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1], "g-", label="virginica")
plt.show()