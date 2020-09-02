from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#Loading Dataset
iris=datasets.load_iris()

#printing description and features
#print(iris.DESCR)
features = iris.data
labels = iris.target
print(features[0],labels[0])

#training the classifier
model=KNeighborsClassifier()
model.fit(features,labels)
#testing the classifier
preds=model.predict([[1,1,1,1]])
print(preds)