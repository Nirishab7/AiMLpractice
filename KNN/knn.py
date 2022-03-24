from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
iris_data=iris.data
iris_label=iris.target

xtrain,xtest,ytrain,ytest=train_test_split(iris_data,iris_label,test_size=0.30)
classifier=KNeighborsClassifier(5)
classifier.fit(xtrain,ytrain)
y_pred=classifier.predict(xtest)


print(confusion_matrix(ytest,y_pred))
print(accuracy_score(ytest,y_pred))
#print(classification_report(ytest,y_pred))