import pandas as pd

msg=pd.read_csv('naive/naivtext.csv',names=['message','labels'])
msg['labelnum']=msg.labels.map({'pos':1,'neg':0})
x=msg.message
y=msg.labelnum

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y)

from sklearn.feature_extraction.text import CountVectorizer
ctn=CountVectorizer()
xtrain_dtm=ctn.fit_transform(xtrain)
xtest_dtm=ctn.transform(xtest)

df=pd.DataFrame(xtrain_dtm.toarray(),columns=ctn.get_feature_names())
print(df[0:5])

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(xtrain_dtm,ytrain)
pred=nb.predict(xtest_dtm)

from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
print(accuracy_score(ytest,pred))
print(precision_score(ytest,pred))
print(confusion_matrix(ytest,pred))