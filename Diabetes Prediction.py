#importing the libraries
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score

#importing dataset 
diabetes=pd.read_csv('diabetes.csv')
X=diabetes.iloc[:,0:-1].values
y=diabetes.iloc[:,-1].values
#some operations
descibbe=diabetes.describe()
head=diabetes.head()

#splitting the data
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20,
                                                stratify=y,random_state=0)
#standerscaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#trainning the model 
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,y_train)

#prediciting
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)