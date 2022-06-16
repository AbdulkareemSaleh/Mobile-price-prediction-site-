import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import pickle

warnings.filterwarnings("ignore")
#read data
data=pd.read_csv('data/train.csv')
#split data
X= data.drop('price_range', axis=1) 
y = data.price_range
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#build the model 
clf = LogisticRegression()
#train the model
clf.fit(X_train, y_train)
#predict using the model 
y_pred=clf.predict(X_test)
# evaluate the model 
print(accuracy_score(y_pred, y_test))
pickle.dump(clf, open('clf.pkl', 'wb'))  