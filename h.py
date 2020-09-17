import pandas as pd
import math
dataset = pd.read_csv("weight-height.csv")
import warnings
import pickle
warnings.filterwarnings("ignore")
dataset.info()
dataset.describe()
dataset.isnull().sum()
dataset['Gender'].replace('Female',0, inplace=True)
dataset['Gender'].replace('Male',1, inplace=True)
X = dataset.loc[:, dataset.columns!='Height'].values
y = dataset.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_pred = lin_reg.predict(X_test)
my_height_pred = lin_reg.predict([[0,103]])
print(my_height_pred)
print('R square = ',lin_reg.score(X_train,y_train))
pickle.dump(lin_reg,open('modelh.pkl','wb'))
model=pickle.load(open('modelh.pkl','rb'))
