import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv('Salary_Data.csv')
X = data['YearsExperience']
y = data['Salary']
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.8,random_state=0)
lr = LinearRegression()
lr.fit(X_train.values.reshape(-1,1),y_train)
pickle.dump(lr,open('model.pkl','wb')) # wb ----> Write Bytes







