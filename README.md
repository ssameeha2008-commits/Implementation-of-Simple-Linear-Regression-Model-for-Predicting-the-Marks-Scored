# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
df=pd.read_csv("student_scores.csv")
df.head(10)
plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
x_train
y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.plot(x_train,lr.predict(x_train),color='red')
lr.coef_
lr.intercept_
y_pred=lr.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("MSE:",mse)
print("RMSE:",rmse)
print("MAE:",mae)
print("R2:",r2)
```

## Output:
<img width="1019" height="577" alt="{2ADC7BA8-9B34-4BFF-B2AC-E3BD58223E5D}" src="https://github.com/user-attachments/assets/0d6d8a45-1492-429d-bf26-8b9dd9d743ba" />
<img width="1157" height="555" alt="{71D18270-5503-4E85-9E41-448A0620D0D5}" src="https://github.com/user-attachments/assets/a09d9a74-3d0c-4900-a923-306bb70fd96c" />
<img width="1272" height="754" alt="{D7A85DCD-0853-4DD2-9FF0-9553BE52A994}" src="https://github.com/user-attachments/assets/dcc33743-8409-4912-9bc0-7ad434df93b2" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
