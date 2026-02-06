import pandas as pd
import numpy as np
import sklearn

df = pd.read_csv("data/WA_FN-UsecC_-Telco-Customer-Churn.csv" , low_memory=True , nrows = 5000)

print(df.head())
print(df.shape)
print(df.info())

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')

print(df.isnull().sum())

df.dropna(inplace=True)
df.drop('customerID' , axis=1 , inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1 ,  'No': 0 })
df = pd.get_dummies(df, drop_first=True)
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Churn' , axis=1)
Y = df['Churn']

X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size=0.2 , random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report

model =  LogisticRegression(max_iter=1000 )
model.fit(X_train , Y_train)

y_pred = model.predict(X_test)

print("Accuracy:" , accuracy_score(Y_test , y_pred))
print(confusion_matrix(Y_test  , y_pred))
print(classification_report(Y_test , y_pred))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100 , random_state=42)
rf.fit(X_train , Y_train)

rf_pred = rf.predict(X_test)

print("Random Forest Accuracy :" , accuracy_score(Y_test , rf_pred))

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

sns.boxplot(x='Churn' , y='tenure' , data=df)
plt.title("Tenure vs Churn")
plt.show()

sns.boxplot(x='Churn' , y='MonthlyCharges' , data=df)
plt.title("Churn vs MonthlyCharges")
plt.show()

contract_cols = [col for col in df.columns if 'Contract_' in col]
df[contract_cols +['Churn']].groupby('Churn').mean()

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight( class_weight= 'balanced' ,classes=np.unique(Y_train) , y=Y_train
)

weights = {0: class_weights[0], 1 : class_weights[1]}

model = LogisticRegression(max_iter=1000, class_weight=weights)

from sklearn.model_selection import GridSearchCV

param_grid = {'C' : [0.01 , 0.1 , 1 , 10]}

grid = GridSearchCV(LogisticRegression(max_iter=1000, class_weight=weights), param_grid,cv=5,scoring='roc_auc')

grid.fit(X_train , Y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Optimized Accuracy:" , accuracy_score(Y_test, y_pred))

from sklearn.metrics import roc_auc_score

y_prob = best_model.predict_proba(X_test)[:,1]
print("ROC-AUC Score :", roc_auc_score(Y_test, y_prob))