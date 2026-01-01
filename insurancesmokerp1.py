import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score, root_mean_squared_error

df = pd.read_csv('insurance.csv')

#Only keeping the target columns that I think matters for the prediction of medical charges
df1 = df.drop(columns = ['sex', 'children', 'region'])

#Changing the type of data in smoker column from string to integer
df1['smoker'] = df1['smoker'].apply(lambda x: 1 if x == 'yes' else 0)


#Splitting the dataset into train test
X = df1[['age', 'bmi', 'smoker']]
y = df1['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

#Standarizing the bmi column values using StandardScaler
scaler = StandardScaler()
X_train[['bmi']] = scaler.fit_transform(X_train[['bmi']])
X_test[['bmi']] = scaler.transform(X_test[['bmi']])

#Training the model
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

#Accuracy scores
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("The root mean squared error is ", rmse)
print("The r2 score is ", r2)

#Plotting y_pred against y_test
sns.scatterplot(x = y_test, y = y_pred)
plt.title("Predicted charges against Actual charges")
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.show()