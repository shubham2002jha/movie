import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
HouseDF = pd.read_csv('https://raw.githubusercontent.com/huseinzol05/Stock-Prediction-Models/master/dataset/AMD.csv')
print(HouseDF.head())
print(HouseDF.info())
print(HouseDF.describe())
print(HouseDF.columns)
sns.pairplot(HouseDF)
plt.show()
##X = HouseDF[[' open', ' high', 'low',
##               ' close','volume']]
##
##y = HouseDF['adj close']
##from sklearn.model_selection import train_test_split
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
##from sklearn.linear_model import LinearRegression
##lm = LinearRegression()
##lm.fit(X_train,y_train)
##print(lm.intercept_)
##coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
##print(coeff_df)
##predictions = lm.predict(X_test)
##plt.scatter(y_test,predictions)
##sns.distplot((y_test-predictions),bins=50);
##plt.show()
##from sklearn import metrics
##print('MAE:', metrics.mean_absolute_error(y_test, predictions))
##print('MSE:', metrics.mean_squared_error(y_test, predictions))
##print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
