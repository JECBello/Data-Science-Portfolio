import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.rcParams["patch.force_edgecolor"] = True
sns.set()
sns.set_style('whitegrid')

customers = pd.read_csv('Ecommerce Customers')
customers.head()
customers.describe()
customers.info()

sns.jointplot('Time on App','Yearly Amount Spent',data=customers, color = 'grey',
             joint_kws=dict(s = 20, color='teal'),
             marginal_kws=dict(bins = 20))
             
 sns.jointplot('Time on App', 'Length of Membership', data=customers, kind = 'hex', color = 'black')
 
 sns.pairplot(customers)
 
 sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers,
          scatter_kws=dict(color= 'grey',s=15),
          line_kws=dict(color = 'grey'))
          
 X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
 y = customers['Yearly Amount Spent']
 
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
 
 from sklearn.linear_model import LinearRegression
 lm = LinearRegression()
 lm.fit(X_train,y_train)
 print(lm.coef_)
 
 prediction = lm.predict(X_test)
 sns.scatterplot(y_test,prediction)
 
 from sklearn import metrics
 MAE = metrics.mean_absolute_error(y_test,prediction)
 MSE = metrics.mean_squared_error(y_test,prediction)
 RMSE = np.sqrt(MSE)
 
 print(f'MAE: {MAE}\nMSE: {MSE}\nRMSE: {RMSE}')
 
 res_plot = sns.distplot([y_test - prediction], bins = 45, color = 'grey',
                        hist_kws=dict(alpha = 0.5))
 res_plot.set_xlim([-40,50])
 
 df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
 df
