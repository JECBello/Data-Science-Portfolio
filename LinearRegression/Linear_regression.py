# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.rcParams["patch.force_edgecolor"] = True
sns.set()
sns.set_style('whitegrid')

# Read CSV file and view contents
customers = pd.read_csv('Ecommerce Customers')
customers.head()
customers.describe()
customers.info()

# Create Jointplot comparing Time on Website and Yearly Amount Spend
sns.jointplot('Time on App','Yearly Amount Spent',data=customers, color = 'grey',
             joint_kws=dict(s = 20, color='teal'),
             marginal_kws=dict(bins = 20))

 # Create Jointplot comparing Time on App and Length of Membership using graph of kind hex            
 sns.jointplot('Time on App', 'Length of Membership', data=customers, kind = 'hex', color = 'black')
 
 # Create pairplot to quickly visualize any trends in data 
 sns.pairplot(customers)
 
 # Create Linear model plot of Yearly Amount Spent vs. Length of Membership
 sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers,
          scatter_kws=dict(color= 'grey',s=15),
          line_kws=dict(color = 'grey'))
  
 # Set X and y variable to be used for training and testing model 
 X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
 y = customers['Yearly Amount Spent']
 
 # import train_test_split and use it on data  
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
 
 # Create Regression Model and Train it 
 from sklearn.linear_model import LinearRegression
 lm = LinearRegression()
 lm.fit(X_train,y_train)
 print(lm.coef_)
 
 # Use model to predict test data. Plot data to visualize trends 
 prediction = lm.predict(X_test)
 sns.scatterplot(y_test,prediction)
 
 # Determine MAE, MSE, and RMSE 
 from sklearn import metrics
 MAE = metrics.mean_absolute_error(y_test,prediction)
 MSE = metrics.mean_squared_error(y_test,prediction)
 RMSE = np.sqrt(MSE)
 
 print(f'MAE: {MAE}\nMSE: {MSE}\nRMSE: {RMSE}')
 
 # Plot histogram of residuals
 res_plot = sns.distplot([y_test - prediction], bins = 45, color = 'grey',
                        hist_kws=dict(alpha = 0.5))
 res_plot.set_xlim([-40,50])
 
 # Observe coefficients of Linear Regression Model to determine what company should focus on 
 df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
 df
