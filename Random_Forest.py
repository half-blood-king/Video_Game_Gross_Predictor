import numpy as np
import matplotlib. pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
data=pd.read_csv('vgsales.csv')

c=data.columns
data= data.fillna(0)
#Missing Value Handled
values = {'Critic_Score': 0, 'Critic_Count': 0, 'User_Score': 0, 'User_Count': 0}
data['Critic_Score'].fillna(0, inplace=True)
data['Critic_Count'].fillna(0, inplace=True)
data['User_Score'].fillna(0, inplace=True)
data['User_Count'].fillna(0, inplace=True)
data.loc[data['Publisher']=='Nintendo', ['Developer']] = 'Nintendo'
data.loc[data['Publisher']=='Nintendo', ['Rating']] = 'E'
data.loc[data['Publisher']=='Activision', ['Rating']] = 'M'
data.loc[data['Publisher']=='Activision', ['Developer']] = 'Treyarch'
# the user score to 0 where user score is tbd
data.loc[data['User_Score']=='tbd', ['User_Score']] = 0
#convert user score to a np.number
data['User_Score']=data['User_Score'].astype(np.number)
data.head()

#Label Encoding

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df=pd.DataFrame(data=data['Publisher'])
ytarget=data.Global_Sales
features = ['Critic_Score','NA_Sales','EU_Sales']
X=data[features]

X_train, X_test, y_train, y_test = train_test_split(X, ytarget ,test_size = 0.2)

#Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)
regressor.fit(X_train, y_train)
prd=regressor.predict(X_test)
errors = mean_squared_error(y_test, prd)
print(errors)