#Using Hyperparameter Tuning
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#import data
data = pd.read_csv(r"C:\Users\amart\OneDrive\Documents\Projects\Practice\RandomForestModel\heart_cleveland_upload.csv")

'''
checking for missing values
print(data.head())
print(data.isnull().sum())
'''

#extract target variable & drop target variable from training data
y = data['condition']
x = data.drop(['condition'], axis = 1 )
'''
print(f"x: {x.shape}")
'''

#split data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=101)
'''
print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}")
print(f"y_test: {y_test.shape}")
'''

#build random forest with hyper paramters
n_estimators = [int(x) for x in np.linspace(10,80,10)] #number of trees to test
max_features = ['sqrt', 'log2']                        #use log2 of all features / use sqrt of total features 
max_depth = [2,4]                                      #max number of levels in tree
min_samples_split = [2,5]                              #min number of samples required to split a node
min_samples_leaf = [1,2]                               #min number of samples per leaf node
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
'''
print(param_grid)
'''

rfModel = RandomForestClassifier(random_state=101)
rfGrid = GridSearchCV(estimator = rfModel, param_grid=param_grid, cv = 3, verbose = 0, n_jobs = 4)  #test every combo

rfGrid.fit(x_train,y_train)
'''
print(rfGrid.best_params_)
'''

print (f"Train Accuracy: {rfGrid.score(x_train,y_train):.3f}")
print (f"Test Accuracy: {rfGrid.score(x_test,y_test):.3f}")