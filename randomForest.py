import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#import data
data = pd.read_csv(r"C:\Users\amart\OneDrive\Documents\Projects\Practice\RandomForestModel\heart_cleveland_upload.csv")

#checking for missing values
#print(data.head())
#print(data.isnull().sum())


#extract target variable
y = data['condition']

#drop target variable from training data
x = data.drop(['condition'], axis = 1 )

#split data into train/test
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state= 101)

#create model
rfModel = RandomForestClassifier(oob_score=True)    #oob score to get out of bag accuracy
rfModel.fit(X_train,y_train)                        #train model on train data

print (f"Train Accuracy: {rfModel.score(X_train,y_train):.3f}")
print (f"Test Accuracy: {rfModel.score(X_test,y_test):.3f}")          
