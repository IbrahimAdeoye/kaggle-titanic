import numpy as np 
import pandas as pd

titanic_train = pd.read_csv('/Users/adeoyedipo/Downloads/titanic-4/train.csv')
titanic_test = pd.read_csv('/Users/adeoyedipo/Downloads/titanic-4/test.csv')
gender_sub =pd.read_csv('/Users/adeoyedipo/Downloads/titanic-4/gender_submission.csv')


titanic_target = titanic_train["Survived"].copy()
titanic_train = titanic_train.drop(["Survived",'Name','Cabin','Ticket'], axis=1)
titanic_train.set_index('PassengerId',inplace=True) #'PassengerId'
titanic_test = titanic_test.drop(['Name','Cabin','Ticket'], axis=1)
titanic_test.set_index('PassengerId',inplace=True)


from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer,make_column_selector

scaling = make_pipeline(SimpleImputer(strategy='most_frequent'),StandardScaler())

age_pipeline = make_pipeline(SimpleImputer(strategy='median'),
                            StandardScaler())
fare_test_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),
                            StandardScaler())
hot_encode = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
     ('scaler',scaling,['Fare','SibSp','Parch']),
    ('age',age_pipeline,['Age']),
    ('cat',hot_encode,['Sex','Embarked'])],remainder='passthrough')




from sklearn.ensemble import RandomForestClassifier
full_pipeline = make_pipeline(
         preprocessing, RandomForestClassifier())

full_pipeline.fit(titanic_train,titanic_target)




full_pipeline.predict(titanic_test)



# evaluate on the test set
full_pipeline.score(titanic_test,gender_sub['Survived'])

