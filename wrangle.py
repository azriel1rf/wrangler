#%%
import pandas as pd

#%%
data = pd.read_csv("./input_data/bank/train.csv", sep=",", header=0, quotechar="\"")

#%%
#%%
data


#%%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC


#%%
numeric_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
numeric_transformer = StandardScaler()

categorical_features = ['job','marital', 'default', 'housing', 'loan', 'contact']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

#%%
clf = Pipeline(steps=[('preprocessor', preprocessor),
    ('classifier', SVC())])

#%%
classifier = 

#%%
y = data.y
X = data.drop(columns={'id', 'y'})

#%%
clf.fit(X, y)


#%%
clf.predict(X)

#%%

#%%

#%%
#%%
