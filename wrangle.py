#%%
import pandas as pd

#%%
# data = pd.read_csv("../input_data/bank/train.csv", sep=",", header=0, quotechar="\"")
test = pd.read_csv('./input_data/bank/test.csv', sep=",", header=0, quotechar="\"")
#%%
#%%
#%%
data


#%%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import VotingClassifier

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
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', VotingClassifier(estimators=[
        ('svc', SVC()),
        ('lgbm', LGBMClassifier()),
    ]))
])

#%%
clf.fit(X, y)

#%%
#%%
y = data.y
X = data.drop(columns={'id', 'y'})

#%%
prepedX = preprocessor.fit_transform(X)

#%%
prepedX
#%%

#%%
#%%
clf_svc = SVC(probability=True)

#%%
clf_lgbm = LGBMClassifier()


#%%
clf_svc.fit(prepedX, y)

#%%
clf_lgbm.fit(prepedX, y)

#%%
y_svc = clf_svc.predict_proba(prepedX)

#%%
y_lgbm = clf_lgbm.predict_proba(prepedX)

#%%
eclf = VotingClassifier(estimators=[
    ('svc', clf_svc),
    ('lgbm', clf_lgbm),
], voting='soft')

#%%
eclf.fit(prepedX, y)

#%%
y_est = eclf.predict(prepedX)

#%%
test_X = test.drop(columns={'id'})

#%%
preped_test_X = preprocessor.transform(test_X)

#%%
y_est = eclf.predict(preped_test_X)

#%%
submit = test[["id"]].copy()
submit["pred"] = y_est

#%%
submit.to_csv(
    path_or_buf="./submit/submit_voting_20190615.csv", # 出力先
    sep=",",                                            # 区切り文字
    index=False,                                        # indexの出力有無
    header=False                                        # headerの出力有無
)

#%%
