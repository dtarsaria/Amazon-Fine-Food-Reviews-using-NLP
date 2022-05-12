import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


df = pd.read_csv("bike_crash.csv")

df.head()


df1 = df
df1.replace('No Data',np.nan,inplace = True)
df1.replace('Unknown',np.nan,inplace = True)
df1.replace('None',np.nan,inplace = True)
df1.isnull().sum()

df.drop(['Average Daily Traffic Amount','Highway System'],inplace = True,axis = 1)

df.isnull().sum()

df.fillna()

duplicate_data = df.duplicated()
duplicate_data.sum()
df.drop_duplicates(inplace = True)

df1 = df.loc[:,df.dtypes==np.object]
df1

df1["Crash Severity"].unique()

df1["Crash Severity"].value_counts()

df['Speed Limit'].value_counts()

df.boxplot(column = "Crash Time")

df.boxplot(column = "Crash Year")

df.groupby('Crash Severity').mean()

df.groupby('Day of Week').mean()

df.groupby('Light Condition').mean()

cors = df.corr()
cors

df.head()

df.info()

df.head()

encode_Data =pd.get_dummies(df,columns =['Day of Week',
                                           'Intersection Related',
                                          'Light Condition',
                                          'Road Class',
                                          'Roadway Part',
                                          'Surface Condition',
                                          'Traffic Control Type',
                                          'Weather Condition',
                                          'Person Helmet'])
encode_Data.columns

encode_Data.drop(['Light Condition_Dark, Unknown Lighting',
                  "Traffic Control Type_Other (Explain In Narrative)",
                  "Surface Condition_Other (Explain In Narrative)",
                  "Person Helmet_Unknown If Worn",
                  'Weather Condition_Other (Explain In Narrative)',
                  'Road Class_Other Roads',
                  'Roadway Part_Other (Explain In Narrative)'
                 ],axis = 1,inplace= True)

from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

encode_Data["$1000 Damage to Any One Person's Property"] = encoder.fit_transform(encode_Data["$1000 Damage to Any One Person's Property"])
encode_Data['Active School Zone Flag'] = encoder.fit_transform(encode_Data['Active School Zone Flag'])
encode_Data['At Intersection Flag'] = encoder.fit_transform(encode_Data['At Intersection Flag'])
encode_Data['Construction Zone Flag'] = encoder.fit_transform(encode_Data['Construction Zone Flag'])

encode_Data['Speed Limit'].unique()

cors = encode_Data.corr().abs()
cors

upper_tri = cors.where(np.triu(np.ones(cors.shape),k=1).astype(np.bool))
feature_col = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]

feature_col

sns.heatmap(cors)

from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(encode_Data.loc[:,encode_Data.columns!='Crash Severity'],
                                                encode_Data['Crash Severity'], test_size = 0.2,random_state = 4)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42,max_depth=4)   
rf_model.fit(X_train,y_train)

#Accuracy

yrf_predict = rf_model.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
round(accuracy_score(y_test,yrf_predict)* 100,1) 

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, yrf_predict))
print(classification_report(y_test, yrf_predict))

#Feature Importance
importances = rf_model.feature_importances_

indices = np.argsort(importances)

fig, ax = plt.subplots()
ax.barh(range(len(importances)), importances[indices])
ax.set_yticks(range(len(importances)))
_ = ax.set_yticklabels(np.array(X_train.columns)[indices])

from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(random_state=42,max_depth=4)   
gb_model.fit(X_train,y_train)


ygb_predict = gb_model.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
round(accuracy_score(y_test,ygb_predict) * 100,1)

#Confusion Matrix

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, ygb_predict))
print(classification_report(y_test, ygb_predict))

import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
model = xgb.XGBClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
round(accuracy_score(y_test,y_pred)* 100,1) 
