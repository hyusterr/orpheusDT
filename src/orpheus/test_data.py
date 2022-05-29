#!usr/bin/python3
# encoding: utf-8
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn_json as skljson

# data path
DATA_NAME = '../../data/WFH_WFO_dataset.csv'

# features and target name
df = pd.read_csv(DATA_NAME)
target = 'Target'
features = list(df.columns); features.remove(target)
df_x = pd.get_dummies(df[features].copy())
FEATURE_NAMES = df_x.columns

# simulate 2 version data
DF_VER1 = df.iloc[:int(len(df)/2), :].copy()
X_VER1, Y_VER1 = df_x[], 
X_VER1 = pd.get_dummies(X_VER1)
DF_VER2 = df.copy()
X_VER2, Y_VER2 = DF_VER2[features], DF_VER2[target]
X_VER2 = pd.get_dummies(X_VER2)


# simulate 2 version model
TREE_VER1 = DecisionTreeClassifier().fit(X_VER1, Y_VER1)
TREE_VER2 = DecisionTreeClassifier().fit(X_VER2, Y_VER2)

skljson.to_json(TREE_VER1, 'tree1.json')
skljson.to_json(TREE_VER2, 'tree2.json')

print(tree.export_text(TREE_VER1).split('\n'))
