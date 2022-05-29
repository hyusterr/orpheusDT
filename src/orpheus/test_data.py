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
FEATURE_NAMES = list(df_x.columns)

# simulate 2 version data
N_ver1 = int(len(df)/2)
X_VER1, Y_VER1 = df_x.iloc[:N_ver1], df[target].iloc[:N_ver1].copy() 
X_VER2, Y_VER2 = df_x.copy(), df[target].copy()

# simulate 2 version model
TREE_VER1 = DecisionTreeClassifier().fit(X_VER1, Y_VER1)
TREE_VER2 = DecisionTreeClassifier().fit(X_VER2, Y_VER2)

skljson.to_json(TREE_VER1, 'tree1.json')
skljson.to_json(TREE_VER2, 'tree2.json')

print(tree.export_text(TREE_VER1).split('\n'))
