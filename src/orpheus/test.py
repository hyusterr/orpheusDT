#!usr/bin/python3
# encoding: utf-8

import pandas as pd
from orpheus import Orpheus
from sklearn.tree import DecisionTreeClassifier

data_name = '../../data/WFH_WFO_dataset.csv'
target = 'Target'

df = pd.read_csv(data_name)
features = list(df.columns); features.remove(target)
X, y = df[features], df[target]

# usage for our class
orpheus_instance = Orpheus(
    data_name,
    DecisionTreeClassifier(random_state=0),
    X,
    y,
)
orpheus_instance.train()
orpheus_instance.save_metaData_toDB()
