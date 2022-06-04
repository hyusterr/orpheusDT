#!usr/bin/python3
# encoding: utf-8

# yus's test

import pandas as pd
from orpheus import Orpheus
from sklearn.tree import DecisionTreeClassifier

# orpheus_instance.save_metaData_toDB()
from orpheus import Orpheus
import sklearn_json as skljson
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from pymongo import MongoClient
from pymongo.database import Database
import click
import numpy as np


if __name__ == "__main__":

    # data preparation
    data_name = 'WFH_WFO_dataset.csv'
    target = 'Target'

    df = pd.read_csv(data_name)
    features = list(df.columns)
    features.remove(target)
    X, y = pd.get_dummies(df[features]), df[target]
    arow = X.loc[0].to_frame().T
    # model preparation

    model_name = "dfc_v4"
    model = DecisionTreeClassifier()

    # usage for our class
    orpheus_instance = Orpheus(
        "WFH_WFO",
        "Alex",
    )

    # train
    # orpheus_instance.train(data_name, model_name, model)
    # orpheus_instance.show_diff()
    orpheus_instance.view_all_model(data_name)
    # orpheus_instance.view_all_data()
    # orpheus_instance.delete_data(data_name)
    orpheus_instance.view_models_with_input(arow, ["dfc_v4", "dfc_v3"])
    # orpheus_instance.save_Data_to_DB(data_name, X, y)
    # orpheus_instance.fit()
    # orpheus_instance.show_loss_curve()


### min's test


def dbtest():


    client = MongoClient(
        host='localhost',
        port=27017,
        # username=username,
        # password=password,
        # authSource=self.dbname,
        # authMechanism='SCRAM-SHA-256'
    )
    dbname = 'easyTest'
    db = Database(client, name=dbname)
    data_name = "easyData_v10"
    model_name = "dfc_v2"
    current_collection = db["metadata_collection"]
    # current_collection = db["data_collection"]

    filterQ = {"data_name": data_name, "model_name": model_name}
    projectionQ = {"_id": False, "meta_dict": {"evaluation_score": 1}}

    cursor = current_collection.find(filterQ, projectionQ)

    for i in cursor:
        print(i)

def orpheusTest():
    orpheus = Orpheus("easyTest", "tom", "123")
    data_name = "easyData"
    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    model_name = "dfc_v2"
    model = DecisionTreeClassifier()

    # orpheus.view_all_data()
    #
    # orpheus.delete_data("all")
    # orpheus.view_all_data()
    orpheus.select_model_with_score_above(data_name, 0.1)
    # orpheus.save_Data_to_DB(data_name, X, Y)
    # orpheus.train(data_name, model_name, model)
    # orpheus.view_all_data()
    # orpheus.view_all_model("easyData")



