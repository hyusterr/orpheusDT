#!usr/bin/python3
# encoding: utf-8
import json
import pandas as pd
from typing import Dict, List
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, auc, f1_score
from sklearn.ensemble import RandomForestClassifier
import sklearn_json as skljson
import sklearn
import pymongo
import json
import datetime
import os

class Orpheus:
    def __init__(
        self,
        data_name: str,
        data_x,
        data_y,
        model: str,
    ):
        self.X = data_x
        self.Y = data_y
        self.data_name = data_name

    def train(self):
        X = self.X
        Y = self.Y

        # TODO X, Y to mongoDB
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0).fit(X, Y)
        score = model.score(X, Y)
        ct = datetime.datetime.now()

        return model, score, ct

    def save_metaData_to_DB(self, model, score, ct):
        temp_json_file = "./model.json"
        skljson.to_json(model, temp_json_file)

        if os.path.exists(temp_json_file):
            model_dict = json.load(temp_json_file)
            os.remove(temp_json_file)


        model_name = type(model).__name__
        id = 1  # not sure yet how to name model

        tags = model._get_tags()
        tags.pop('preserves_dtype', None)
        meta_dict = {
            "data": self.data_name,
            "evaluation score": score,
            "scikit_learn_version": sklearn.__version__,
            "data_creater": "_",
            "data_user": "_",
            "model_creater": "_",
            "model_user": "_",
            "train_timestamp": ct,
            "estimator_tags": tags,
        }

        complete_data = {
            "training id": "{}_{}".format(model_name, id),
            "model_dict": model_dict,
            "meta_dict": meta_dict
        }

        #TODO save complete_data to mongodb

