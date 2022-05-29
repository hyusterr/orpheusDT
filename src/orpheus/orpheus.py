#!usr/bin/python3
# encoding: utf-8
import json
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
from argparse import ArgumentError
from multipledispatch import dispatch
import numpy as np

class Orpheus:
    def __init__(
        self,
        user
    ):
        self.user = user


    @dispatch(str, list, list, str, object)
    def train(self, data_name, data_x, data_y, model_name, model):
        self.data_name = data_name
        self.X = data_x
        self.Y = data_y
        self.model_name = model_name
        self.model = model
        self.score, self.ct = self.fit()


    @dispatch(str, list, list, str)
    def train(self, data_name, data_x, data_y, model_name):
        self.data_name = data_name
        self.X = data_x
        self.Y = data_y
        self.model_name = model_name
        self.model = self.extract_model_fromDB(self.model_name)
        self.score, self.ct = self.fit()


    @dispatch(str, str, object)
    def train(self, data_name, model_name, model):

        self.data_name = data_name
        self.X, self.Y = self.extract_data_fromDB(self.data_name)
        self.model_name = model_name
        self.model = model
        self.score, self.ct = self.fit()


    @dispatch(str, str)
    def train(self, data_name, model_name):
        self.data_name = data_name
        self.X, self.Y = self.extract_data_fromDB(self.data_name)

        self.model_name = model_name
        self.model = self.extract_model_fromDB(self.model_name)
        self.score, self.ct = self.fit()


    def extract_data_fromDB(self, data_name):
        pass

    def extract_model_fromDB(self, model_name):
        pass


    def fit(self):
        trained_model = self.model.fit(self.X, self.Y)
        score = trained_model.score(self.X, self.Y)

        ct = datetime.datetime.now()

        return score, ct


    def save_Data_to_DB(self):
        # save array to DB
        pass

    def save_metaData_to_DB(self, model, score, ct):
        temp_json_file = "./model.json"
        skljson.to_json(self.model, temp_json_file)

        if os.path.exists(temp_json_file):
            model_dict = json.load(temp_json_file)
            os.remove(temp_json_file)


        # model_name = type(self.model).__name__

        tags = model._get_tags()
        tags.pop('preserves_dtype', None)
        meta_dict = {
            "data": self.data_name,
            "evaluation score": self.score,
            "scikit_learn_version": sklearn.__version__,
            "user": self.user,
            "train_timestamp": self.ct,
            "estimator_tags": tags,
        }

        complete_data = {
            "model_name": self.model_name ,
            "model_dict": model_dict,
            "meta_dict": meta_dict
        }


        #TODO save complete_data to mongod