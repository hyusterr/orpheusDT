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
from dbmanager import DatabaseManager
import click

class Orpheus:
    def __init__(
        self,
        dbname, # task name
        username,
        password

    ):
        self.dbname = dbname
        self.username = username
        self.password = password
        self.database_manager = DatabaseManager(self.dbname, self.username, self.password)
        self.client = self.database_manager.client


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
        db = self.database_manager.db

        pass

    def extract_model_fromDB(self, model_name):
        pass


    def fit(self):
        trained_model = self.model.fit(self.X, self.Y)
        score = trained_model.score(self.X, self.Y)
        ct = datetime.datetime.now()
        return score, ct


    def save_Data_to_DB(self):

        #is it ok to store data like this ?
        data_dict = {
            "data_name": self.data_name,
            "X": self.X,
            "Y": self.Y
        }
        data_collection = 'data_collection'
        self.database_manager.insert_document(data_collection, data_dict)

        click.secho(f'The new data named {self.data_name} had been saved to database', fg='green')


    def save_metaData_to_DB(self):
        temp_json_file = "./model.json"
        skljson.to_json(self.model, temp_json_file)

        if os.path.exists(temp_json_file):
            model_dict = json.load(temp_json_file)
            os.remove(temp_json_file)

        tags = self.model._get_tags()
        tags.pop('preserves_dtype', None)
        meta_dict = {
            "evaluation score": self.score,
            "scikit_learn_version": sklearn.__version__,
            "user": self.user,
            "train_timestamp": self.ct,
            "estimator_tags": tags,
        }

        complete_meta_data = {
            "data name": self.data_name,
            "model_name": self.model_name,
            "model_dict": model_dict,
            "meta_dict": meta_dict
        }

        data_collection = 'metadata_collection'
        self.database_manager.insert_document(data_collection, complete_meta_data)

        click.secho(f'This training pair with {self.data_name} and {self.model_name} had been saved to database', fg='green')



