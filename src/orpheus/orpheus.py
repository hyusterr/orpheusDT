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

        self.data_collection = "data_collection"
        self.metadata_collection = "metadata_collection"

    # new data, new model
    @dispatch(str, list, list, str, object)
    def train(self, data_name, data_x, data_y, model_name, model):
        self.data_name = data_name
        self.X = data_x
        self.Y = data_y
        self.model_name = model_name
        self.model = model
        self.score, self.ct = self.fit()

        self.save_Data_to_DB()
        self.save_metaData_to_DB()



    # new data, existing model
    @dispatch(str, list, list, str)
    def train(self, data_name, data_x, data_y, model_name):
        self.data_name = data_name
        self.X = data_x
        self.Y = data_y
        self.model_name = model_name
        self.model = self.extract_model_fromDB(self.model_name)
        self.score, self.ct = self.fit()

        self.save_Data_to_DB()
        self.save_metaData_to_DB()


    # existing data, new model
    @dispatch(str, str, object)
    def train(self, data_name, model_name, model):

        self.data_name = data_name
        self.X, self.Y = self.extract_data_fromDB(self.data_name)
        self.model_name = model_name
        self.model = model
        self.score, self.ct = self.fit()

        self.save_metaData_to_DB()


    # existing data, existing model
    @dispatch(str, str)
    def train(self, data_name, model_name):
        self.data_name = data_name
        self.X, self.Y = self.extract_data_fromDB(self.data_name)

        self.model_name = model_name
        self.model = self.extract_model_fromDB(self.model_name)
        self.score, self.ct = self.fit()

        self.save_metaData_to_DB()


    def extract_data_fromDB(self, data_name):


        cursor = self.database_manager.query_document(self.data_collection, "data_name", data_name)

        if cursor == None:
            print('Data "{}" not found, you may try the "view_all_data()" function'.format(data_name))
        else:
            data_dict = cursor

            X = data_dict['X']
            Y = data_dict['Y']

            click.secho(f'Data "{self.data_name}" had been successfully extracted', fg='green')

            return X, Y



    def extract_model_fromDB(self, model_name):
        cursor = self.database_manager.query_document(self.metadata_collection, "model_name", model_name)

        if cursor == None:
            print('Model "{}" not found'.format(model_name))
        else:
            complete_meta_data_dict = cursor

            model_dict = complete_meta_data_dict['model_dict']
            model = skljson.deserialize_model(model_dict)

            click.secho(f'Model "{self.model_name}" had been successfully extracted', fg='green')

            return model

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
        self.database_manager.insert_document(self.data_collection, data_dict)

        click.secho(f'The new data named "{self.data_name}" had been saved to database', fg='green')


    def save_metaData_to_DB(self):
        temp_json_file = "./model.json"
        skljson.to_json(self.model, temp_json_file)

        if os.path.exists(temp_json_file):
            with open(temp_json_file) as jsonfile:
                model_dict = json.load(jsonfile)
            os.remove(temp_json_file)

        tags = self.model._get_tags()
        tags.pop('preserves_dtype', None)
        meta_dict = {
            "evaluation_score": self.score,
            "scikit_learn_version": sklearn.__version__,
            "user": self.username,
            "train_timestamp": self.ct,
            "estimator_tags": tags,
        }

        complete_meta_data_dict = {
            "data_name": self.data_name,
            "model_name": self.model_name,
            "model_dict": model_dict,
            "meta_dict": meta_dict
        }

        self.database_manager.insert_document(self.metadata_collection, complete_meta_data_dict)

        click.secho(f'This training pair with "{self.data_name}" and "{self.model_name}" had been saved to database', fg='green')


    def view_all_data(self):
        filterQ = {}
        projectionQ = {"data_name": 1, "_id": False}
        num_data = self.database_manager.count_Document(self.data_collection, filterQ)


        if num_data == 0:
            print('There are no datasets under the task "{}" yet'.format(0, self.dbname))
        else:
            print('There are totally {} datasets under the task "{}"'.format(num_data, self.dbname))
            cursor = self.database_manager.custom_query("data_collection", filterQ, projectionQ)
            for data_name in cursor:
                print(data_name)

    def view_all_model(self, data_name):
        filterQ = {"data_name": data_name}
        matchA = {"$match": {"data_name": data_name}}
        projectA = {"$project": {"_id": False,
                                 "model_name": 1,
                                 "evaluation_score": "$meta_dict.evaluation_score"}}
        sortA = {"$sort": {"evaluation_score": -1}}
        num_model = self.database_manager.count_Document(self.metadata_collection, filterQ)


        if num_model == 0:
            print('No trained model found correspond to "{}"'.format(data_name))
        else:
            print('There are totally {} models correspond to "{}"'.format(num_model, data_name))
            cursor = self.database_manager.custom_aggregation(self.metadata_collection, matchA, projectA, sortA)
            for model_info in cursor:
                print(model_info)
