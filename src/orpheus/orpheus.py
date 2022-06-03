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

    @dispatch(str, object, object, str, object)
    def train(self, data_name, data_x, data_y, model_name, model):
        """
        Train with new data and new model
        Save new data and training result to DB
        """

        self.data_name = data_name
        self.X = data_x
        self.Y = data_y
        self.save_Data_to_DB()

        self.model_name = model_name
        self.model = model
        self.score, self.ct = self.fit()

        self.save_metaData_to_DB()



    @dispatch(str, object, object, str)
    def train(self, data_name, data_x, data_y, model_name):
        """
        Train with new data and existing model
        Save new data and training result to DB
        """

        self.data_name = data_name
        self.X = data_x
        self.Y = data_y
        self.save_Data_to_DB()

        self.model_name = model_name
        self.model = self.extract_model_fromDB(self.model_name)
        self.score, self.ct = self.fit()

        self.save_metaData_to_DB()


    @dispatch(str, str, object)
    def train(self, data_name, model_name, model):
        """
        Train with existing data and new model
        Save training result to DB
        """

        self.data_name = data_name
        self.X, self.Y = self.extract_data_fromDB(self.data_name)
        self.model_name = model_name
        self.model = model
        self.score, self.ct = self.fit()

        self.save_metaData_to_DB()


    @dispatch(str, str)
    def train(self, data_name, model_name):
        """
        Train with existing data and existing model
        Save training result to DB
        """
        self.data_name = data_name
        self.X, self.Y = self.extract_data_fromDB(self.data_name)

        self.model_name = model_name
        self.model = self.extract_model_fromDB(self.model_name)
        self.score, self.ct = self.fit()

        self.save_metaData_to_DB()


    def extract_data_fromDB(self, data_name):
        """
        Find the existing data named data_name in DB
        """

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
        """
        Find the existing model named model_name in DB
        """
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
        """
        fit the model according to X, Y
        - will be replaced
        """
        trained_model = self.model.fit(self.X, self.Y)
        score = trained_model.score(self.X, self.Y)
        ct = datetime.datetime.now()
        return score, ct

    @dispatch()
    def save_Data_to_DB(self):
        """
        Automatically insert the data to DB when data_name not exist in DB
        """

        data_dict = {
            "data_name": self.data_name,
            "X": self.X,
            "Y": self.Y
        }

        check_existenceQ = {"data_name": self.data_name}
        dataQ = {"$setOnInsert": data_dict}
        result = self.database_manager.insert_document_if_not_exist(self.data_collection, check_existenceQ, dataQ)



        if result.matched_count == 0:
            click.secho(f'The new data named "{self.data_name}" had been saved to database', fg='green')
        else:
            raise Exception('Data named "{}" already exist, if you wish to use the existed "{}", '
                             'please remove X and Y field'.format(self.data_name, self.data_name))



    @dispatch(str, object, object)
    def save_Data_to_DB(self, data_name, x, y):
        """
        Manually insert the data to DB when data_name not exist in DB
        """
        data_dict = {
            "data_name": data_name,
            "X": x,
            "Y": y
        }

        check_existenceQ = {"data_name": data_name}
        dataQ = {"$setOnInsert": data_dict}

        result = self.database_manager.insert_document_if_not_exist(self.data_collection, check_existenceQ, dataQ)
        if result.matched_count == 0:
            click.secho(f'The new data named "{data_name}" had been saved to database', fg='green')
        else:
            click.secho(f'Insertion failed, Data named "{data_name}" already exist', fg='yellow')



    def save_metaData_to_DB(self):
        """
        Insert training metaData to DB, if the combination of data_name and model_name
        exists already, show warning and extract the evaluation_score
        """
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
        check_existenceQ = {"data_name": self.data_name,  "model_name": self.model_name}
        complete_meta_dataQ = {"$setOnInsert": complete_meta_data_dict}
        result = self.database_manager.insert_document_if_not_exist(self.metadata_collection, check_existenceQ, complete_meta_dataQ)

        if result.matched_count == 0:
            click.secho(f'This training pair with "{self.data_name}" and "{self.model_name}" had been saved to database', fg='green')
        else:
            filterQ = {"data_name": self.data_name,  "model_name": self.model_name}
            projectionQ = {"_id": False, "meta_dict": {"evaluation_score": 1}}
            cursor = self.database_manager.custom_query(self.metadata_collection, filterQ, projectionQ)
            score = cursor[0]['meta_dict']['evaluation_score']
            click.secho(f'Combination Data "{self.data_name}" and Model "{self.model_name}" already exist, '
                        f'with evaluation score = {score}', fg='yellow')


    def view_all_data(self):
        """
        View all the data_name inside DB under current task
        """
        filterQ = {}
        projectionQ = {"_id": False, "data_name": 1}
        num_data = self.database_manager.count_Document(self.data_collection, filterQ)


        if num_data == 0:
            print('There are no data under the task "{}"'.format(self.dbname))
        else:
            print('There are totally {} datasets under the task "{}"'.format(num_data, self.dbname))
            cursor = self.database_manager.custom_query("data_collection", filterQ, projectionQ)
            for data_name in cursor:
                print(data_name)

    def view_all_model(self, data_name):
        """
        View all models trained on "data_name", models are sorted by evaluation score
        """
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

    def delete_data(self, deleted_data_name: str):
        """
        Input one "date_name" to delete correspond data from DB or
        input "all" to delete all data in DB
        """

        if deleted_data_name.lower() == "all":
            filterD = {}
            self.database_manager.delete_Document(self.data_collection, filterD)
            click.secho(f'All the data have been removed', fg='green')

        else:
            filterD = {"data_name": deleted_data_name}
            count = self.database_manager.count_Document(self.data_collection, filterD)
            if count > 0:
                self.database_manager.delete_Document(self.data_collection, filterD)
                click.secho(f'Data {deleted_data_name} have been removed', fg='green')
            else:
                click.secho(f'Deletion failed, Data "{deleted_data_name}" does not exist', fg='red')

    def select_model_with_score_above(self, data_name, min_score):

        matchA = {"$match":
                    {
                        "$and": [
                            {"data_name": data_name},
                            {"meta_dict.evaluation_score": {"$gt": min_score}}
                        ]
                    }
        }

        projectA = {"$project": {"_id": False,
                                 "model_name": 1,
                                 "evaluation_score": "$meta_dict.evaluation_score"}}
        sortA = {"$sort": {"evaluation_score": -1}}
        valid_models = self.database_manager.custom_aggregation(self.metadata_collection, matchA, projectA, sortA)
        valid_models = list(valid_models)
        print(f'There exists {len(valid_models)} models with evaluation score > {min_score} for Data "{data_name}"')

        for model_info in valid_models:
            print(model_info)

# TODO
# Test set not implemneted
# havn't try difficult data
# Username is only used for metadata, not for logging into DB
# metaData need to have more function to make good use of
# show model info