#!usr/bin/python3
# encoding: utf-8
import json
import optuna
# lol: optuna internally support storage like sqlite
import pandas as pd
import sklearn_json as skljson
import matplotlib.pyplot as plt
from typing import Dict, List, Union
from dataclasses import dataclass, field
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, f1_score
from sklearn.model_selection import cross_val_score

from diff_side_by_side import side_by_side

@dataclass
class Orpheus:
    data_name: str
    model: RandomForestClassifier
    X: pd.DataFrame
    y: pd.Series
    metric: Union[accuracy_score, auc, f1_score] = accuracy_score
    best_params: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    # TODO: is there a y_test set? 

    def fit(self):
        '''
        merge optuna for fitting
        '''
        # define input
        X = self.X
        Y = self.y
        clf = self.model.fit(X, y)
        
        # optuna
        objective = self.create_objective(clf, X, y)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        print(study.best_trial)
        

    def view_all_models(self):
        '''
        show history with versions
        TODO: query the DB and get all historic data
        TODO: show something like loss curve?
        '''

        
    @staticmethod
    def show_diff():#self):
        '''
        show difference between last trial and this time
        TODO: show data and visualization
        '''
        # query the model information last time
        last_time_model = skljson.from_json('tree1.json')
        # the model this time
        this_time_model = skljson.from_json('tree2.json')
        # show model diff
        # text
        # TODO: feature naming problem
        text_model1 = tree.export_text(last_time_model)
        text_model2 = tree.export_text(this_time_model)
        # TODO: now this looks ugly
        print(side_by_side(text_model1, text_model2, as_string=True, width=78))

        # graph: used sklearn built-in here, other options: graphviz, dtreeviz
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        tree.plot_tree(last_time_model, ax=ax1, filled=True)
        tree.plot_tree(this_time_model, ax=ax2, filled=True)
        plt.show()


    @staticmethod
    def create_objective(model, X, y):
        def obejective(trial):
            '''
            optuna's fixed usage form
            TODO: check the saved name in DB
            TODO: how to automatically generate the trial component
            '''

            # trial type support: categorical, int, uniform, loguniform, discrete_uniform
            rf_n_estimators = trial.suggest_int('n_estimators', 10, 1000)
            rf_max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
            classifier_obj = model(n_estimators=rf_n_estimators, max_depth=rf_max_depth)
            # TODO: merge into class template
            score = cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=5)
            accuracy = score.mean()

            return accuracy
        return obejective
