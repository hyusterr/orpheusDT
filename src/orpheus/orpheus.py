#!usr/bin/python3
# encoding: utf-8
import json
import optuna
# lol: optuna internally support storage like sqlite
import pandas as pd
import sklearn_json as skljson
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Union
from dataclasses import dataclass, field
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, f1_score
from sklearn.model_selection import cross_val_score

from diff_side_by_side import side_by_side
from test_data import *

@dataclass
class Orpheus:
    data_name: str
    model: DecisionTreeClassifier
    X: pd.DataFrame
    y: pd.Series
    # metric: Union[accuracy_score, auc, f1_score] = accuracy_score 
    # for simplicy, use acc. as default
    best_params: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    # TODO: is there a y_test set? yes
    # TODO: implement training loss curve

    def fit(self):
        '''
        merge optuna for fitting
        '''
        # define input
        X = self.X
        Y = self.y
        clf = self.model
        
        # optuna
        objective = self.create_objective(clf, X, Y)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        model = type(clf)(**study.best_trial.params).fit(X, Y)

        # record loss
        self.loss = model.tree_.impurity

        return model, datetime.now()


    # TODO: discuss the usage scenario, e.g. do we need to restirct calling situation like should call fit beforehead?
    def show_loss_curve(self):

        plt.plot(self.loss)
        plt.show()
        

    def view_all_models(self, input_row: pd.Series):
        '''
        show history with versions
        TODO: query the DB and get all historic data
        TODO: show something like loss curve? e.g. gini
        result: fixed input, show different models' prediction
        '''
        # query all model from DB
        models = [TREE_VER1, TREE_VER2]
        results = []
        for model in models:
            results.append(model.predict_proba(input_row))
        # TODO: need visiualization here?

        return results


    def show_diff(self):
        '''
        show difference between last trial and this time
        TODO: show data and visualization
        '''
        # query the model information last time
        last_time_model = skljson.from_json('tree1.json') # sklearn.DecisionTreeClassifier
        # the model this time
        this_time_model = skljson.from_json('tree2.json')
        # show model diff
        # text
        text_model1_list = tree.export_text(last_time_model, feature_names=FEATURE_NAMES).split('\n')
        text_model2_list = tree.export_text(this_time_model, feature_names=FEATURE_NAMES).split('\n')
        print(side_by_side(text_model1_list, text_model2_list, as_string=True,
                           left_title='old', right_title='new', width=100))

        # graph: used sklearn built-in here, other options: graphviz, dtreeviz
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        tree.plot_tree(last_time_model, ax=ax1, filled=True, feature_names=FEATURE_NAMES)
        tree.plot_tree(this_time_model, ax=ax2, filled=True, feature_names=FEATURE_NAMES)
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
            # the function's components are fixed so far, left automatically generate options in the future work
            criterion_options = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            splitter_options = trial.suggest_categorical('splitter', ['best', 'random'])
            # min_sample_split can be float or int, here we implement int
            min_samples_split_options = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf_options = trial.suggest_int('min_samples_leaf', 2, 10)
            # n_estimators_options = trial.suggest_int('n_estimators', 10, 1000)
            max_depth_options = trial.suggest_int('max_depth', 2, 32, log=True)

            classifier_obj = type(model)(
                criterion=criterion_options,
                splitter=splitter_options,
                min_samples_split=min_samples_split_options,
                min_samples_leaf=min_samples_leaf_options,
                # n_estimators=n_estimators_options,
                max_depth=max_depth_options,
            )
            score = cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=5)
            accuracy = score.mean()

            return accuracy

        return obejective
