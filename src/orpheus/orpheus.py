#!usr/bin/python3
# encoding: utf-8
import json
import pandas as pd
from typing import Dict, List
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, auc, f1_score



@dataclass
class Orpheus:
    model: DecisionTreeClassifier
    X: pd.DataFrame
    y: pd.Series
    metric: str
    best_params: dict = {}
    history: List[dict] = []

    def fit(self):
        '''
        merge optuna for fitting
        '''

    def view_all_models(self):
        '''
        show history with versions
        '''

    def show_diff(self):
        '''
        show difference between last trial and this time
        '''

    @staticmethod
    def obejective(trial):
        '''
        optuna's fixed usage form
        '''
