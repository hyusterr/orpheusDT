from datetime import datetime
from typing import TypedDict, List

class ModelSchema(TypedDict):
    # meta
    user: str
    create_time: str

    # hyperparameter
    # model parameter
    # evaluation scores

class DataSchema(TypedDict):
    # meta
    create_time: str