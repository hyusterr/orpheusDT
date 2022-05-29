from orpheus import Orpheus
from sklearn.ensemble import RandomForestClassifier
import numpy as np
if __name__ == "__main__":
    orpheus = Orpheus("Tom")
    data_name = "data_v1"
    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    model_name = "model_v1"
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0).fit(X, Y)

    orpheus.train(data_name, X, Y, model_name, model)
    print(orpheus.score)