#!usr/bin/python3
# encoding: utf-8

# yus's test

import pandas as pd
from orpheus import Orpheus
from sklearn.tree import DecisionTreeClassifier



if __name__ == "__main__":

    # data preparation
    data_name = 'WFH_WFO_dataset_v9'

    data_path = './WFH_WFO_dataset.csv'
    target = 'Target'

    df = pd.read_csv(data_path)
    features = list(df.columns)
    features.remove(target)
    X, y = pd.get_dummies(df[features]), df[target]
    arow = X.loc[0].to_frame().T


    # model preparation
    model_name = "dfc_v4"
    model = DecisionTreeClassifier()


    # instance our class with your task name and your name
    orpheus_instance = Orpheus(
        "WFH_WFO",
        "Alex",
    )

    ### There are 4 training mode
    ### 1. new data + new model
    # orpheus_instance.train(data_name, X, y, model_name, model)
    ### 2. new data + existed model
    # orpheus_instance.train(data_name, X, y, model_name)
    ### 3. existed data + new model
    # orpheus_instance.train(data_name, model_name, model)
    ### 4. existed data + existed model
    # orpheus_instance.train(data_name, model_name)

    ### Show the difference of last trial and current with plot
    # orpheus_instance.show_diff(data_name)

    ### Based on one existing data, view all the model that had trained on it
    # orpheus_instance.view_all_model(data_name)

    ### View all the data saved in database
    # orpheus_instance.view_all_data()


    ### Delete specified data or set data_name to "all" to delete all data
    # orpheus_instance.view_all_data()
    # orpheus_instance.delete_data('./WFH_WFO_dataset.csv')
    # orpheus_instance.view_all_data()


    ### fixed input, show different models' prediction
    # orpheus_instance.view_models_with_input(arow, ["dfc_v4", "dfc_v3"])

    ### Save data to database
    # orpheus_instance.save_Data_to_DB(data_name, X, y)

    ### Restore database by snapshot
    orpheus_instance.restore_DB()







# def dbtest():
#
#
#     client = MongoClient(
#         host='localhost',
#         port=27017,
#         # username=username,
#         # password=password,
#         # authSource=self.dbname,
#         # authMechanism='SCRAM-SHA-256'
#     )
#     dbname = 'easyTest'
#     db = Database(client, name=dbname)
#     data_name = "easyData_v10"
#     model_name = "dfc_v2"
#     current_collection = db["metadata_collection"]
#     # current_collection = db["data_collection"]
#
#     filterQ = {"data_name": data_name, "model_name": model_name}
#     projectionQ = {"_id": False, "meta_dict": {"evaluation_score": 1}}
#
#     cursor = current_collection.find(filterQ, projectionQ)
#
#     for i in cursor:
#         print(i)
#
# def orpheusTest():
#     orpheus = Orpheus("easyTest", "tom", "123")
#     data_name = "easyData"
#     X = [[0, 0], [1, 1]]
#     Y = [0, 1]
#     model_name = "dfc_v2"
#     model = DecisionTreeClassifier()
#
#     # orpheus.view_all_data()
#     #
#     # orpheus.delete_data("all")
#     # orpheus.view_all_data()
#     orpheus.select_model_with_score_above(data_name, 0.1)
#     # orpheus.save_Data_to_DB(data_name, X, Y)
#     # orpheus.train(data_name, model_name, model)
#     # orpheus.view_all_data()
#     # orpheus.view_all_model("easyData")



