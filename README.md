# orpheusDT
Database 2022 Spring final project.
We propose orpheusDT, a version control module that manage the model and the dataset in machine learning pipe
lines, which also supports automatic snapshot and hyperparameter tuning during training.
Orpheus also utilizes the metadata saved in each training, so that user can observe the difference between
models, list the models sorted by evaluations scores and make recommendation on model choosing. 

## Basic usages
### Create instance 
Specify task and name.
```angular2html
orpheusDT = OrpheusDT(
    "WFH_WFO",
    "Alex"
)
```

### Train
Train with data and model.
Data and model could be either new or existed in database.
```angular2html
# 1. Training with new data + new model
# Data preparation
data_name = 'version3'
X = data_versions[data_name][0]
y = data_versions[data_name][1]
# Model preparation
model_name = "dfc_v1"
model = DecisionTreeClassifier()

# Train
orpheusDT.train(data_name, X, y, model_name, model)
```
```angular2html
# 2. Training with new data + existed model
# Data preparation
data_name = 'version5'
X = data_versions[data_name][0]
y = data_versions[data_name][1]
# Model preparation
model_name = "dfc_v1"
model = DecisionTreeClassifier()

# Train
orpheusDT.train(data_name, X, y, model_name)
```
```angular2html
3. Training with existed data + new model
# Data preparation
data_name = 'version5'
X = data_versions[data_name][0]
y = data_versions[data_name][1]
# Model preparation
model_name = "dfc_v1"
model = DecisionTreeClassifier()

# Train
orpheusDT.train(data_name, X, y, model_name)
```

### Other functions
There are still other functions listed below which had been explained in our demo code.
Please refer demo.ipynb for further information.
- orpheusDT.orpheusDT.show_diff(data_name)
- orpheusDT.view_all_model(data_name)
- orpheusDT.view_all_data()
- orpheus_instance.delete_data(deleted_data_name)
- orpheusDT.save_Data_to_DB(data_name, X, y)
- orpheusDT.restore_DB()
- orpheusDT.model_audition(example_tags)

### Demo video
[link](https://drive.google.com/file/d/1V5n57PIxlIc6vPEpByQOwwKzR19YYXsF/view?usp=sharing)
