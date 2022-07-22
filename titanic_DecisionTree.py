import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

#using pandas to create a DataFrame to organize the data
df = pd.read_csv("titanic_train.csv")
df = df.drop("passenger_id", axis = 'columns')
df = df.drop("name", axis = 'columns')
df = df.drop("ticket", axis = 'columns')
df = df.drop("embarked", axis = 'columns')
df = df.drop("cabin", axis = 'columns')
df = df.drop("boat", axis = 'columns')
df = df.drop("body", axis = 'columns')
df = df.drop("home.dest", axis = 'columns')
df = df.dropna(subset = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare'])
print(df.head)

#creates a new column in the DataFrame with boolean values denoting if certain passengers are male or female
df['male'] = df['sex'] == 'male'
print(df.head)

#defining the feature matrix and creating a NumPy array to hold the data of the DataFrame
x = df[['pclass', 'male', 'age', 'sibsp', 'parch', 'fare']].values
print(x)

#defining the target and creating a NumPy array to hold the data of the DataFrame
y = df['survived'].values
print(y)

#creating the decision tree model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier() #default creation of model sets its impurity to gini

#splitting the dataset into training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 22)

#running the decision tree on the training and test datasets
model.fit(x_train, y_train)

#using the recently created model to predict the survivability of a certain passenger
print(model.predict([[3, True, 38.0, 0, 0, 8.6625]]))

#printing the accuracy, precision, and recall score of the model
from sklearn.metrics import accuracy_score, precision_score, recall_score
print("accuracy: ", model.score(x_test, y_test))
y_pred = model.predict(x_test)
print("precision: ", precision_score(y_test, y_pred))
print("recall: ", recall_score(y_test, y_pred))

#comparing a decision tree with gini impurity with a decision tree of entropy impurity using k-fold cross validation
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5, shuffle = True)
for criterion in ['gini', 'entropy']:
  print("Decision Tree - {}".format(criterion))
  accuracy = []
  precision = []
  recall = []
  for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    new_model = DecisionTreeClassifier(criterion = 'entropy') #creating another decision tree model based on entropy impurity
    new_model.fit(x_train, y_train)
    y_pred = new_model.predict(x_test)
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    print("accuracy: ", np.mean(accuracy))
    print("precision: ", np.mean(precision))
    print("recall: ", np.mean(recall))

#creating another decision tree with features of only passenger class and male/female
feature_names = ['pclass', 'male']
new_x = df[feature_names].values
new_y = df['survived'].values
modelThree = DecisionTreeClassifier()
modelThree.fit(new_x, new_y)

#creating an image to visualize the decision tree
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image
dot_file = export_graphviz(modelThree, feature_names = feature_names)
graph = graphviz.Source(dot_file)
graph.render(filename = 'modelThreeTree', format = 'png', cleanup = 'True')

#conducting pre-pruning on the decision tree to prevent it from overfitting
pruned_decision_tree = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 2, max_leaf_nodes = 10)
pruned_decision_tree.fit(new_x, new_y)

#looping through values on the decision tree using grid search
from sklearn.model_selection import GridSearchCV
param_grid = {
              'max-depth': depth,
              'min_samples_leaf': leaves,
              'max_leaf_nodes': [10, 20, 35, 50]}
modelFour = DecisionTreeClassifier()
gs = GridSearchCV(modelFour, param_grid, scoring = 'f1', cv = 5)
gs.fit(new_x, new_y)
print("best params: ", gs.best_params_)
print("best score: ", gs.best_score_)