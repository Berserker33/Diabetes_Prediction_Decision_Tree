# Importing necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV

# Loading the dataset
df = pd.read_csv(r"C:\Users\Shiva\Downloads\archive (1)\diabetes.csv")

# Exploring the dataset
print(df.head())  # Display the first 5 rows
print(df.tail())  # Display the last 5 rows
print(df.shape)   # Dimensions of the dataset
print(df.size)    # Total number of elements in the dataset
print(df.isnull().sum())  # Checking for missing values

# Dataset Information
df.info()
print(df.columns)  # Column names
print(df['Outcome'].value_counts())  # Count of each outcome category

# Visualizing the Outcome variable distribution using a pie chart
fig = px.pie(
    df,
    names='Outcome',
    color='Outcome',
    color_discrete_map={0: 'skyblue', 1: 'red'}
)
fig.show()

# Proportion of each Outcome category
print(df.Outcome.value_counts(normalize=True))

# Filtering data for Outcome = 1
dfn = df.loc[df["Outcome"] == 1, :]
print(dfn)

# Extracting numeric columns
num = df.select_dtypes(include=[np.number])
print(num.head(3))

# Splitting the data into features (X) and target (y)
x = num.drop(['Outcome'], axis=1)
y = num[['Outcome']]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=120)

# Training a Decision Tree Classifier (without pruning)
dt = DecisionTreeClassifier(criterion='gini')
dt.fit(X_train, y_train)

# Visualizing the decision tree
independent_variable = list(x.columns)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=300)
plot_tree(
    dt,
    feature_names=independent_variable,
    class_names=['No', 'Yes'],
    filled=True,
    node_ids=True,
    fontsize=2
)
plt.show()

# Predictions on the training data
train = pd.concat([X_train, y_train], axis=1)
train['Predicted'] = dt.predict(X_train)

# Confusion matrix and accuracy for training data
matrix = confusion_matrix(train['Predicted'], train['Outcome'])
print(matrix)
accuracy_train = ((354 + 183) / 537) * 100
print(f"Training Accuracy: {accuracy_train}%")

# Classification report for training data
print(classification_report(train['Outcome'], train['Predicted']))

# Controlling overfitting by pruning the decision tree
dt = DecisionTreeClassifier(
    criterion='gini',
    min_samples_split=180,
    min_samples_leaf=100,
    max_depth=5
)
dt.fit(X_train, y_train)

# Visualizing the pruned decision tree
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), dpi=300)
plot_tree(
    dt,
    feature_names=independent_variable,
    class_names=['No', 'Yes'],
    filled=True,
    node_ids=True,
    fontsize=3
)
plt.show()

# Predictions on the training data with pruned tree
train['Predicted'] = dt.predict(X_train)

# Confusion matrix and accuracy for pruned training data
matrix = confusion_matrix(train['Predicted'], train['Outcome'])
print(matrix)
train_accuracy = ((324 + 85) / 537) * 100
print(f"Pruned Training Accuracy: {train_accuracy}%")
print(classification_report(train['Predicted'], train['Outcome']))

# Predictions on the testing data
test = pd.concat([X_test, y_test], axis=1)
test['Predicted'] = dt.predict(X_test)

# Confusion matrix and accuracy for testing data
matrix = confusion_matrix(test['Predicted'], test['Outcome'])
print(matrix)
test_accuracy = ((129 + 37) / 231) * 100
print(f"Test Accuracy: {test_accuracy}%")

# Classification report for testing data
print(classification_report(test['Predicted'], test['Outcome']))

# Grid Search for hyperparameter tuning
params = {
    'min_samples_split': [300, 200, 150, 100],
    'min_samples_leaf': [100, 50, 90],
    'max_depth': [3, 4, 5]
}

grid_search_cv = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    params,
    verbose=1,
    cv=10
)
grid_search_cv.fit(X_train, y_train)

# Displaying the best estimator
print(grid_search_cv.best_estimator_)
