import sys
import os.path
import itertools

import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt
from typing import Callable, Union, Tuple


def create_labels(data: pd.DataFrame):
    "Split data into data and label"
    X = data.iloc[:, 2:]
    # Using label binarizer to encode to binary (B=0; M=1)
    lbl = preprocessing.LabelBinarizer()
    y = lbl.fit_transform(data.iloc[:, 1]).ravel()
    return X, y


class FeatureSelector:
    """ Class to contain feature selection algorithms
        To use this class, models should be defined
        as a function and return a dict containing:
        Error, Model and Params,
        Re-using from HW4
    """

    def __init__(
        self, model: Callable, X: pd.DataFrame, y: pd.Series, minimum=True
    ) -> None:
        self.model = model
        self.X = X
        self.y = y
        self.minimum = minimum

    @staticmethod
    def best_model(results: list, _minimum=True) -> dict:
        """Returns `best` model from a list of model results
        Best is defined as the model with minimum error when _minimum is True
        otherwise, best is the maximum.
        """
        f = min if _minimum else max
        return f(results, key=lambda x: x["Error"])

    def best_subset(self, k: int) -> dict:
        "Returns best model of any k parameters"
        results = [
            self.model(self.X[list(params)], self.y)
            for params in itertools.combinations(self.X.columns, k)
        ]
        return FeatureSelector.best_model(results, _minimum=self.minimum)

    def best_n_subsets(self, n: int) -> list:
        "Returns a list of best subset models for each parameter size upto n"
        return [self.best_subset(k) for k in range(1, n + 1)]

    def _forward_selection(self, predictors: list) -> dict:
        "Inner loop of forward selection, find the next best predictor p"
        remaining_predictors = [p for p in self.X.columns if p not in predictors]
        results = [
            self.model(self.X[list(predictors) + [p]], self.y)
            for p in remaining_predictors
        ]
        return FeatureSelector.best_model(results, _minimum=self.minimum)

    def forward_selection(self, n: Union[bool, int] = None) -> list:
        "Find the best model of n parameters using forward selection"
        models, predictors = [], []
        n = n or len(self.X.columns)
        for _ in range(n):
            model = self._forward_selection(predictors)
            models.append(model)
            predictors = model.params
        return models


class Model:
    """Throwaway class to help reduce boiler plate
    Dict is returned to fit the interface of FeatureSelector
    """

    def __init__(self, model=None, metric=None, **kwargs):
        self._model = model
        self.metric = metric
        self.kwargs = kwargs

    def __call__(self, train_X, train_y, test_X=None, test_y=None):
        "Support separate train and test sets"
        test_X = train_X if test_X is None else test_X
        test_y = train_y if test_y is None else test_y
        m = self._model(**self.kwargs)
        m.fit(train_X, train_y)
        Y_hat = m.predict(test_X)
        error = self.metric(test_y, Y_hat)
        return {"Error": error, "Model": m, "Params": train_X.columns.values}


if __name__ == "__main__":
    data = pd.read_excel("Data/data.xlsx")

    X, Y = create_labels(data)

    # Question 5
    # Define a new logistic model with accuracy
    log_acc = Model(model=linear_model.LogisticRegression, metric=accuracy_score)

    # Find the best 3 feature subset with highest accuracy
    model_best_acc = FeatureSelector(log_acc, X, Y, minimum=False).best_subset(k=3)
    print(
        f"Best accuracy model was: {model_best_acc['Error']} using features: {model_best_acc['Params']}"
    )
    assert round(model_best_acc["Error"], 2) == 0.96  # checkpoint

    # Question 6
    log_rec = Model(model=linear_model.LogisticRegression, metric=recall_score)
    # Return model with highest recall
    log_rec_r = FeatureSelector(log_rec, X, Y, minimum=False)
    # Find the best 2 feature subset
    model_best_recall = log_rec_r.best_subset(k=2)
    print(
        f"Best recall model was: {model_best_recall['Error']} using features: {model_best_recall['Params']}"
    )
    assert round(model_best_recall["Error"], 4) == 0.9198  # checkpoint

    # Question 7
    tree_model = Model(DecisionTreeClassifier, accuracy_score, max_leaf_nodes=3)
    classifier = tree_model(X, Y)
    print(f"Decision Tree accuracy: {classifier['Error']}")
    assert round(classifier["Error"], 4) == 0.9402  # Checkpoint

    fig = plt.figure(figsize=(15, 10))
    _ = plot_tree(
        classifier["Model"],
        feature_names=list(X.columns),
        class_names=["B", "M"],
        rounded=True,
        fontsize=12,
        filled=True,
    )
    # fig.show()

    # for testing, lets just make sure that data with the supplied criteria returns the same label
    test_data = data[
        (data["radius_worst"] < 12)
        & (data["radius_mean"] > 9)
        & (data["concave points_worst"] < 0.1)
    ]

    test_X, test_Y = create_labels(test_data)

    # Here we are training on the original data and testing on the subset data
    # We should get a perfect match (accuracy=1) and the class should always predict "Benign"
    dt_clf = Model(DecisionTreeClassifier, accuracy_score, max_leaf_nodes=3)
    dt_m = dt_clf(X, Y, test_X, test_Y)
    assert dt_m["Error"] == 1
    assert all(test_Y == [0])  # all are benign

    # Question 8
    rf_clf = Model(RandomForestClassifier, accuracy_score, n_estimators=10, max_depth=3)
    rf_m = rf_clf(X, Y)
    print(f"Random forest accuracy: {rf_m['Error']}")
