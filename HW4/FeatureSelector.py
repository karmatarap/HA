import itertools
import pandas as pd
from typing import Callable, Union


class FeatureSelector:
    """Class to contain feature selection algorithms
    To use this class, models should be defined
    as a function and return a dict containing:
    Error, Model and Params
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
