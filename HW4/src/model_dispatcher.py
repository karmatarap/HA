from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    # Linear models
    "ols": linear_model.LinearRegression(),
    "ridge": linear_model.Ridge(),
    "lasso": linear_model.Lasso(),
    # SVM
    "linear_svc": svm.SVC(kernel="linear"),
    "rbf_svc": svm.SVC(kernel="rbf"),
    # Ensemble methods
    "rf": RandomForestRegressor(),
    "boost": GradientBoostingRegressor(),
}
