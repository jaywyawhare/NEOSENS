from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from typing import Dict, Union


def create_sklearn_models(
    y_dim: int,
) -> Dict[str, Union[LinearRegression, MultiOutputRegressor]]:
    """
    Create a dictionary of sklearn models.

    Args:
        y_dim (int): Output dimension of the target variable.

    Returns:
        Dict[str, Union[LinearRegression, MultiOutputRegressor]]: Dictionary of models.
    """
    models = {}
    lr = LinearRegression()
    svr = SVR()
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    knn = KNeighborsRegressor(n_neighbors=5)
    if y_dim > 1:
        models["Linear Regression"] = MultiOutputRegressor(lr)
        models["SVR"] = MultiOutputRegressor(svr)
        models["Random Forest"] = MultiOutputRegressor(rf)
        models["KNN"] = MultiOutputRegressor(knn)
    else:
        models["Linear Regression"] = lr
        models["SVR"] = svr
        models["Random Forest"] = rf
        models["KNN"] = knn
    return models
