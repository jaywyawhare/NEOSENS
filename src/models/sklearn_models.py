from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

def create_sklearn_models(y_dim):
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

