from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor ,AdaBoostRegressor
from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pickle

class ML_Regression():
    def __init__(self, score="MAE"):
        self.score = score
        params_Linear = {}
        self.GridSearch_Linear_Regression = GridSearchCV(LinearRegression(), param_grid=params_Linear, cv=2, n_jobs=-1, verbose=5)

        params_svr = {
            'C': [1, 1.5 ,5, 10, 15, 20],
            'kernel': ['linear', "poly", "rbf", "sigmoid"],
            'gamma': ['auto', "scale"]
        }
        self.GridSearch_svr_Regressor = GridSearchCV(SVR(), param_grid=params_svr, cv=2, n_jobs=-1, verbose=5)

        params_DT = {
            'criterion': ['squared_error', "absolute_error"],
            'max_depth': [1, 2, 3, 4, 5, 8, 12]
        }
        self.GridSearch_DT_Regressor = GridSearchCV(DecisionTreeRegressor(), param_grid=params_DT, cv=2, n_jobs=-1, verbose=5)

        params_RF = {
            'n_estimators': [100, 200, 300],
            'criterion': ['squared_error', "absolute_error"],
            'max_depth': [1, 2, 3, 4, 5, 8, 12]
        }
        self.GridSearch_Random_Forest_Regressor = GridSearchCV(RandomForestRegressor(), param_grid=params_RF, cv=2, n_jobs=-1, verbose=5)

        params_AB = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [1.0, 0.1, 0.01, 0.5, 0.7, 0.05],
            'loss': ["linear", "square", "exponential"]
        }
        self.GridSearch_AdaBoost_Regressor = GridSearchCV(AdaBoostRegressor(), param_grid=params_AB, cv=2, n_jobs=-1, verbose=5)

        params_GB = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [1.0, 0.1, 0.01, 0.5, 0.7, 0.05],
            'loss': ["squared_error", "absolute_error"],
            'criterion': ["friedman_mse", "squared_error"]
        }
        self.GridSearch_GBoost_Regressor = GridSearchCV(GradientBoostingRegressor(), param_grid=params_GB, cv=2, n_jobs=-1, verbose=5)

        params_catboost = {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'eval_metric': ["MAE"]
        }
        self.GridSearch_CatBoost_Regressor = GridSearchCV(CatBoostRegressor(), param_grid=params_catboost, cv=2, n_jobs=-1, verbose=5)


        params_xgboost = {
            'learning_rate': [0.1, 0.01, 0.5, 0.7, 0.05]
        }
        self.GridSearch_XGBoost_Regressor = GridSearchCV(XGBRFRegressor(), param_grid=params_xgboost, cv=2, n_jobs=-1, verbose=5)


    def fit_data(self,X_train,Y_train,X_test,Y_test):
        linear = self.GridSearch_Linear_Regression.fit(X_train,Y_train)
        svr = self.GridSearch_svr_Regressor.fit(X_train,Y_train)
        dt = self.GridSearch_DT_Regressor.fit(X_train,Y_train)
        random_forest = self.GridSearch_Random_Forest_Regressor.fit(X_train,Y_train)
        GBoost = self.GridSearch_GBoost_Regressor.fit(X_train,Y_train)
        AdaBoost = self.GridSearch_AdaBoost_Regressor.fit(X_train,Y_train)
        CatBoost = self.GridSearch_CatBoost_Regressor.fit(X_train,Y_train)
        XGBoost = self.GridSearch_XGBoost_Regressor.fit(X_train,Y_train)

        linear_predict = linear.predict(X_test)
        svr_predict = svr.predict(X_test)
        dt_predict = dt.predict(X_test)
        rf_predict = random_forest.predict(X_test)
        gboost_predict = GBoost.predict(X_test)
        adaboost_predict = AdaBoost.predict(X_test)
        catboost_predict = CatBoost.predict(X_test)
        xgboost_predict = XGBoost.predict(X_test)

        if self.score == "MAE":
            result = {
                linear: metrics.mean_absolute_error(Y_test,linear_predict),
                svr: metrics.mean_absolute_error(Y_test,svr_predict),
                dt: metrics.mean_absolute_error(Y_test,dt_predict),
                random_forest: metrics.mean_absolute_error(Y_test,rf_predict),
                GBoost: metrics.mean_absolute_error(Y_test,gboost_predict),
                AdaBoost: metrics.mean_absolute_error(Y_test,adaboost_predict),
                CatBoost: metrics.mean_absolute_error(Y_test,catboost_predict),
                XGBoost: metrics.mean_absolute_error(Y_test,xgboost_predict)
            }
            self.best_model = min(result, key=lambda x: result[x])
            min_loss = min(result.values())

        elif self.score == "MSE":
            result = {
                linear: metrics.mean_squared_error(Y_test,linear_predict),
                svr: metrics.mean_squared_error(Y_test,svr_predict),
                dt: metrics.mean_squared_error(Y_test,dt_predict),
                random_forest: metrics.mean_squared_error(Y_test,rf_predict),
                GBoost: metrics.mean_squared_error(Y_test,gboost_predict),
                AdaBoost: metrics.mean_squared_error(Y_test,adaboost_predict),
                CatBoost: metrics.mean_squared_error(Y_test,catboost_predict),
                XGBoost: metrics.mean_squared_error(Y_test,xgboost_predict)
            }
            self.best_model = min(result, key=lambda x: result[x])
            min_loss = min(result.values())

        print(f"Best Model: {self.best_model.best_estimator_}")
        print(f"Accuracy: {min_loss}")
        return self.best_model
    
    
    def predict_data(self,X):
        predicted = self.best_model.predict(X)
        return predicted
    

    def save_model(self):
        with open("best_model.h5","wb") as f:
            pickle.dump(self.best_model,f)
        
    
def load_model(model):
    with open(f"{model}","rb") as f:
        model = pickle.load(f)
        