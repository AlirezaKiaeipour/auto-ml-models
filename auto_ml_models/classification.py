from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier ,AdaBoostClassifier
from xgboost import XGBRFClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pickle

class ML_Classifier():
    def __init__(self, score="accuracy"):
        self.score = score
        params_knn = {
            'n_neighbors': [1, 3, 5, 7, 11],
            'weights': ['distance', "uniform"],
            'algorithm': ['auto', "ball_tree", "kd_tree","brute"]
        }
        self.GridSearch_knn_classifier = GridSearchCV(KNeighborsClassifier(), param_grid=params_knn, cv=2, n_jobs=-1, verbose=5)

        params_svc = {
            'C': [0.5, 1, 1.5, 2, 5, 10, 15, 20],
            'kernel': ['linear', "poly", "rbf", "sigmoid"],
            'gamma': ['auto', "scale"]
        }
        self.GridSearch_svc_classifier = GridSearchCV(SVC(), param_grid=params_svc, cv=2, n_jobs=-1, verbose=5)

        params_DT = {
            'criterion': ['gini', "entropy", "log_loss"]
        }
        self.GridSearch_DT_classifier = GridSearchCV(DecisionTreeClassifier(), param_grid=params_DT, cv=2, n_jobs=-1, verbose=5)

        params_RF = {
            'n_estimators': [100, 200, 300],
            'criterion': ['gini', "entropy", "log_loss"]
        }
        self.GridSearch_Random_Forest_classifier = GridSearchCV(RandomForestClassifier(), param_grid=params_RF, cv=2, n_jobs=-1, verbose=5)

        params_AB = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [1.0, 0.1, 0.01, 0.5, 0.7, 0.05],
            'algorithm': ["SAMME", "SAMME.R"]
        }
        self.GridSearch_AdaBoost_classifier = GridSearchCV(AdaBoostClassifier(), param_grid=params_AB, cv=2, n_jobs=-1, verbose=5)

        params_GB = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [1.0, 0.1, 0.01, 0.5, 0.7, 0.05],
            'loss': ["log_loss", "exponential"],
            'criterion': ["friedman_mse", "squared_error"]
        }
        self.GridSearch_GBoost_classifier = GridSearchCV(GradientBoostingClassifier(), param_grid=params_GB, cv=2, n_jobs=-1, verbose=5)

        params_catboost = {
            'iterations': [50, 100, 200, 300],
            'learning_rate': [0.1, 0.01, 0.5, 0.7, 0.05],
            'eval_metric': ["Accuracy"]
        }
        self.GridSearch_CatBoost_classifier = GridSearchCV(CatBoostClassifier(), param_grid=params_catboost, cv=2, n_jobs=-1, verbose=5)


        params_xgboost = {
            'learning_rate': [0.1, 0.01, 0.5, 0.7, 0.05]
        }
        self.GridSearch_XGBoost_classifier = GridSearchCV(XGBRFClassifier(), param_grid=params_xgboost, cv=2, n_jobs=-1, verbose=5)


    def fit_data(self,X_train,Y_train,X_test,Y_test):
        knn = self.GridSearch_knn_classifier.fit(X_train,Y_train)
        svc = self.GridSearch_svc_classifier.fit(X_train,Y_train)
        dt = self.GridSearch_DT_classifier.fit(X_train,Y_train)
        random_forest = self.GridSearch_Random_Forest_classifier.fit(X_train,Y_train)
        GBoost = self.GridSearch_GBoost_classifier.fit(X_train,Y_train)
        AdaBoost = self.GridSearch_AdaBoost_classifier.fit(X_train,Y_train)
        CatBoost = self.GridSearch_CatBoost_classifier.fit(X_train,Y_train)
        XGBoost = self.GridSearch_XGBoost_classifier.fit(X_train,Y_train)

        knn_predict = knn.predict(X_test)
        svc_predict = svc.predict(X_test)
        dt_predict = dt.predict(X_test)
        rf_predict = random_forest.predict(X_test)
        gboost_predict = GBoost.predict(X_test)
        adaboost_predict = AdaBoost.predict(X_test)
        catboost_predict = CatBoost.predict(X_test)
        xgboost_predict = XGBoost.predict(X_test)

        if self.score == "accuracy":
            result = {
                knn: metrics.accuracy_score(Y_test,knn_predict),
                svc: metrics.accuracy_score(Y_test,svc_predict),
                dt: metrics.accuracy_score(Y_test,dt_predict),
                random_forest: metrics.accuracy_score(Y_test,rf_predict),
                GBoost: metrics.accuracy_score(Y_test,gboost_predict),
                AdaBoost: metrics.accuracy_score(Y_test,adaboost_predict),
                CatBoost: metrics.accuracy_score(Y_test,catboost_predict),
                XGBoost: metrics.accuracy_score(Y_test,xgboost_predict),
            }
            self.best_model = max(result, key=lambda x: result[x])
            max_accuracy = max(result.values())

        elif self.score == "precision":
            result = {
                knn: metrics.precision_score(Y_test,knn_predict),
                svc: metrics.precision_score(Y_test,svc_predict),
                dt: metrics.precision_score(Y_test,dt_predict),
                random_forest: metrics.precision_score(Y_test,rf_predict),
                GBoost: metrics.precision_score(Y_test,gboost_predict),
                AdaBoost: metrics.precision_score(Y_test,adaboost_predict),
                CatBoost: metrics.precision_score(Y_test,catboost_predict),
                XGBoost: metrics.precision_score(Y_test,xgboost_predict)
            }
            self.best_model = max(result, key=lambda x: result[x])
            max_accuracy = max(result.values())

        elif self.score == "recall":
            result = {
                knn: metrics.recall_score(Y_test,knn_predict),
                svc: metrics.recall_score(Y_test,svc_predict),
                dt: metrics.recall_score(Y_test,dt_predict),
                random_forest: metrics.recall_score(Y_test,rf_predict),
                GBoost: metrics.recall_score(Y_test,gboost_predict),
                AdaBoost: metrics.recall_score(Y_test,adaboost_predict),
                CatBoost: metrics.recall_score(Y_test,catboost_predict),
                XGBoost: metrics.recall_score(Y_test,xgboost_predict)
            }
            self.best_model = max(result, key=lambda x: result[x])
            max_accuracy = max(result.values())

        elif self.score == "f1":
            result = {
                knn: metrics.f1_score(Y_test,knn_predict),
                svc: metrics.f1_score(Y_test,svc_predict),
                dt: metrics.f1_score(Y_test,dt_predict),
                random_forest: metrics.f1_score(Y_test,rf_predict),
                GBoost: metrics.f1_score(Y_test,gboost_predict),
                AdaBoost: metrics.f1_score(Y_test,adaboost_predict),
                CatBoost: metrics.f1_score(Y_test,catboost_predict),
                XGBoost: metrics.f1_score(Y_test,xgboost_predict)
            }
            self.best_model = max(result, key=lambda x: result[x])
            max_accuracy = max(result.values())

        print(f"Best Model: {self.best_model.best_estimator_}")
        print(f"Accuracy: {max_accuracy}")
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
        