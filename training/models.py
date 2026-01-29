from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_models_and_grids():
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000),
            {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            }
        ),

        "random_forest": (
            RandomForestClassifier(),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        ),

        "gradient_boosting": (
            GradientBoostingClassifier(),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }
        ),

        "svm": (
            SVC(probability=True),
            {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            }
        ),

        "xgboost": (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5, 7]
            }
        )
    }