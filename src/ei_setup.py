from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from eipy.ei import EnsembleIntegration
from eipy.additional_ensembles import MeanAggregation, CES
from eipy.metrics import fmax_score

base_predictors = {
                    'AdaBoost': AdaBoostClassifier(),
                    'DT': DecisionTreeClassifier(),
                    'GradientBoosting': GradientBoostingClassifier(),
                    'KNN': KNeighborsClassifier(),
                    'LR': LogisticRegression(),
                    'NB': GaussianNB(),
                    'MLP': MLPClassifier(),
                    'RF': RandomForestClassifier(),
                    'SVM': SVC(probability=True), 
                    'XGB': XGBClassifier()
                        }

ensemble_predictors = {
                'Mean' : MeanAggregation(),
                'CES' : CES(scoring=lambda y_test, y_pred: fmax_score(y_test, y_pred)[0]),
                'AdaBoost': AdaBoostClassifier(),
                'DT': DecisionTreeClassifier(max_depth=1),
                'GradientBoosting': GradientBoostingClassifier(),
                'KNN': KNeighborsClassifier(),
                'LR': LogisticRegression(),
                'NB': GaussianNB(),
                'MLP': MLPClassifier(alpha=1),
                'RF': RandomForestClassifier(max_depth=1),
                'SVM': SVC(probability=True, C=0.01),
                'XGB': XGBClassifier()
                }

def initiate_EI(model_building=False):
    EI = EnsembleIntegration(
                            base_predictors=base_predictors,
                            ensemble_predictors=ensemble_predictors,
                            k_outer=10,
                            k_inner=10,
                            n_samples=10,
                            sampling_strategy="undersampling",
                            sampling_aggregation="mean",
                            n_jobs=-5,
                            random_state=42,
                            parallel_backend="loky",
                            project_name="cell-division",
                            model_building=model_building,
                            )
    return EI
