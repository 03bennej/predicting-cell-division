from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from eipy.ei import EnsembleIntegration

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

meta_predictors = {
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
    EI = EnsembleIntegration(base_predictors=base_predictors,
                            meta_models=meta_predictors,
                            k_outer=10,
                            k_inner=10,
                            n_samples=10,
                            sampling_strategy="undersampling",
                            sampling_aggregation="mean",
                            n_jobs=-1,  # set as -1 to use all available CPUs
                            random_state=42,
                            parallel_backend="loky",
                            project_name="cell-division",
                            additional_ensemble_methods = ["Mean", "CES"],
                            model_building=model_building,
                            )
    return EI
