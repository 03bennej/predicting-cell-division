from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from eipy.ei import EnsembleIntegration

base_predictors = {
                    'XGB': XGBClassifier(),
                    }

def initiate_EI(model_building=False):
    EI = EnsembleIntegration(base_predictors=base_predictors,
                            k_outer=10,
                            k_inner=2,  # not relevant since we only use "base predictors"
                            n_samples=1,
                            sampling_strategy='undersampling',
                            sampling_aggregation='mean',
                            n_jobs=-1,  # set as -1 to use all available CPUs
                            random_state=42,
                            parallel_backend='loky',
                            project_name='cell-division',
                            model_building=model_building,
                            )
    return EI