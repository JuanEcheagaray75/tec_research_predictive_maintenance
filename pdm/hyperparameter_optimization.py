"""XGBoost hyperparameter optimization.

Run TPE to find a better configuration of hyperparameters
for the proposed XGBoost models:
- RUL estimator
- Lower / Upper Bound (Prediction Intervals)
- Median (used for testing and development, not part of the research)

Runs TPE and saves trials and results as pickle files
"""
import pathlib
import pickle
from datetime import datetime
from typing import Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from loss import PHMAP
# Important Paths
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
BASE_MODEL_PATH = pathlib.Path.cwd() / 'model' / 'xgb' / start_time
BASE_MODEL_PATH.mkdir(exist_ok=True)


def preprocess(dtrain: xgb.DMatrix,
               dtest: xgb.DMatrix,
               param: str) -> Tuple[xgb.DMatrix, xgb.DMatrix, str]:
    """Preprocess step for CV using PCA.

    Parameters
    ----------
    dtrain : xgb.DMatrix
        Training data and labels
    dtest : xgb.DMatrix
        Test data and labels
    param : str
        Extra param

    Returns
    -------
    Tuple[xgb.DMatrix, xgb.DMatrix, str]
        Training and testing data, and another parameter
    """
    train_data = np.asarray(dtrain.get_data().todense())
    test_data = np.asarray(dtest.get_data().todense())

    pca = PCA()
    pca.fit(train_data)

    train_m = xgb.DMatrix(
        pca.transform(train_data), dtrain.get_label())
    test_m = xgb.DMatrix(
        pca.transform(test_data), dtest.get_label())

    return train_m, test_m, param


def main() -> None:
    """Run TPE to get best set of hyperparameters for XGB model."""
    models_objectives = {
        'estimator': {'objective': 'reg:squarederror'},
        'lower_bound': {'objective': 'reg:quantileerror',
                        'quantile_alpha': 0.05},
        'upper_bound': {'objective': 'reg:quantileerror',
                        'quantile_alpha': 0.95},
        'median': {'objective': 'reg:quantileerror',
                   'quantile_alpha': 0.5}
        }

    SEARCH_SPACE = {
        'variance_threshold': hp.uniform('variance_threshold', 0.1, 1),
        'num_boost_rounds': hp.uniformint('num_boost_rounds', 100, 10_000),
        'eta': hp.loguniform('eta', -4.5, 0),
        'max_depth': hp.uniformint('max_depth', 2, 30),
        'min_child_weight': hp.uniformint('min_child_weight', 2, 50),
        'subsample': hp.uniform('subsample', 0.01, 1),
        'gamma': hp.uniform('gamma', 0, 100),
        'alpha': hp.uniform('alpha', 0, 100),
        'lambda': hp.uniform('lambda', 0, 100)
    }
    max_rounds = 5

    for config in models_objectives:

        def objective(search_space: dict) -> float:
            """Evaluate parameter config and get loss.

            Parameters
            ----------
            search_space : dict
                Current hyperparameter configuration

            Returns
            -------
            float
                Loss value for XGBoost model with the current hyperparameters
            """
            X = pd.read_csv('data/phmap_dataset.csv').drop(
                labels=['unit_names', 'hs'],
                axis=1)
            y = pd.read_csv('data/ruls.csv')

            var_tresh = VarianceThreshold(
                threshold=search_space['variance_threshold']
                )
            X = var_tresh.fit_transform(X)
            d_train = xgb.DMatrix(X, y)

            params = {
                'booster': 'gbtree',
                'device': 'gpu',
                'tree_method': 'hist',
                # Learning Parameters,
                'eta': search_space['eta'],
                # Tree Hyperparameters
                'max_depth': int(search_space['max_depth']),
                'min_child_weight': int(search_space['min_child_weight']),
                # Stochastic Sampling
                'subsample': search_space['subsample'],
                # Regularization
                'gamma': search_space['gamma'],
                'alpha': search_space['alpha'],
                'lambda': search_space['lambda']
            }
            params.update(models_objectives[config])

            if params['objective'] != 'reg:squarederror':
                loss_name = 'test-quantile-mean'
                custom_metric = None

            else:
                loss_name = 'test-PHMAP-mean'
                custom_metric = PHMAP

            try:
                results = xgb.cv(
                    params=params,
                    dtrain=d_train,
                    nfold=5,
                    custom_metric=custom_metric,
                    seed=7501,
                    num_boost_round=int(search_space['num_boost_rounds']),
                    maximize=False,
                    shuffle=True,
                    early_stopping_rounds=50,
                    fpreproc=preprocess
                )
                loss = results[loss_name].tail(1).item()

            # I know it's bad practice, but this exception is meant
            # to catch XGBoost GPU errors that occur somewhat randomly
            except Exception:
                loss = 9999

            return {'loss': loss, 'status': STATUS_OK}

        MODEL_NAME = BASE_MODEL_PATH / f'{start_time}_{config}'

        trials = Trials()
        best_parameters = fmin(fn=objective,
                               space=SEARCH_SPACE,
                               algo=tpe.suggest,
                               trials=trials,
                               max_evals=max_rounds,
                               rstate=np.random.default_rng(seed=7501))

        print(f'Finished optimizing {config}, saving the best parameters...')
        with open(f'{MODEL_NAME}_parms.pkl', 'wb') as f:
            pickle.dump(space_eval(SEARCH_SPACE, best_parameters), f)

        print('Saving trials')
        with open(f'{MODEL_NAME}_study.pkl', 'wb') as f:
            pickle.dump(trials, f)


if __name__ == '__main__':
    main()
