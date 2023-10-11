import pathlib
import pickle
from timeit import default_timer as timer
from datetime import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

# Important Paths
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_PATH = pathlib.Path.cwd() / 'model' / 'lgbm' / start_time
MODEL_PATH.mkdir(exist_ok=True)


# RUL Losses
def phmap_loss(y_true, y_pred):

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    diff = y_true - y_pred
    alpha = np.where(diff <= 0, -1/10, 1/13)
    nasa = np.mean(np.exp(alpha * diff) - 1)

    return 0.5*(rmse + nasa)


def phmap_agg_loss(y_pred, train_data):
    labels = train_data.get_label()

    return 'mean_phmap', phmap_loss(labels, y_pred), False


models_objectives = {
    'lower_bound': {'objective': 'quantile', 'alpha': 0.05},
    'upper_bound': {'objective': 'quantile', 'alpha': 0.95},
    'median': {'objective': 'quantile', 'alpha': 0.5},
    'estimator': {'objective': 'regression'}
    }

SEARCH_SPACE = {
    # Objectives and reproducibility
    'task': 'train',
    'boosting_type': 'gbdt',
    'deterministic': True,
    'num_threads': 6,
    'seed': 7501,
    'verbose': -1,
    'force_col_wise': True,
    'extra_trees': True,
    # Processor
    'variance_treshold': hp.uniform('variance_treshold', 0, 0.9),
    # Learning parameters
    'max_bin': hp.uniformint('max_bin', 10, 400),
    'feature_fraction_bynode': hp.uniform('feature_fraction_bynode', 0.5, 1),
    'max_depth': hp.uniformint('max_depth', 2, 15),
    'min_sum_hessian_in_leaf': hp.uniform(
        'min_sum_hessian_in_leaf', 1, 20),
    'feature_fraction': hp.uniform('feature_fraction', 0, 1),
    'bagging_freq': hp.uniformint('bagging_freq', 1, 100),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
    'min_data_in_leaf': hp.uniformint('min_data_in_leaf', 2, 40),
    'early_stopping_round': hp.uniform('early_stopping_round', 0.05, 0.3),
    'n_estimators': hp.uniformint('n_estimators', 100, 10_000),
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.5),
    'num_leaves': hp.uniformint('num_leaves', 10, 64),
    'lambda_l1': hp.uniform('lambda_l1', 0, 100),
    'lambda_l2': hp.uniform('lambda_l2', 0, 100),
    'linear_lambda': hp.uniform('linear_lambda', 0, 100)
}

for config in models_objectives:

    print(f'Fitting {config}')
    if config in ('estimator', 'median'):
        max_rounds = 1_000
    else:
        max_rounds = 1_000

    def objective(search_space):
        X = pd.read_csv('data/phmap_dataset.csv')
        y = pd.read_csv('data/ruls.csv')

        X_train, X_test, y_train, y_test = train_test_split(
            X.drop(['unit_names', 'hs'], axis=1),
            y,
            test_size=0.25,
            random_state=7501)

        pca = PCA()
        pca.fit(X_train)
        with open(MODEL_PATH / 'pca_scaler.pkl', 'wb') as f:
            pickle.dump(pca, f)

        params = {
            # Do not change
            'task': 'train',
            'boosting_type': 'gbdt',
            'deterministic': True,
            'num_threads': 6,
            'seed': 7501,
            'verbose': -1,
            'force_col_wise': True,
            # Learning parameters
            'max_bin': int(search_space['max_bin']),
            'feature_fraction_bynode': search_space['feature_fraction_bynode'],
            'max_depth': int(search_space['max_depth']),
            'min_sum_hessian_in_leaf': search_space['min_sum_hessian_in_leaf'],
            'extra_trees': search_space['extra_trees'],
            'feature_fraction': search_space['feature_fraction'],
            'bagging_freq': int(search_space['bagging_freq']),
            'bagging_fraction': search_space['bagging_fraction'],
            'min_data_in_leaf': int(search_space['min_data_in_leaf']),
            'early_stopping_round': int(
                search_space['n_estimators'] *
                search_space['early_stopping_round']),
            'learning_rate': search_space['learning_rate'],
            'num_leaves': int(search_space['num_leaves']),
            'lambda_l1': search_space['lambda_l1'],
            'lambda_l2': search_space['lambda_l2'],
            'linear_lambda': search_space['linear_lambda']
            }
        params.update(models_objectives[config])

        # Have to re-create dataset inside the train loop
        # lgbm_max bin works on the dataset keeping it from
        # being reused for each iteration
        d_train = lgb.Dataset(data=pca.transform(X_train),
                              label=y_train,
                              params={'verbose': -1})
        d_test = lgb.Dataset(data=pca.transform(X_test),
                             label=y_test,
                             params={'verbose': -1})

        if params['objective'] != 'regression':

            start = timer()
            model = lgb.train(params=params,
                              train_set=d_train,
                              valid_sets=[d_train, d_test],
                              num_boost_round=int(search_space['n_estimators'])
                              )
            end = timer()
            loss = model.best_score['valid_1']['quantile']

        else:
            start = timer()
            model = lgb.train(params=params,
                              train_set=d_train,
                              valid_sets=[d_train, d_test],
                              feval=phmap_agg_loss,
                              num_boost_round=int(search_space['n_estimators'])
                              )
            end = timer()
            loss = model.best_score['valid_1']['mean_phmap']

        # Training time in seconds
        train_time = end - start

        return {'loss': loss, 'status': STATUS_OK, 'train_time': train_time}

    trials = Trials()
    best_hyperparams = fmin(fn=objective,
                            space=SEARCH_SPACE,
                            algo=tpe.suggest,
                            max_evals=max_rounds,
                            trials=trials,
                            verbose=True)

    time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")

    with open(MODEL_PATH / f'{time_tag}_{config}_parms.pkl', 'wb') as f:
        best_parameters = space_eval(SEARCH_SPACE, best_hyperparams)
        best_parameters.update(models_objectives[config])
        best_parameters['early_stopping_round'] = int(
            best_parameters['early_stopping_round'] *
            best_parameters['n_estimators']
        )
        print(best_parameters)
        pickle.dump(best_parameters, f)

    with open(MODEL_PATH / f'{time_tag}_{config}trials_result.pkl', 'wb') as f:
        pickle.dump(trials, f)
