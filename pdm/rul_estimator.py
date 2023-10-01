import pathlib
import pickle
from datetime import datetime
import numpy as np
import lightgbm as lgb
from scipy.stats import kurtosis, skew, variation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

from dataset_loader import DataLoader


def phmap_loss(y_true, y_pred):

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    diff = y_true - y_pred
    alpha = np.where(diff <= 0, -1/10, 1/13)
    nasa = np.mean(np.exp(alpha * diff))

    return 0.5*(rmse + nasa)


def grad_phmap_loss(y_pred, train_data):
    y_true = train_data.get_label()
    residual = y_true - y_pred
    alpha = np.where(residual < 0, -1/10, 1/13)

    # grad = 0.5*(-1*np.sign(residual) - alpha * np.exp(alpha*residual))
    # hess = 0.5 * alpha ** 2 * np.exp(alpha*residual)

    grad = - (residual + 0.5 * alpha * np.exp(alpha*residual))
    hess = np.ones_like(alpha) + 0.5 * alpha ** 2 * np.exp(alpha*residual)

    return grad, hess


def phmap_agg_loss(y_pred, train_data):
    labels = train_data.get_label()

    return 'mean_phmap', phmap_loss(labels, y_pred), False


DATA_DIR = pathlib.Path.cwd().parent / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_PATH = pathlib.Path.cwd() / 'model' / 'lgbm'
MODEL_PATH.mkdir(exist_ok=True)

funcs = [
        (np.min, 'min', {'axis': 0}),
        (np.quantile, 'perc_25',
         {'axis': 0, 'q': 0.25, 'method': 'median_unbiased'}),
        (np.median, 'median', {'axis': 0}),
        (np.quantile, 'perc_75',
         {'axis': 0, 'q': 0.75, 'method': 'median_unbiased'}),
        (np.max, 'max', {'axis': 0}),
        (np.mean, 'mean', {'axis': 0}),
        (np.var, 'variance', {'axis': 0}),
        (np.std, 'std_dev', {'axis': 0}),
        (variation, 'var_coeff', {'axis': 0}),
        (skew, 'skew', {'axis': 0}),
        (kurtosis, 'kurtosis', {'axis': 0})
    ]

temperatures = ['T24', 'T30', 'T48', 'T50']
pressures = ['P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50']
motor_vars = ['Nf', 'Nc', 'Wf']
environment = ['alt', 'Mach', 'TRA', 'T2']
predictors = temperatures + pressures + motor_vars + environment
extra = ['cycle', 'Fc', 'RUL', 'hs', 'unit']
min_size = 0.12
jump = 0.05
bks = 3

dataset = DataLoader(
    processed_data_dir=PROCESSED_DATA_DIR,
    predictor_names=predictors,
    extra_names=extra,
    stat_funcs=funcs,
    min_size=min_size,
    jump=jump,
    n_splits=bks
)

X, y = dataset.get_data()
SEARCH_SPACE = {
    # Do not change
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': grad_phmap_loss,
    'deterministic': True,
    'num_threads': 12,
    'seed': 7501,
    'verbose': -1,
    'force_col_wise': True,
    # Learning parameters
    'max_bin': hp.uniformint('max_bin', 10, 400),
    'feature_fraction_bynode': hp.uniform('feature_fraction_bynode', 0.5, 1),
    'max_depth': hp.uniformint('max_depth', 5, 200),
    'min_sum_hessian_in_leaf': hp.uniform(
        'min_sum_hessian_in_leaf', 1e-5, 1e-1),
    'extra_trees': hp.choice('extra_trees', [True, False]),
    'feature_fraction': hp.uniform('feature_fraction', 0, 1),
    'bagging_freq': hp.uniformint('bagging_freq', 1, 100),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
    'min_data_in_leaf': hp.uniformint('min_data_in_leaf', 2, 40),
    'early_stopping_round': hp.uniform('early_stopping_round', 0.05, 0.3),
    'n_estimators': hp.uniformint('n_estimators', 100, 10_000),
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.12),
    'num_leaves': hp.uniformint('num_leaves', 10, 64),
    'lambda_l1': hp.uniform('lambda_l1', 0, 100),
    'lambda_l2': hp.uniform('lambda_l2', 0, 100),
    'linear_lambda': hp.uniform('linear_lambda', 0, 100)
}


def objective(search_space):
    params = {
        # Do not change
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'deterministic': True,
        'num_threads': 12,
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
            search_space['n_estimators'] * search_space['early_stopping_round']),
        'learning_rate': search_space['learning_rate'],
        'num_leaves': int(search_space['num_leaves']),
        'lambda_l1': search_space['lambda_l1'],
        'lambda_l2': search_space['lambda_l2'],
        'linear_lambda': search_space['linear_lambda']
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(['unit_names', 'hs'], axis=1),
        y,
        test_size=0.25,
        random_state=7501)

    pca = PCA()
    d_train = lgb.Dataset(data=pca.fit_transform(X_train),
                          label=y_train,
                          params={'verbose': -1})
    d_test = lgb.Dataset(data=pca.transform(X_test),
                         label=y_test,
                         params={'verbose': -1})

    model = lgb.train(params=params,
                      train_set=d_train,
                      feval=phmap_agg_loss,
                      valid_sets=[d_train, d_test],
                      num_boost_round=int(search_space['n_estimators'])
                      )

    loss = model.best_score['valid_1']['mean_phmap']

    return {'loss': loss, 'status': STATUS_OK}


trials = Trials()
best_hyperparams = fmin(fn=objective,
                        space=SEARCH_SPACE,
                        algo=tpe.suggest,
                        max_evals=3_000,
                        trials=trials,
                        verbose=True)

time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")

with open(MODEL_PATH / f'{time_tag}_model_parms.pkl', 'wb') as f:
    pickle.dump(space_eval(SEARCH_SPACE, best_hyperparams), f)

with open(MODEL_PATH / f'{time_tag}_trials_result.pkl', 'wb') as f:
    pickle.dump(trials, f)
