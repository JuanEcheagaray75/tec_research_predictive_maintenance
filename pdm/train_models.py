"""Train XGBoost models with the best hyperparameters.

Read the pickled parameters found through TPE and train a
XGBoost model with them; it makes use of sklearn pipelines
to encompass both variance feature selector and PCA into
a single function.
"""
import pathlib
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss
from sklearn.pipeline import Pipeline
from loss import PHMAP, phmap_loss


def get_model_params(path: pathlib.PosixPath) -> dict:
    """Extract dictionary of best hyperparameters.

    Parameters
    ----------
    path : pathlib.PosixPath
        Path to the dict. of results from TPE for the given model

    Returns
    -------
    dict
        Dictionary of parameters and values
    """
    with open(path, 'rb') as f:
        params = pickle.load(f)
    return params


def main() -> None:
    """Train each model with the best parameters found through TPE."""
    BASE_MODELS = pathlib.Path('model/xgb/20231016-180521')
    model_paths = [i for i in BASE_MODELS.glob('*parms.pkl')]
    names = [i.stem.split('_')[1] for i in model_paths]
    names_paths = {}
    for name, path in zip(names, model_paths):
        names_paths[name] = {'path': path,
                             'params': get_model_params(path)}

    X = pd.read_csv('data/phmap_dataset.csv').drop(
                labels=['unit_names', 'hs'],
                axis=1)
    y = pd.read_csv('data/ruls.csv').values.reshape(1, -1)[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    for model_name in names_paths:
        estimator_parms = names_paths[model_name]['params']
        print(f'Fitting {model_name}')
        print(f'Using params {estimator_parms}')

        if model_name == 'estimator':
            estimator_parms['objective'] = 'reg:squarederror'
            eval_metric = PHMAP
        else:
            estimator_parms['objective'] = 'reg:quantileerror'
            eval_metric = None

            if model_name == 'lower':
                estimator_parms['quantile_alpha'] = 0.05
            elif model_name == 'upper':
                estimator_parms['quantile_alpha'] = 0.95
            else:
                estimator_parms['quantile_alpha'] = 0.5

        estimator_parms['device'] = 'gpu'

        var_trans = VarianceThreshold(
            threshold=estimator_parms.pop('variance_threshold'))
        pca = PCA()
        n_estimators = estimator_parms.pop('num_boost_rounds')
        model = xgb.XGBRegressor(
            **estimator_parms,
            eval_metric=eval_metric,
            n_estimators=n_estimators
            )

        pipe = Pipeline(steps=[('variance_threshold', var_trans),
                               ('pca', pca),
                               (model_name, model)])

        pipe.fit(X_train, y_train)
        if 'estimator' in model_name:
            print(phmap_loss(y_test, pipe.predict(X_test)))
        else:
            print(mean_pinball_loss(y_true=y_test,
                                    y_pred=pipe.predict(X_test),
                                    alpha=estimator_parms['quantile_alpha']))

        # Saving Pipe
        save_path = f'model/trained/xgb_pipe_{model_name}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(pipe, f)


if __name__ == '__main__':
    main()
