"""Calculate Shapley values for the training data of an algorithm.

The script uses the Permutation Explainer by default
(as does the SHAP library) since it doesn't know if the
provided algorithm already has a fast implementation at hand.

* Do not remove the loss import (pickle serialize errors)
"""
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
from loss import PHMAP


def main() -> None:
    """Calculate Shapley values via permutation algorithm."""
    with open('model/trained/xgb_pipe_estimator.pkl', 'rb') as f:
        estimator = pickle.load(f)

    X = pd.read_csv('data/phmap_dataset.csv').drop(
                labels=['unit_names', 'hs'],
                axis=1)
    y = pd.read_csv('data/ruls.csv').values.reshape(1, -1)[0]
    X = estimator.steps[0][1].transform(X)
    # X = estimator.steps[1][1].transform(X)
    feature_names = estimator.steps[0][1].get_feature_names_out()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    def composed_predict(x):
        x = estimator.steps[1][1].transform(x)
        return estimator.steps[2][1].predict(x)

    explainer = shap.PermutationExplainer(model=composed_predict,
                                          masker=X_train,
                                          feature_names=feature_names,
                                          max_evals=2*X_train.shape[1]+1,
                                          seed=7501)

    shap_values = explainer(X_train)

    with open('model/trained/final/xgb_shap_values.pkl', 'wb') as f:
        pickle.dump(shap_values, f)


if __name__ == '__main__':
    main()
