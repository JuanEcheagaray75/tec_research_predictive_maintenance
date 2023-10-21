"""Calculate statistical descriptors on PHMAP processed dataset.

Reads every processed parquet file  obtained from reduce_data and creates
2 csv files containing just statistical descriptors (and helper features) and
RUL labels
"""
import pathlib
import numpy as np
from scipy.stats import kurtosis, skew, variation
from dataset_loader import DataLoader


def main() -> None:
    """Calculate statistical descriptors on PHMAP processed dataset."""
    DATA_DIR = pathlib.Path.cwd().parent / 'data'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
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

    print('Loading dataset')
    X, y = dataset.get_data()

    X.to_csv('data/phmap_dataset.csv', index=False)
    y.to_csv('data/ruls.csv', index=False)


if __name__ == '__main__':
    main()
