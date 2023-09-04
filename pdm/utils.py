import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# Typing
from typing import Union, Tuple, TypedDict
from sklearn.base import BaseEstimator


# Variable Names from the NCMPASS dataset
VAR_NAMES = {
    'alt': 'Altitude (ft)',
    'Mach': 'Mach',
    'TRA': 'Throttle-resolver angle (°)',
    'T2': 'Total temperature at fan inlet (°R)',
    'T24': 'Total temperature at LPC outlet (°R)',
    'T30': 'Total temperature at HPC outlet (°R)',
    'T48': 'Total temperature at HPT outlet (°R)',
    'T50': 'Total temperature at LPT outlet (°R)',
    'P15': 'Total pressure in bypass-duct (psia)',
    'P2': 'Total pressure at fan inlet (psia)',
    'P21': 'Total pressure at fan outlet (psia)',
    'P24': 'Total pressure at LPC outlet (psia)',
    'Ps30': 'Static pressure at HPC outlet (psia)',
    'P40': 'Total pressure at burner outlet (psia)',
    'P50': 'Total pressure at LPT outlet (psia)',
    'Nf': 'Physical fan speed (rpm)',
    'Nc': 'Physical core speed (rpm)',
    'Wf': 'Fuel ﬂow (pps)'
}

class SplitProccessedData(TypedDict):
    # Base class descriptor for the split and processed
    # time series dataframes
    X_train_processed: np.ndarray
    X_test_processed: np.ndarray
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame


def get_healthy_faulty(file_path: Union[str, pathlib.PosixPath],
                       descriptors: list[str],
                       target_cols: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns dataframes of time series with healthy and faulty operations

    Parameters
    ----------
    file_path : Union[str, pathlib.PosixPath]
        Path to processed parquet file to separate
    descriptors : list[str]
        List of environmental descriptors used in normalizer
    target_cols : list[str]
        Sensor measures to be predicted with descriptor columns

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Healthy and Faulty pandas dataframes containing descriptors and target columns

    Raises
    ------
    ValueError
        In case the provided path leads to a non existent file
    ValueError
        In case the set of target columns and environmental descriptors
        contains a column not present in the processed dataframe
    """

    df = pd.read_parquet(file_path)

    if not pathlib.Path(file_path).exists():
        raise ValueError('Provide a valid database file')

    cols_to_use = descriptors + target_cols

    not_found_cols = list(set(cols_to_use) - set(df.columns))
    if not_found_cols:
        raise ValueError(f'{not_found_cols} not present in df columns')


    healthy = df[df['hs'] == 1].copy()
    faulty = df[df['hs'] == 0].copy()

    # Creates a linearly increasing time feature ranging from 0 (start of cycle)
    # to 1 (end of cycle)
    healthy['time'] = healthy.groupby(['unit', 'cycle'])['cycle'].transform(
        lambda x: np.linspace(0, 1, len(x))

    )
    faulty['time'] = faulty.groupby(['unit', 'cycle'])['cycle'].transform(
        lambda x: np.linspace(0, 1, len(x))
    )

    cols_to_use.append('time')
    df_healthy = healthy[cols_to_use]
    df_faulty = faulty[cols_to_use]

    return df_healthy, df_faulty



def split_scale_data(df: pd.DataFrame,
                     targets: list[str],
                     test_size: float,
                     random_state: int,
                     scaler: BaseEstimator) -> Tuple[SplitProccessedData, BaseEstimator]:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Base dataframe to be splitted
    targets : list[str]
        List of target columns to be predicted
    test_size : float
        Fraction of df to be used for testing (between 0 and 1)
    random_state : int
        Random state used for reproducibility
    scaler : BaseEstimator
        Sklearn scaler, like MinMaxScaler, StandardScaler

    Returns
    -------
    Tuple[SplitProccessedData, BaseEstimator]
        Dictionary with train and test splits before preprocessing and with processing
        Fitted scaler on training data
    """

    # Creates a helper dataframe of units and their corresponding samples
    # These are the single xs of the general X
    # Each time series is considered as a sample (not every entry from the original
    # dataframme)
    unit_cycles = df.groupby(['unit', 'cycle']).size().reset_index()
    unit_cycles.drop(0, axis=1, inplace=True)
    train = unit_cycles.sample(frac=(1 - test_size), random_state=random_state)
    test = pd.concat([unit_cycles, train, train]).drop_duplicates(keep=False)

    X_train_raw = df.merge(train, on=['unit', 'cycle'], how='inner')
    X_test_raw = df.merge(test, on=['unit', 'cycle'], how='inner')

    y_train = X_train_raw[targets]
    y_test = X_test_raw[targets]

    X_train_raw = X_train_raw.drop(targets, axis=1)
    X_test_raw = X_test_raw.drop(targets, axis=1)

    scaler.fit(X_train_raw)

    X_train_processed = scaler.transform(X_train_raw)
    X_test_processed = scaler.transform(X_test_raw)

    unit_col = X_train_raw.columns.get_loc('unit')

    # Drop unit column
    X_train_processed = np.delete(X_train_processed, unit_col, axis=1)
    X_test_processed = np.delete(X_test_processed, unit_col, axis=1)

    results = {'X_train_processed': X_train_processed,
               'X_test_processed': X_test_processed,
               'y_train': y_train,
               'y_test': y_test,
               'X_train_raw': X_train_raw,
               'X_test_raw': X_test_raw}

    return results, scaler


def format_predictions(predictions: np.ndarray,
                       X_raw: pd.DataFrame,
                       y: pd.DataFrame,
                       target_cols: list[str],
                       descriptors_cols: list[str]) -> pd.DataFrame:
    """Returns formatted predictions with real values and
    predictions for comparison

    Parameters
    ----------
    predictions : np.ndarray
        Array of predictions generated by the model
    X_raw : pd.DataFrame
        Raw inputs (before preprocessing) of the model
    y : pd.DataFrame
        Real values of the predictions
    target_cols : list[str]
        Features predicted by the model
    descriptors_cols : list[str]
        Descriptor/Environmental measures used to predict target_cols

    Returns
    -------
    pd.DataFrame
        Dataframe with real and predicted values concatenated
    """

    preds = predictions.copy()
    preds = pd.DataFrame(preds, columns=target_cols)
    real = pd.DataFrame(y.values, columns=target_cols)
    # Concatenate real and predicted values, ignoring index
    preds = pd.concat([real, preds], axis=1, ignore_index=True)
    preds.columns = target_cols + [f'{t}_pred' for t in target_cols]

    # Must use X_raw since it includes the unit
    # descriptors_cols.remove('time')
    for col in descriptors_cols:
        preds[col] = X_raw[col].values

    return preds


def plot_results_single(preds: pd.DataFrame,
                        sample_col: str,
                        unit: int,
                        cycle: int,
                        model_name: str) -> None:
    """Saves a plot of the predictions of a model for a
    certain column, a given cycle for a given unit

    Parameters
    ----------
    preds : pd.DataFrame
        Formatted predictions
    sample_col : str
        Column to be plotted
    unit : int
        Unit (id) of the machine to filter
    cycle : int
        Operative cycle (id) of the machine to filter
    model_name : str
        Name of the model to be used as a directory for the resulting image
    """

    sample = preds[(preds['unit'] == unit) & (preds['cycle'] == cycle)]
    sample = sample[[sample_col, f'{sample_col}_pred', 'time',
                    'cycle', 'alt', 'Mach', 'TRA', 'T2']]
    sample.set_index('time', inplace=True)
    sample.sort_index(inplace=True)
    t = sample.index

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(8, 8, figure=fig)

    ax1 = fig.add_subplot(gs[0:8, 0:4])
    ax2 = fig.add_subplot(gs[0:4, 4:6])
    ax3 = fig.add_subplot(gs[0:4, 6:8])
    ax4 = fig.add_subplot(gs[4:8, 4:6])
    ax5 = fig.add_subplot(gs[4:8, 6:8])

    ax1.plot(t, sample[f'{sample_col}'], label='Real')
    ax1.plot(t, sample[f'{sample_col}_pred'], label='Predicted')
    ax1.set_xlabel('Time (% of cycle)')
    ax1.set_ylabel(VAR_NAMES[sample_col])
    ax1.legend()

    ax2.plot(t, sample['alt'])
    ax2.set_ylabel('Altitude (ft)')

    ax3.plot(t, sample['Mach'])
    ax3.set_ylabel('Mach Number')

    ax4.plot(t, sample['TRA'])
    ax4.set_ylabel('TRA (%)')
    ax4.set_xlabel('Time (% of cycle)')

    ax5.plot(t, sample['T2'])
    ax5.set_ylabel('Temperature (°R)')
    ax5.set_xlabel('Time (% of cycle)')
    fig.text(0.5, 1.07, f'{VAR_NAMES[sample_col]}', fontsize=16, ha='center')
    fig.text(0.5, 1.02,
                f'Operaring Conditions for Unit {unit} - Cycle {cycle}',
                fontsize=12,
                ha='center')

    plt.tight_layout()
    # Apply grid to every subplot
    for ax in fig.axes:
        ax.grid()

    img_path = pathlib.Path('img') / model_name
    img_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(img_path / f'{sample_col}_cycle_{cycle}_pred_op_conditions.png',
                dpi=300,
                bbox_inches='tight')
    plt.close()