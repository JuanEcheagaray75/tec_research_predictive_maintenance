import click
from normalizer import DescriptorPredictor
import pathlib
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam, Lion, Adamax
from utils import get_healthy_faulty, split_scale_data
from typing import Tuple, Union
from sklearn.base import BaseEstimator

# General parameters
DATA_DIR = pathlib.Path.cwd().parent / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_PATH = pathlib.Path('model/normalizer/')
LOG_PATH = pathlib.Path('model/logs_normalizer/')
OPTIMIZERS = {'adam': Adam(),
              'lion': Lion(),
              'adamax': Adamax()}


def fetch_process_data(file: pathlib.PosixPath,
                       descriptors: list[str],
                       targets: list[str],
                       scaler: BaseEstimator,
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray,
                                                        pd.DataFrame, pd.DataFrame, BaseEstimator]:
    """Returns the train test split of the data with the fitted scaler

    Parameters
    ----------
    file : pathlib.PosixPath
        File path pointing to the parquet file used for training
    descriptors : list[str]
        List of predictor variables to be used to predict targets
    targets : list[str]
        List of sensor variables to be predicted with descriptors
    scaler : BaseEstimator
        Scikit-Learn scaler
    random_state : int, optional
        Random state used to split the data, by default 42

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, BaseEstimator]
        Processed train and test dataframes, pandas dataframes with targets and a fitted scaler
    """
    df_healthy, _ = get_healthy_faulty(file_path=file,
                                            descriptors=descriptors,
                                            target_cols=targets)

    split_scaled_data, fit_scaler = split_scale_data(df=df_healthy,
                                        targets=targets,
                                        test_size=0.2,
                                        random_state=random_state,
                                        scaler=scaler)

    X_train_processed = split_scaled_data['X_train_processed']
    X_test_processed = split_scaled_data['X_test_processed']
    y_train = split_scaled_data['y_train']
    y_test = split_scaled_data['y_test']


    return X_train_processed, X_test_processed, y_train, y_test, fit_scaler


def set_model_paths(file: str,
                    time_tag: str,
                    dense: int,
                    neurons: Union[int, list[int]],
                    batch_size: int,
                    max_epochs: int,
                    patience: int,
                    optimizer: str) -> Tuple[str, pathlib.PosixPath, pathlib.PosixPath]:
    """Helper function to get default callbacks paths

    Parameters
    ----------
    file : str
        Parquet file used in training
    time_tag : str
        Formatted time stamp
    dense : int
        Number of dense layers in the model
    neurons : Union[int, list[int]]
        Neurons for each layer
    batch_size : int
        Batch size used in training
    max_epochs : int
        Maximum number of epochs to train a model
    patience : int
        Number of epochs to wait for improvement in validation loss
    optimizer : str
        Optimizer string representation

    Returns
    -------
    Tuple[str, pathlib.PosixPath, pathlib.PosixPath]
        Name of the model, path used to store checkpoints, tensorboard logs path

    Raises
    ------
    ValueError
        _description_
    """

    if isinstance(neurons, int):
        neurons_ls = [neurons] * dense
    elif isinstance(neurons, list):
        neurons_ls = neurons
    else:
        raise ValueError('Neurons provided is not a list or int')

    neurons_repr = '_'.join(map(str, neurons_ls))

    model_name = f'{time_tag}_{file}_batch_size_{batch_size}_max_epochs_{max_epochs}_patience_{patience}_dense_{dense}_neurons_{neurons_repr}_optimizer_{optimizer}'
    tb_log = LOG_PATH / model_name
    model_experiment_path = MODEL_PATH / model_name
    tb_log.mkdir(parents=True, exist_ok=True)
    model_experiment_path.mkdir(parents=True, exist_ok=True)

    return model_name, model_experiment_path, tb_log


def set_callbacks(patience: int,
                  model_experiment_path: pathlib.PosixPath,
                  tensorboard_log_path: pathlib.PosixPath) -> list[callable]:
    """Helper function to set basic tensorflow callbacks

    Parameters
    ----------
    patience : int
        Number of epochs to wait for an improvement in validation loss
    model_experiment_path : pathlib.PosixPath
        Path where checkpoints are to be stored
    tensorboard_log_path : pathlib.PosixPath
        Path to store tensorboard logs

    Returns
    -------
    list[callable]
        List of all callbacks
    """    

    early_stopper = EarlyStopping(monitor='val_loss',
                                  patience=patience)
    checkpoint = ModelCheckpoint(
        filepath=str(model_experiment_path),
        monitor='val_loss',
        mode='min',
        save_weights_only=False,
        save_best_only=True,
        verbose=1
    )
    tensor_board = TensorBoard(
        log_dir=tensorboard_log_path
    )
    print(f'''
    Callbacks Initialized:
    ------------------
    Model checkpoint at: {model_experiment_path}
    Tensorboard: tensorboard --logdir={tensorboard_log_path.resolve()}
    ''')

    return [early_stopper, checkpoint, tensor_board]


@click.command()
@click.option('--file', default='N-CMAPSS_DS01-005_decimated_5.parquet', help='File to be used in training')
def main(file: str):

    file_path = pathlib.Path(PROCESSED_DATA_DIR / file)

    if not file_path.exists():
        raise ValueError(f'Provided file {file} cannot be found in processed directory')

    print(f'Starting training on {file}')

    # Data loading and preprocessing
    descriptors = ['alt', 'Mach', 'TRA', 'T2', 'unit', 'cycle', 'Fc']
    targets = ['T24', 'T30', 'T48', 'T50', 'P15',
            'P2', 'P21', 'P24', 'Ps30', 'P40',
            'P50', 'Nf', 'Nc', 'Wf']
    scaler = MinMaxScaler()

    X_train, X_test, y_train, y_test, fit_scaler = fetch_process_data(
        file=file_path,
        descriptors=descriptors,
        targets=targets,
        scaler=scaler
    )

    # Model parameters
    n_dense = 5
    n_neurons = 64
    output_size = 14
    optimizer = 'adamax'
    # Model Hyperparameters
    batch_size = 64
    max_epochs = 100
    patience = int(0.1 * max_epochs)
    l2_reg = 0.005
    time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Callbacks Setup
    model_name, model_experiment, tb_path = set_model_paths(file=file_path.stem,
                                                            time_tag=time_tag,
                                                            dense=n_dense,
                                                            neurons=n_neurons,
                                                            batch_size=batch_size,
                                                            max_epochs=max_epochs,
                                                            patience=patience,
                                                            optimizer=optimizer)
    model_callbacks = set_callbacks(patience=patience,
                                    model_experiment_path=model_experiment,
                                    tensorboard_log_path=tb_path)

    # Model definition & training
    model = DescriptorPredictor(output_size=output_size,
                                n_dense=n_dense,
                                neurons=n_neurons,
                                optimizer=OPTIMIZERS[optimizer],
                                l2_regularization=l2_reg)
    model.compile(model.optimizer,
                  model.loss_fn)
    model.fit(x=X_train,
              y=y_train,
              batch_size=batch_size,
              epochs=max_epochs,
              validation_data=(X_test, y_test),
              callbacks=model_callbacks,
              verbose=2)

    # Save scaler
    joblib.dump(fit_scaler, MODEL_PATH / f'{model_name}_scaler.joblib')


if __name__ == '__main__':
    main()