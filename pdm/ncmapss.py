import numpy as np
import pandas as pd
import h5py
import pathlib


class NcmapssLoader:

    def __init__(self, data_dir: pathlib.Path, file: str, decimation: int = 10):
        self.data_dir = data_dir
        self.available_measurements = ['W', 'X_s', 'X_v', 'T', 'Y', 'A']
        self.decimation = decimation
        self.file_path = data_dir / file
        self.df = None


    def decimate_dataframe(self, df):

        filtered_rows = []
        grouped = df.groupby(['unit', 'cycle'])
        for name, group in grouped:
            unit, cycle = name
            rows = group.iloc[::self.decimation, :]
            filtered_rows.append(rows)

        return pd.concat(filtered_rows)


    def load_file(self, measure_interest):

        if measure_interest not in self.available_measurements:
            raise ValueError(f"measure_interest must be one of {self.available_measurements}")

        train = f'{measure_interest}_dev'
        test = f'{measure_interest}_test'

        with h5py.File(self.file_path, 'r') as hdf:

            if measure_interest == 'Y':
                labels = ['RUL']
                data_type = np.uint8
            else:
                labels = np.array(hdf.get(f'{measure_interest}_var'))
                labels = list(np.array(labels, dtype=str))

                if measure_interest == 'A':
                    data_type = np.uint8
                else:
                    data_type = np.float32

            train_measure = np.array(hdf.get(train), dtype=data_type)
            test_measure = np.array(hdf.get(test), dtype=data_type)

        data = np.concatenate((train_measure, test_measure), axis=0)
        data = pd.DataFrame(data, columns=labels)

        return data


    def create_dataset(self):

        W = self.load_file('W')
        X_s = self.load_file('X_s')
        A = self.load_file('A')
        Y = self.load_file('Y')
        df = pd.concat([W, X_s, A, Y], axis=1)
        self.df = self.decimate_dataframe(df)


    def save_dataset(self):

        processed_dir = self.data_dir / 'processed'
        processed_dir.mkdir(exist_ok=True)

        file_name = f'{self.file_path.stem}_decimated_{self.decimation}.parquet'
        file_path = processed_dir / file_name
        self.df.to_parquet(file_path)