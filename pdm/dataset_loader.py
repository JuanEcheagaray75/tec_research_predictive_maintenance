import pathlib
from typing import Tuple, List
import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.preprocessing import MinMaxScaler
from time_segmentator import TimeSegmentProcessor


class DataLoader():

    def __init__(self,
                 processed_data_dir: pathlib.Path,
                 predictor_names: list[str],
                 extra_names: list[str],
                 stat_funcs: List[Tuple[callable, str, dict]],
                 min_size: float = 0.1,
                 jump: float = 0.05,
                 n_splits: int = 2):
        """PHMAP2021 Data Loader Init

        Parameters
        ----------
        processed_data_dir : pathlib.Path
            Path where reduced datasets are stored
        predictor_names : list[str]
            Descriptor variables used to predict RUL
        extra_names : list[str]
            Helper variables or flight-constant features
        stat_funcs : List[Tuple[callable, str, dict]]
            List of descriptor functions to be applied on each segment
        min_size : float, optional
            Proportion of flight length to be considered
            as the minimal size for a given segment, by default 0.1
        jump : float, optional
            Will search for changepoints at
            int(len(array) * jump), by default 0.05
        n_splits : int, optional
            Number of breakpoints to find in array, by default 2
        """

        self.processed_data_dir = processed_data_dir
        self.predictor_names = predictor_names
        self.extra_names = extra_names
        self.stat_funcs = stat_funcs
        self.min_size = min_size
        self.jump = jump
        self.n_splits = n_splits

    def get_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns statistical descriptor matrix and RULS

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Matrix of statistical descriptors and array of RULS

        Raises
        ------
        ValueError
            In case the specified segmentation algorithm doesn't
            find the specified number of break points
        """

        results = []
        ruls = []
        unit_names = []
        hs_l = []

        for file in self.processed_data_dir.iterdir():
            df = pd.read_parquet(self.processed_data_dir / file)

            # Get all available flights for each plane
            flights = df.groupby(['unit', 'cycle']).size().reset_index()
            flights.drop(0, axis=1, inplace=True)
            file_name = file.stem.split('_decimated')[0].split('N-CMAPSS_')[1]

            for i, (_, row) in enumerate(flights.iterrows()):

                # Filter flight
                plane, n_flight = row
                flight_sample = df[
                    (df['unit'] == plane) & (df['cycle'] == n_flight)
                    ]

                # Helper values that remain constant for the flight
                fc = flight_sample['Fc'].iloc[0]
                rul = flight_sample['RUL'].iloc[0]
                hs = flight_sample['hs'].iloc[0]

                # Initial preprocessing on data
                # Distributes weights fairly
                scaler = MinMaxScaler()
                data = flight_sample.drop(self.extra_names, axis=1)
                data = data[self.predictor_names]
                flight_processed = scaler.fit_transform(data)

                # Generate multivariate array splits
                segmentator = TimeSegmentProcessor(
                    search_method=rpt.Binseg,
                    model='l2',
                    min_size=int(self.min_size*flight_processed.shape[0]),
                    jump=int(self.jump*flight_processed.shape[0]),
                    n_bks=self.n_splits,
                    stat_funcs=self.stat_funcs
                )

                segment_statistics = segmentator.process_segments(
                    flight_processed
                    )
                segment_statistics = np.append(segment_statistics, n_flight)
                segment_statistics = np.append(segment_statistics, fc)

                results.append(segment_statistics)
                ruls.append(rul)
                unit_names.append(f'{file_name}_{plane}')
                hs_l.append(hs)

        ref_shape = results[0].shape
        if not np.all(np.array([arr.shape for arr in results]) == ref_shape):
            # This error occurs if the provided segmentation algorithm didn't
            # find the specified number of break points
            raise ValueError("Arrays do not have the same shape")

        res_array = np.vstack(results)
        ruls = np.array(ruls)

        # Column names for pandas dataframe
        fnames = [f[1] for f in self.stat_funcs]

        col_names = []

        for i in range(self.n_splits + 1):
            for var in self.predictor_names:
                for func in fnames:
                    col_names.append(f'{var}_{func}_{i}')
        col_names.extend(['n_flight', 'fc'])

        res_df = pd.DataFrame(res_array, columns=col_names)
        unit_names = pd.Series(unit_names, name='unit_names')
        hs_l = pd.Series(hs_l, name='hs')
        res_df = pd.concat([res_df, unit_names, hs_l], axis=1)
        ruls = pd.Series(ruls, name='RUL')

        res_df['n_flight'] = res_df['n_flight']

        return res_df, ruls
