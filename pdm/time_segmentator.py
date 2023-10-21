"""MTS segmentator wrapper.

Applies an offline time series segmentation technique
on multivariate data, extracts a set of statistical properties
from the array.
"""
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from typing import List


class TimeSegmentProcessor():
    """MTS processor to get stats of relevant segments in series."""

    def __init__(self,
                 search_method: rpt.base.BaseEstimator,
                 model: str,
                 min_size: int,
                 jump: int,
                 n_bks: int,
                 stat_funcs: list):
        """Multivariate time series changepoint detection wrapper.

        Parameters
        ----------
        search_method : rpt.base.BaseEstimator
            Ruptures search algorithm
        model : str
            Ruptures cost function acronym (aka. l1, l2)
        min_size : int
            Min size between changepoints (aka. split array size)
        jump : int
            Only looks for changes at every kth idx
        n_bks : int
            Number of splits to estimate in array
        stat_funcs : list[callable]
            List of statistical functions to apply to extract features
            Must accept an axis parameter for multivariate time series
        """
        self.search_method = search_method
        self.model = model
        self.min_size = min_size
        self.jump = jump
        self.n_bks = n_bks
        self.stat_funcs = stat_funcs

    def calculate_change_points(self, array: np.ndarray):
        """Detect changepoints in a given ndarray.

        Parameters
        ----------
        array : np.ndarray
            Array in which to find splits

        Raises
        ------
        ValueError
            In case the provided model parameters make impossible to
            split the array for the given number of splits
        """
        self.algo = self.search_method(
                    model=self.model,
                    jump=self.jump,
                    min_size=self.min_size
                )
        self.algo.fit(array)
        self.bks_idx = self.algo.predict(self.n_bks)

        if len(self.bks_idx) != self.n_bks + 1:
            raise ValueError(
                f"Algo couldn't find {self.n_bks} changepoints in the array"
                )

    def split_array(self, array: np.ndarray) -> List[np.ndarray]:
        """Get a list of the splitted arrays.

        Parameters
        ----------
        array : np.ndarray
            Base array to be splitted

        Returns
        -------
        list[np.ndarray]
            List of numpy arrays (of varying length)
        """
        if len(array.shape) == 1:

            sub_arrays = []
            for i in range(len(self.bks_idx)):
                start = 0 if i == 0 else self.bks_idx[i - 1]
                end = self.bks_idx[i]
                sub = array[start:end]
                sub_arrays.append(sub)

        if len(array.shape) == 2:

            sub_arrays = []
            for i in range(len(self.bks_idx)):
                start = 0 if i == 0 else self.bks_idx[i - 1]
                end = self.bks_idx[i]
                sub = array[start:end, :]
                sub_arrays.append(sub)

        return sub_arrays

    def segment_stats(self, segment: np.ndarray) -> np.ndarray:
        """Calculate statistical properties of a segment.

        Parameters
        ----------
        segment : np.ndarray
            Segment to which stats will be calculated

        Returns
        -------
        np.ndarray
            Flattened array of stats properties of the segment
        """
        stats = [f[0](segment, **f[2]) for f in self.stat_funcs]
        # Calculates resulting statistics "f" for each variable
        # Each row represents different variables
        return np.vstack(stats).T.flatten()

    def process_segments(self, array: np.ndarray) -> np.ndarray:
        """Calculate change-points and provide the final stats of the arrays.

        Parameters
        ----------
        array : np.ndarray
            Base array to be analyzed

        Returns
        -------
        np.ndarray
            Array of stats of each segment found in the base array
        """
        self.calculate_change_points(array)
        sub_arrs = self.split_array(array)
        res = []
        for arr in sub_arrs:

            res.append(self.segment_stats(arr))

        return np.concatenate(res, axis=0)

    def plot_segments(self, array: np.ndarray):
        """Plot the segment results.

        Parameters
        ----------
        array : np.ndarray
            Array to be plotted
        """
        rpt.show.display(array, self.bks_idx)
        plt.show()
