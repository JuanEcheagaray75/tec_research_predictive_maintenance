import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt


class TimeSegmentProcessor():

    def __init__(self,
                 search_method: rpt.base.BaseEstimator,
                 model: str,
                 min_size: int,
                 jump: int,
                 n_bks: int,
                 stat_funcs: list[callable]):
        """Multivariate time series changepoint detection wrapper

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
        """Detect changepoints in a given ndarray

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
            # false_split = (self.bks[-1] + self.bks[-2]) // 2
            # self.bks.insert(len(self.bks) - 1, false_split)

            raise ValueError(f"Algo couldn't find {self.n_bks} changepoints in the array")


    def split_array(self, array: np.ndarray) -> list[np.ndarray]:
        """Helper function to get a list of the splitted arrays

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
        """Helper function to calculate statistical properties of a segment

        Parameters
        ----------
        segment : np.ndarray
            Segment to which stats will be calculated

        Returns
        -------
        np.ndarray
            Flattened array of stats properties of the segment
        """

        if len(segment.shape) == 1:
            axis = None
        elif len(segment.shape) == 2:
            axis = 0

        stats = [f(segment, axis=axis) for f in self.stat_funcs]

        if axis is None:
            return stats
        elif axis == 0:
            return np.concatenate(stats)


    def process_segments(self, array: np.ndarray) -> np.ndarray:
        """Wrapper function to calculate changepoints and provide the final stats of the arrays

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
        """Helper function to plot the segment results

        Parameters
        ----------
        array : np.ndarray
            Array to be plotted
        """

        rpt.show.display(array, self.bks_idx)
        plt.show()