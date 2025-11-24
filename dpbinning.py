# Sourced from https://github.com/givasile/RHALE/tree/main/code/pythia

import typing
import numpy as np
import matplotlib.pyplot as plt

# global (script-level) variables
big_M = 1.0e10

def filter_points_in_bin(data: np.ndarray, data_effect: np.ndarray, limits: np.ndarray):
    filt = np.logical_and(limits[0] <= data, data <= limits[1])
    data_effect = data_effect[filt]
    data = data[filt]
    return data, data_effect

class BinBase:
    big_M = big_M

    def __init__(
        self,
        feature: int,
        xs_min: float,
        xs_max: float,
        data: typing.Union[None, np.ndarray],
        data_effect: typing.Union[None, np.ndarray],
        mu: typing.Union[None, callable],
        sigma: typing.Union[None, callable],
        axis_limits: typing.Union[None, np.ndarray] = None,
    ):
        """Initializer.

        Parameters
        ----------
        feature: feature index
        xs_min: min value on axis
        xs_max: max value on axis
        data: np.ndarray with X if binning is based on data, else, None
        data_effect: np.ndarray with dy/dX if binning is based on data, else, None
        mu: mu(xs) function
        sigma: sigma(xs) function if binning
        axis_limits: np.ndarray (2, D) or None, if it given it gets priority over xs_min, xs_max
        """
        # set immediately
        self.feature: int = feature
        self.xs_min: float = xs_min
        self.xs_max: float = xs_max

        self.data = data
        self.data_effect = data_effect
        self.mu = mu
        self.sigma = sigma
        self.axis_limits = axis_limits

        # set after execution of method `find`

        # min_points:
        # None -> .find() not executed or we don't care about min_points,
        # int -> min points per bin
        self.min_points: typing.Union[None, int] = None

        # limits:
        #  - None -> .find() not executed yet,
        #  - False -> .find() executed and didn't find acceptable solution
        #  - np.ndarray -> the limits
        self.limits: typing.Union[None, False, np.ndarray] = None

    def _bin_loss(self, start: float, stop: float):
        """Cost of creating the bin with limits [start, stop].

        If the bin contains less points than the specified min_points (min_points is not None)
        then big_M is passed as return value.

        Returns:
            cost of creating the particular bin
        """
        return NotImplementedError

    def _bin_valid(self, start: float, stop: float):
        """Whether the bin is valid.

        Returns a boolean value
        """
        return NotImplementedError

    def _none_bin_possible(self):
        """True, if non bin is possible

        Returns a boolean value
        """
        return NotImplementedError

    def _one_bin_possible(self):
        """True, if only one bin is possible

        Returns a boolean value
        """
        return NotImplementedError

    def find(self, *args):
        """Finds the optimal bins
        If it is possible to find a set of optimal bins, it returns the limits as np.ndarray
        If it is not, returns False

        """
        return NotImplementedError

    def plot(self, s=0, block=False):
        assert self.limits is not None
        limits = self.limits

        plt.figure()
        plt.title("Bin splitting for feature %d" % (s + 1))
        if self.data is not None:
            xs = self.data[:, self.feature]
            dy_dxs = self.data_effect[:, self.feature]
            plt.plot(xs, dy_dxs, "bo", label="local effects")
        elif self.mu is not None:
            xs = np.linspace(self.xs_min, self.xs_max, 1000)
            dy_dxs = self.mu(xs)
            plt.plot(xs, dy_dxs, "b-", label="mu(x)")

        y_min = np.min(dy_dxs)
        y_max = np.max(dy_dxs)
        plt.vlines(limits, ymin=y_min, ymax=y_max, linestyles="dashed", label="bins")
        plt.xlabel("x_%d" % (s + 1))
        plt.ylabel("dy/dx_%d" % (s + 1))
        plt.legend()
        plt.show(block=block)


class DPBase(BinBase):
    def __init__(
        self, feature, xs_min, xs_max, data, data_effect, mu, sigma, axis_limits
    ):

        # self.dx_list = None
        self.matrix = None
        self.argmatrix = None

        super(DPBase, self).__init__(
            feature, xs_min, xs_max, data, data_effect, mu, sigma, axis_limits
        )

    def _index_to_position(self, index_start, index_stop, K):
        dx = (self.xs_max - self.xs_min) / K
        start = self.xs_min + index_start * dx
        stop = self.xs_min + index_stop * dx
        return start, stop

    def _cost_of_move(self, index_before, index_next, K, discount):
        """Compute the cost of move.

        Computes the cost for moving from the index of the previous bin (index_before)
        to the index of the next bin (index_next).
        """

        big_M = self.big_M
        if index_before > index_next:
            cost = big_M
        elif index_before == index_next:
            cost = 0
        else:
            start, stop = self._index_to_position(index_before, index_next, K)
            cost = self._bin_loss(start, stop, discount)
        return cost

    def _argmatrix_to_limits(self, K):
        assert self.argmatrix is not None
        argmatrix = self.argmatrix
        dx = (self.xs_max - self.xs_min) / K

        lim_indices = [int(argmatrix[-1, -1])]
        for j in range(K - 2, 0, -1):
            lim_indices.append(int(argmatrix[int(lim_indices[-1]), j]))
        lim_indices.reverse()

        lim_indices.insert(0, 0)
        lim_indices.append(argmatrix.shape[-1])

        # remove identical bins
        lim_indices_1 = []
        before = np.nan
        for i, lim in enumerate(lim_indices):
            if before != lim:
                lim_indices_1.append(lim)
                before = lim

        limits = self.xs_min + np.array(lim_indices_1) * dx
        dx_list = np.array(
            [limits[i + 1] - limits[i] for i in range(limits.shape[0] - 1)]
        )
        return limits, dx_list

    def find(self, k_max: int = 30, min_points: int = 10, discount: float = 0.2):
        """

        Parameters
        ----------
        min_points: minimum points per bin
        k_max: maximum number of bins

        Returns
        -------

        """
        self.min_points = min_points
        big_M = self.big_M
        nof_limits = k_max + 1
        nof_bins = k_max

        if self._none_bin_possible():
            self.limits = False
        elif k_max == 1:
            self.limits = np.array([self.xs_min, self.xs_max])
            self.dx_list = np.array([self.xs_max - self.xs_min])
        else:
            # init matrices
            matrix = np.ones((nof_limits, nof_bins)) * big_M
            argmatrix = np.ones((nof_limits, nof_bins)) * np.nan

            # init first bin_index
            bin_index = 0
            for lim_index in range(nof_limits):
                matrix[lim_index, bin_index] = self._cost_of_move(
                    bin_index, lim_index, k_max, discount
                )

            # for all other bins
            for bin_index in range(1, k_max):
                for lim_index_next in range(k_max + 1):
                    # find best solution
                    tmp = []
                    for lim_index_before in range(k_max + 1):
                        tmp.append(
                            matrix[lim_index_before, bin_index - 1]
                            + self._cost_of_move(
                                lim_index_before, lim_index_next, k_max, discount
                            )
                        )
                    # store best solution
                    matrix[lim_index_next, bin_index] = np.min(tmp)
                    argmatrix[lim_index_next, bin_index] = np.argmin(tmp)

            # store solution matrices
            self.matrix = matrix
            self.argmatrix = argmatrix

            # find indices
            self.limits, _ = self._argmatrix_to_limits(k_max)

        return self.limits


class DP(DPBase):
    def __init__(self, data, data_effect, feature, axis_limits = None):
        xs_min = (
            data[:, feature].min() if axis_limits is None else axis_limits[0, feature]
        )
        xs_max = (
            data[:, feature].max() if axis_limits is None else axis_limits[1, feature]
        )
        self.nof_points = data.shape[0]
        self.feature = feature
        super(DP, self).__init__(
            feature, xs_min, xs_max, data, data_effect, None, None, axis_limits
        )

    def _none_bin_possible(self):
        dy_dxs = self.data_effect[:, self.feature]
        return dy_dxs.size < self.min_points

    def _bin_valid(self, start: float, stop: float) -> bool:
        data = self.data[:, self.feature]
        data_effect = self.data_effect[:, self.feature]
        data, data_effect = filter_points_in_bin(
            data, data_effect, np.array([start, stop])
        )
        if data_effect.size < self.min_points:
            valid = False
        else:
            valid = True
        return valid

    def _bin_loss(self, start, stop, discount):
        min_points = self.min_points
        data = self.data[:, self.feature]
        data_effect = self.data_effect[:, self.feature]
        data, data_effect = filter_points_in_bin(
            data, data_effect, np.array([start, stop])
        )

        # compute cost
        if data_effect.size < min_points:
            cost = self.big_M
            cost_var = self.big_M
        else:
            # cost = np.std(data_effect) * (stop-start) / np.sqrt(data_effect.size)
            discount_for_more_points = 1 - discount * (
                data_effect.size / self.nof_points
            )
            cost = np.var(data_effect) * (stop - start) * discount_for_more_points
            cost_var = np.var(data_effect)
        return cost