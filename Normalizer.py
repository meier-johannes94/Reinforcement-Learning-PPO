import numpy as np


class Normalizer:
    # change: Autocheck for code quality
    """A class supporting online normalization"""  # change
    """
    Attributes:
        absolute_max_value : float
            Maximum calculated normalized value. Especially at the
            beginning when only a few samples are available the standard
            deviation can become very small. This can lead to a very
            large normalized values. To dampen this effect absolute
            values larger than absolute_max_value are cutt off.
        dim : int
            Dimensions of the variable to be normalized.

    Methods
        add(float):
            Recalculate mean and standard deviation also taking into
            account the new number.
        normalize(float):
            Normalize the parameter with the most recent mean and
            standard deviation and return the result.
        add_and_normalize(float):
            First add the number, then normalize it.
    """

    def __init__(self, dim, absolute_max_value):
        """
        Args:
            dim : int
                Dimension of the variable to be normalized
            absolute_max_value : float
                Maximum calculated normalized value. Especially at the
                beginning when only a few samples are available the standard
                deviation can become very small. This can lead to a very
                large normalized values. To dampen this effect absolute
                values larger than absolute_max_value are cutt off.
        """
        self._n = 0
        if dim == 1:
            self._m = 0.0
            self._s = 0.0
            self._t = 0.0
            self._mean = 0.0
            self._std = 1.0
        else:
            self._m = np.zeros(dim, dtype=float)
            self._s = np.zeros(dim, dtype=float)
            self._t = np.zeros(dim, dtype=float)
            self._mean = np.zeros(dim, dtype=float)
            self._std = np.ones(dim, dtype=float)
        self.absolute_max_value = absolute_max_value
        self.dim = dim

    def add(self, x):
        """
        Calculate the mean using the following formula:
        1. j <- j+1
        2. T_j = T_{j-1} + x_j
        3. Mean_{j} = T_j / j

        Calculate the standard deviation using the following formula
        for each dimension:
        1. j <- j+1
        2. M_{j} = M_{j-1} + (x-mean{j-1}) * (x-mean{j})
        3. Std_{j} = M_{j} / n

        Source:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_varianc
        e - Welford's online algorithm - Numerically optimized version

            Parameters
            ----------
                x : float / float[dim]

            Returns
            -------
                None
        """
        if self.dim > 1 and not isinstance(x, np.floating):
            x = np.array(x, dtype=float)

        if self._n == 0:
            self._n = 1
            self._t = x.copy() if self.dim > 1 else x
            self._mean = x.copy() if self.dim > 1 else x
        else:
            self._n += 1

            self._t += x
            previous_mean = self._mean.copy() if self.dim > 1 else self._mean
            self._mean = self._t / self._n
            self._m = self._m + (x-previous_mean) * (x-self._mean)
            self._std = np.sqrt(self._m / (self._n))

    def normalize(self, x):
        """Normalize x using mean and std.
        1.Centered_X = X - Mean

        2.Normalized_X = Centered_X / (Std + 1e-6)

        3.Clipped_Normalized_X = Np.clip(
            Normalized_X,
            -absolute_max_value,
            absolute_max_value)


        Args:
            x (float): Value to be normalized

        Returns:
            float: Normalized version of x
        """
        if self.dim > 1 and not isinstance(x, np.floating):
            x = np.array(x, dtype=float)

        return np.clip(
            (x-self._mean) / (self._std+1e-6),
            -self.absolute_max_value,
            self.absolute_max_value)

    def add_and_normalize(self, x):
        """First add x, then return the normalized version of x.

        Args:
            x (float): Value to be added and to be normalized.

        Returns:
            float: Normalized version of x
        """
        if self.dim > 1 and not isinstance(x, np.floating):
            x = np.array(x, dtype=float)

        self.add(x)
        return self.normalize(x)
