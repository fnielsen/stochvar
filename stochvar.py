"""Stochastic variables and algebra.

Example
-------
>>> plt.ion()
>>> n = Normal()
>>> n.plot_pdf_from_generated()
>>> n.plot_pdf(linewidth=3)
>>> plt.show()

"""


import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats
import numpy as np


class StochasticVariable(object):

    """Abstract class for stochastic variables."""

    def generate(self):
        """Generate realizations of the stochastic variable."""
        raise NotImplementedError

    def plot_pdf(self, **kwargs):
        """Plot the probability density function."""
        raise NotImplementedError

    def plot_pdf_from_generated(self, samples=10000, **kwargs):
        """Generate a histogram of samples."""
        data = [self.generate() for _ in range(samples)]
        plt.hist(data, bins=round(sqrt(samples)), normed=True, **kwargs)


class Normal(StochasticVariable):

    """Normal distribution.

    Attributes
    ----------
    location : float
        Center value
    scale : float
        Standard deviation

    Example
    -------
    >>> n = Normal()
    >>> n.location
    0.0
    >>> isinstance(n, StochasticVariable)
    True

    """

    def __init__(self, location=0.0, scale=1.0):
        """Set location and scale for normal distribution.

        Example
        -------
        >>> n = Normal(scale=2.0)
        >>> n.scale
        2.0

        """
        self.location = float(location)
        self.scale = float(scale)

    def __add__(self, other):
        """Add two independent normal distibutions.

        Examples
        --------
        >>> n1 = Normal()
        >>> n2 = Normal()
        >>> n3 = n1 + n2
        >>> n3.location
        0.0
        >>> n3.scale > 1.4
        True
        >>> n4 = n1 + 3
        >>> n4.location
        3.0

        """
        if isinstance(other, Normal):
            location = self.location + other.location
            scale = sqrt(self.scale**2 + other.scale**2)
            return Normal(location, scale)
        elif isinstance(other, int) or isinstance(other, float):
            location = self.location + other
            return Normal(location, self.scale)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        """Multiply stochastical variable."""
        if isinstance(other, Dirac):
            location = self.location * other.location
            scale = self.scale * other.location
            return Normal(location, scale)
        else:
            raise NotImplementedError

    def generate(self):
        """Generate realization of normal distribution.

        Example
        -------
        >>> n = Normal()
        >>> type(n.generate()) is float
        True

        """
        return stats.norm.rvs(loc=self.location, scale=self.scale)

    def plot_pdf(self, **kwargs):
        """Plot probability distribution for normal distribution."""
        x_low = self.location - 5 * self.scale
        x_high = self.location + 5 * self.scale
        x = np.linspace(x_low, x_high, 100)
        pdf = stats.norm.pdf(x, loc=self.location, scale=self.scale)
        plt.plot(x, pdf, **kwargs)


class Cauchy(StochasticVariable):

    """Cauchy distribution."""

    def __init__(self, location=0.0, scale=1.0):
        """Set location for Dirac distribution.

        Example
        -------
        >>> c = Cauchy(3)
        >>> c.location
        3.0

        """
        self.location = float(location)
        self.scale = scale

    def generate(self):
        """Generate realization of Cauchy distribution.

        Example
        -------
        >>> c = Cauchy()
        >>> type(c.generate()) is float
        True

        """
        # 'rvs' returns numpy.float64
        return float(stats.cauchy.rvs(loc=self.location, scale=self.scale))

    def plot_pdf(self, **kwargs):
        """Plot probability distribution for Cauchy distribution."""
        x_low = self.location - 50 * self.scale
        x_high = self.location + 50 * self.scale
        x = np.linspace(x_low, x_high, 100)
        pdf = stats.cauchy.pdf(x, loc=self.location, scale=self.scale)
        plt.plot(x, pdf, **kwargs)


class Dirac(StochasticVariable):

    """Dirac distribution.

    Example
    -------
    >>> d = Dirac()
    >>> isinstance(d, StochasticVariable)
    True

    """

    def __init__(self, location=0.0):
        """Set location for Dirac distribution.

        Example
        -------
        >>> d = Dirac(3)
        >>> d.location
        3.0

        """
        self.location = float(location)

    def __add__(self, other):
        """Add Dirac distribution with another distribution.

        Example
        -------
        >>> d1 = Dirac(2)
        >>> d2 = Normal(1, 2)
        >>> d3 = d1 + d2
        >>> d3.location
        3.0
        >>> d3.scale
        2.0

        """
        if isinstance(other, Dirac):
            location = self.location + other.location
            return Dirac(location)
        elif isinstance(other, Normal):
            location = self.location + other.location
            scale = other.scale
            return Normal(location, scale)
        else:
            raise NotImplementedError

    def __div__(self, other):
        """Divide a Dirac distribution with another distribution.

        Example
        -------
        >>> d1 = Dirac(-3)
        >>> d2 = Dirac(6)
        >>> d3 = d1 / d2
        >>> d3.location
        -0.5

        """
        if isinstance(other, Dirac):
            location = self.location / other.location
            return Dirac(location)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        """Multiple a Dirac distribution with another distribution.

        Examples
        --------
        >>> d1 = Dirac(-3)
        >>> d2 = Dirac(6)
        >>> d3 = d1 * d2
        >>> d3.location
        -18.0

        >>> d4 = Normal(scale=2.0)
        >>> d5 = d1 * d4
        >>> d5.scale
        6.0
        >>> abs(d5.location)
        0.0

        """
        if isinstance(other, Dirac):
            location = self.location * other.location
            return Dirac(location)
        elif isinstance(other, Normal):
            if self.location == 0:
                return Dirac()
            else:
                location = self.location * other.location
                scale = abs(self.location * other.scale)
                return Normal(location, scale)
        else:
            raise NotImplementedError

    def __pow__(self, other):
        """Take the power.

        Example
        -------
        >>> d1 = Dirac(2)
        >>> d2 = d1 ** 3
        >>> d2.location
        8.0

        >>> d3 = Dirac(-2)
        >>> d4 = d1 ** d3
        >>> d4.location
        0.25

        """
        if isinstance(other, int):
            location = self.location ** other
            return Dirac(location)
        elif isinstance(other, Dirac):
            location = self.location ** other.location
            return Dirac(location)
        else:
            raise NotImplementedError

    def generate(self):
        """Generate realizations of Dirac distribution.

        Examples
        --------
        >>> d = Dirac()
        >>> d.generate()
        0.0

        """
        return self.location

    def plot_pdf(self, **kwargs):
        """Plot Dirac distribution."""
        plt.plot([self.location, self.location], [0.0, 1.0], **kwargs)
