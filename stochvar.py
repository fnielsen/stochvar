"""Stochastic variables."""

import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats
import numpy as np


class NotImplementedYet(Exception):

    """Exception for combinations not yet implemented."""

    pass


class StochasticVariable(object):

    """SV."""

    def generate(self, size=1):
        """Generate realizations of the stochastic variable."""
        raise NotImplementedYet


    def plot_pdf(self, **kwargs):
        """Plot the probability density function."""
        raise NotImplementedYet


class UnivariateStochasticVariable(StochasticVariable):

    """Univariate stochastic variable."""
    
    pass


class UnivariateNormal(UnivariateStochasticVariable):

    """Univariate normal distribution.

    Attributes
    ----------
    location : float
        Center value
    scale : float
        Standard deviation

    """

    def __init__(self, location=0.0, scale=1.0):
        """Set location and scale for normal distribution."""
        self.location = float(location)
        self.scale = float(scale)

    def __add__(self, other):
        """Add two independent normal distibutions.
        
        Examples
        --------
        >>> a = UnivariateNormal()
        >>> b = UnivariateNormal()
        >>> c = a + b
        >>> c.location
        0.0
        >>> c.scale > 1.4
        True
        
        """
        if isinstance(other, UnivariateNormal):
            location = self.location + other.location
            scale = sqrt(self.scale**2 + other.scale**2)
            return UnivariateNormal(location, scale)
        else:
            raise NotImplementedYet

    def generate(self, size=1):
        """Generate realization of normal distribution."""
        return stats.norm.rvs(loc=self.location, scale=self.scale, size=size)

    def plot_pdf(self, **kwargs):
        """Plot probability distribution for normal distribution."""
        x_low = self.location - 5 * self.scale
        x_high = self.location + 5 * self.scale
        x = np.linspace(x_low, x_high, 100)
        pdf = stats.norm.pdf(x, loc=self.location, scale=self.scale)
        plt.plot(x, pdf, **kwargs)
        

class UnivariateDirac(UnivariateStochasticVariable):

    """Dirac distribution."""
    
    def __init__(self, location=0.0):
        """Set location for Dirac distribution."""
        self.location = float(location)

    def __add__(self, other):
        """Add Dirac distribution with another distribution."""
        if isinstance(other, UnivariateDirac):
            location = self.location + other.location
            return UnivariateDirac(location)
        else:
            raise NotImplementedYet

    def __mul__(self, other):
        """Multiple a Dirac distribution with another distribution.

        Examples
        --------
        >>> d1 = UnivariateDirac(-3)
        >>> d2 = UnivariateDirac(6)
        >>> d3 = d1 * d2
        >>> d3.location
        -18.0

        """
        if isinstance(other, UnivariateDirac):
            location = self.location * other.location
            return UnivariateDirac(location)
        else:
            raise NotImplementedYet

    def generate(self, size=1):
        """Generate realizations of Dirac distribution.
        
        Examples
        --------
        >>> d = UnivariateDirac()
        >>> d.generate()
        0.0

        """
        if size == 1:
            return self.location
        else: 
            return self.location * np.ones(size)

    def plot_pdf(self, **kwargs):
        """Plot Dirac distribution."""
        plt.plot([self.location, self.location], [0.0, 1.0])
