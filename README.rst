stochvar
========

Stochastic variables.

    >>> from stochvar import Normal, Dirac
    >>> n1 = Normal(2, 2)
    >>> n2 = Normal()
    >>> d1 = Dirac(3)
    >>> n3 = n1 * d1
    >>> n3.location
    6.0

Plotting:

    >>> import matplotlib.pyplot as plt
    >>> plt.ion()
    >>> n3.plot_pdf(linewidth=10, alpha=0.5)
    >>> plt.hold(True)
    >>> empirical = [n1.generate() * d1.generate() for n in range(10000)]
    >>> _ = plt.hist(empirical, bins=100, normed=True)
