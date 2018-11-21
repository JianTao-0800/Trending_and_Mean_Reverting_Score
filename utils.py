import numpy as np

def trend_score(x, alpha=3.0):
    """
    Trend Score

    Z in [-100, 100] by construction, and only realizes the extremal scores in the event that x is in
    a line. Scores near 0 exhibit little to no trend characteristics.

    Args:
        x(list): the values of the time series data.
        alpha(float): instead output rho, the final trend score is rho with power alpha.
    
    Returns:
        int: trend score

    Examples:
    --------
    >>> x1 = [1,2,3,4,5,6]
    >>> trend_score(x1)
    100
    >>> x2 = [6,5,4,3,2,1]
    >>> trend_score(x2)
    -100
    >>> x3 = [-1,1,-1,1,-1,1,-1]
    >>> trend_score(x3)
    0
    >>> x4 = [1,3,2,4,3,5]
    >>> trend_score(x4)
    57
    >>> trend_score(x4, alpha=1)
    83
    >>> x5 = [i**2 for i in range(0,1000)]
    >>> trend_score(x5, alpha=3)
    91
    >>> x6 = [i**2 for i in range(300,1000)]
    >>> trend_score(x6, alpha=3)
    97
    >>> x7 = [i**2 for i in range(600,1000)]
    >>> trend_score(x7, alpha=3)
    99
    """
    x = [v for v in x if not np.isnan(v)]
    n = len(x)
    x_m = sum(x)/float(n)
    t_m = (1+n)/2.0
    numr = sum([(x[i]-x_m)*(i+1-t_m) for i in range(n)])
    deno = np.sqrt(sum([(x[i]-x_m)**2 for i in range(n)]) *
                   sum([(i+1-t_m)**2 for i in range(n)]))
    rho = numr/deno
    Z = int(round(100 * np.sign(rho) * abs(rho)**alpha, 0))
    return Z

def mean_rev_score(x, beta=15.0):
    """
    Mean Reversion Score
    
    This mean reversion score is only computed for a time series if the associated trend score is
    between -25 and 25.
        
    Args:
        x(list): the values of the time series data.
        k(float): chosen in order to force the mean reversion and trend scores to be approximately
          comparable.

    Returns:
        int: mean reversion score.
    
    Examples:
    --------
    >>> x1 = [-1,1,-1,1,-1]
    >>> trend_score(x1)
    0
    >>> mean_rev_score(x1)
    46
    >>> x2 = [-1,1,-1,1,-1,1,-1]
    >>> mean_rev_score(x2)
    61
    >>> x3 = [1 if i//2 else -1 for i in range(0,1000)]
    >>> mean_rev_score(x3)
    98
    >>> x4 = [1,3,2,4,3,5]
    >>> mean_rev_score(x4)
    23
    """
    x = [v for v in x if not np.isnan(v)]
    n = len(x)
    qv = sum([(x[i]-x[i-1])**2 for i in range(1,n)])
    x_m = sum(x)/float(n)
    x_s = sum([(x[i]-x_m)**2 for i in range(n)]) / (n-1)
    Z = int(round(100 * 2**(-beta*x_s/qv)))
    return Z


if __name__ == "__main__":
    import doctest
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(filename)s Line %(lineno)s %(funcName)s: %(message)s')
    doctest.testmod()
