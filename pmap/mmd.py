# Maximum mean discrepancy metric with average Hamming distance kernel (variables are binary)
# http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBorRasSchetal12.pdf, Equation (5)

import numpy as np
from scipy.special import logsumexp


def logHK(S1, S2):
    # Note that Hamming(x, y) = <x, 1-y> + <1-x, y>
    return -np.hstack((1 - S1, S1)) @ np.hstack((S2, 1 - S2)).T / S1.shape[1]


def logMMDlinear(S1, S2):
    # Standard implementation
    assert (
        (S1 == 0).sum() + (S1 == 1).sum()
        == S1.size
        == (S2 == 0).sum() + (S2 == 1).sum()
        == S2.size
    )
    S1, S2 = S1.astype(np.float64), S2.astype(np.float64)
    return np.log(
        np.exp(logHK(S1, S1)).mean()
        + np.exp(logHK(S2, S2)).mean()
        - 2 * np.exp(logHK(S1, S2)).mean()
    )


def logMMD(S1, S2):
    # More stable implementation in log space
    assert (
        (S1 == 0).sum() + (S1 == 1).sum()
        == S1.size
        == (S2 == 0).sum() + (S2 == 1).sum()
        == S2.size
    )
    S1, S2 = np.array(S1).astype(np.float64), np.array(S2).astype(np.float64)
    log_pos = np.logaddexp(logsumexp(logHK(S1, S1)), logsumexp(logHK(S2, S2)))
    log_neg = logsumexp(logHK(S1, S2)) + np.log(2)
    return np.log1p(-np.exp(log_neg - log_pos)) + log_pos - 2 * np.log(S1.shape[0])
