import numpy as np
from scipy import stats

def newey_west_covariance(X, bandwidth=None, kernel='bartlett'):
    """
    Computes the Newey-West HAC spectral covariance matrix at frequency zero:
        Omega = Gamma_0 + sum_{j=1}^L w(j, L) * (Gamma_j + Gamma_j')
    X: shape (T, K)
    bandwidth: integer lag truncation L (default: int(4 * (T / 100)**(2/9)))
    kernel: 'bartlett' or 'uniform'
    Returns: Omega (K, K)
    """
    X = np.asarray(X, dtype=float)
    T, K = X.shape
    
    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    bandwidth = max(0, int(bandwidth))
    
    # Demean moments
    u = X - np.mean(X, axis=0, keepdims=True)
    
    # Gamma_0
    gamma_0 = (u.T @ u) / T
    omega = gamma_0.copy()
    
    for j in range(1, bandwidth + 1):
        gamma_j = (u[j:].T @ u[:-j]) / T
        if kernel == 'bartlett':
            w = 1.0 - (j / (bandwidth + 1.0))
        elif kernel == 'uniform':
            w = 1.0
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        omega += w * (gamma_j + gamma_j.T)
        
    return omega

def wald_test(theta_hat, V_hat, R=None, r=None):
    """
    Computes Wald statistic for H0: R * theta = r:
        W = (R * theta_hat - r)' [R * V_hat * R']^{-1} (R * theta_hat - r)
    theta_hat: shape (K,)
    V_hat: covariance of theta_hat, shape (K, K)
    R: contrast matrix, shape (q, K). Default: identity matrix (q=K, r=0)
    Returns: dict with 'stat', 'pvalue', 'df', 'rank'
    """
    theta_hat = np.asarray(theta_hat, dtype=float).reshape(-1, 1)
    K = theta_hat.shape[0]
    
    if R is None:
        R = np.eye(K)
    else:
        R = np.asarray(R, dtype=float)
        
    q = R.shape[0]
    if r is None:
        r = np.zeros((q, 1), dtype=float)
    else:
        r = np.asarray(r, dtype=float).reshape(-1, 1)
        
    diff = (R @ theta_hat) - r
    middle = R @ V_hat @ R.T
    
    # Regularized inversion / pseudo-inverse for numerical stability
    try:
        middle_inv = np.linalg.inv(middle)
        rank = q
    except np.linalg.LinAlgError:
        middle_inv = np.linalg.pinv(middle)
        rank = int(np.linalg.matrix_rank(middle))
        
    stat = float((diff.T @ middle_inv @ diff)[0, 0])
    stat = max(0.0, stat)
    pvalue = float(1.0 - stats.chi2.cdf(stat, df=rank))
    
    return {
        'stat': stat,
        'pvalue': pvalue,
        'df': rank,
        'q': q
    }
