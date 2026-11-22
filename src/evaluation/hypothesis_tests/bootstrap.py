import numpy as np

def generate_moving_block_indices(T, block_length, rng=None):
    """
    Generates time indices for a single paired moving-block bootstrap sample.
    T: length of the time series
    block_length: length of each continuous block b
    Returns: 1D array of indices of length T
    """
    if rng is None:
        rng = np.random.default_rng()
        
    num_blocks = int(np.ceil(T / block_length))
    max_start = T - block_length
    if max_start < 0:
        return np.arange(T)
        
    starts = rng.integers(0, max_start + 1, size=num_blocks)
    indices = []
    for s in starts:
        indices.extend(range(s, s + block_length))
    return np.array(indices[:T], dtype=int)

def paired_block_bootstrap_matrix(X, block_length=20, n_boot=2000, seed=42):
    """
    Generates B bootstrap replicated matrices using paired moving-block bootstrap.
    X: shape (T, K)
    block_length: block length b
    n_boot: number of bootstrap replications
    seed: fixed random seed for reproducibility
    Yields: X_star (T, K)
    """
    X = np.asarray(X, dtype=float)
    T, K = X.shape
    rng = np.random.default_rng(seed)
    
    for _ in range(n_boot):
        idx = generate_moving_block_indices(T, block_length, rng=rng)
        yield X[idx]
