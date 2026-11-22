import numpy as np
import pandas as pd

def holm_bonferroni(pvalues):
    """
    Applies Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER).
    pvalues: list or 1D array of p-values
    Returns: array of adjusted p-values in original order
    """
    pvals = np.asarray(pvalues, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=float)
        
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    
    adj_sorted = np.zeros(n, dtype=float)
    running_max = 0.0
    for i in range(n):
        multiplier = n - i
        val = sorted_p[i] * multiplier
        running_max = max(running_max, val)
        adj_sorted[i] = min(1.0, running_max)
        
    # Reorder back to original
    adj_pvals = np.zeros(n, dtype=float)
    adj_pvals[order] = adj_sorted
    return adj_pvals

def adjust_dataframe_column(df, pval_col='pvalue', adj_col='pvalue_adj'):
    """
    Takes a DataFrame and returns a copy with adjusted p-values.
    """
    df_out = df.copy()
    if pval_col in df_out.columns and len(df_out) > 0:
        df_out[adj_col] = holm_bonferroni(df_out[pval_col].values)
    return df_out
