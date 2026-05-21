import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import sys
import os

sys.path.append(os.path.abspath('.'))

from hrp_engine.data import fetch_data
from hrp_engine.hrp import get_quasi_diag, optimize_hrp

def trace_hrp(pool_name: str):
    prices_df = fetch_data(pool_name)
    # Forward/backward fill to get clean dataframe for trace
    prices_clean = prices_df.ffill().bfill()
    returns = prices_clean.pct_change().dropna()
    cov = returns.cov() * 252
    
    assets = list(cov.columns)
    N = len(assets)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(2.0 * np.clip(1.0 - corr.values, 0.0, 2.0))
    
    dist_cols = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((dist[i] - dist[j])**2))
            dist_cols[i, j] = d
            dist_cols[j, i] = d
            
    condensed_dist = squareform(dist_cols)
    Z = sch.linkage(condensed_dist, method='single')
    ordered_indices = get_quasi_diag(Z)
    sort_items = [assets[i] for i in ordered_indices]
    
    print(f"\n=================== TRACING HRP FOR {pool_name.upper()} ===================")
    print(f"Quasi-diagonalized leaf order:\n  {sort_items}\n")
    
    # Trace recursive bisection
    w = pd.Series(1.0, index=sort_items)
    queue = [(sort_items, 1.0, "Root")]
    
    steps = []
    while len(queue) > 0:
        current, weight_scale, path = queue.pop(0)
        if len(current) <= 1:
            steps.append((current[0], weight_scale, path))
            continue
            
        mid = len(current) // 2
        left = current[:mid]
        right = current[mid:]
        
        # Left cluster variance
        cov_left = cov.loc[left, left]
        diag_left = np.diag(cov_left)
        diag_left_safe = np.where(diag_left == 0, 1e-8, diag_left)
        w_left = 1.0 / diag_left_safe
        w_left /= w_left.sum()
        var_left = np.dot(w_left, np.dot(cov_left, w_left))
        
        # Right cluster variance
        cov_right = cov.loc[right, right]
        diag_right = np.diag(cov_right)
        diag_right_safe = np.where(diag_right == 0, 1e-8, diag_right)
        w_right = 1.0 / diag_right_safe
        w_right /= w_right.sum()
        var_right = np.dot(w_right, np.dot(cov_right, w_right))
        
        alpha = 1.0 - var_left / (var_left + var_right) if (var_left + var_right) > 0 else 0.5
        
        print(f"Split Node at path '{path}':")
        print(f"  Left Cluster ({len(left)} assets): {left}")
        print(f"    Inverse-Var Portfolio Vol: {np.sqrt(var_left)*100:.2f}%")
        print(f"  Right Cluster ({len(right)} assets): {right}")
        print(f"    Inverse-Var Portfolio Vol: {np.sqrt(var_right)*100:.2f}%")
        print(f"  Allocation Ratio (Alpha / Weight to Left): {alpha:.4f} (Left: {alpha*100:.1f}%, Right: {(1-alpha)*100:.1f}%)")
        print(f"  Parent Weight: {weight_scale*100:.2f}% -> Left: {weight_scale*alpha*100:.2f}%, Right: {weight_scale*(1-alpha)*100:.2f}%\n")
        
        queue.append((left, weight_scale * alpha, path + " -> Left"))
        queue.append((right, weight_scale * (1.0 - alpha), path + " -> Right"))
        
    print("Final Weights:")
    final_w = pd.Series({item: w_scale for item, w_scale, _ in steps})
    print(final_w.apply(lambda x: f"{x:.2%}"))

trace_hrp('european')
trace_hrp('etf')
