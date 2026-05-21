import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform, pdist
from typing import List

def get_quasi_diag(link: np.ndarray) -> List[int]:
    """
    Computes the quasi-diagonalized order of elements from the linkage tree.
    This corresponds to the leaf order in the dendrogram.
    """
    link = link.astype(int)
    # The last merge cluster size is the number of original items
    num_items = link[-1, 3]
    
    # Start with the two children of the root node
    sort_obs = [link[-1, 0], link[-1, 1]]
    
    # Expand clusters recursively (indices >= num_items represent cluster nodes)
    while any(x >= num_items for x in sort_obs):
        new_sort = []
        for x in sort_obs:
            if x >= num_items:
                # x is a cluster node, look up its children
                idx = x - num_items
                new_sort.extend([link[idx, 0], link[idx, 1]])
            else:
                new_sort.append(x)
        sort_obs = new_sort
        
    return sort_obs

def recursive_bisection_index(cov: pd.DataFrame, sort_items: List[str]) -> pd.Series:
    """
    Allocates portfolio weights recursively using the original index-based bisection.
    Splits the leaf-ordered list strictly in half by index.
    """
    w = pd.Series(1.0, index=sort_items)
    queue = [sort_items]
    
    while len(queue) > 0:
        current = queue.pop(0)
        if len(current) <= 1:
            continue
            
        # Split the cluster in half by index
        mid = len(current) // 2
        left = current[:mid]
        right = current[mid:]
        
        # Calculate left cluster variance
        cov_left = cov.loc[left, left]
        diag_left = np.diag(cov_left)
        diag_left_safe = np.where(diag_left == 0, 1e-8, diag_left)
        w_left = 1.0 / diag_left_safe
        w_left /= w_left.sum()
        var_left = np.dot(w_left, np.dot(cov_left, w_left))
        
        # Calculate right cluster variance
        cov_right = cov.loc[right, right]
        diag_right = np.diag(cov_right)
        diag_right_safe = np.where(diag_right == 0, 1e-8, diag_right)
        w_right = 1.0 / diag_right_safe
        w_right /= w_right.sum()
        var_right = np.dot(w_right, np.dot(cov_right, w_right))
        
        # Calculate split factor alpha
        if var_left + var_right > 0:
            alpha = 1.0 - var_left / (var_left + var_right)
        else:
            alpha = 0.5
            
        # Scale weights of each sub-cluster
        w[left] *= alpha
        w[right] *= (1.0 - alpha)
        
        # Append sub-clusters to the queue for further bisection
        queue.append(left)
        queue.append(right)
        
    return w

def recursive_bisection_tree(cov: pd.DataFrame, Z: np.ndarray, assets: List[str]) -> pd.Series:
    """
    Allocates portfolio weights recursively along the actual branches of the linkage tree Z.
    This resolves issues with unbalanced trees where index-based bisection starves assets.
    """
    N = len(assets)
    w = pd.Series(1.0, index=assets)
    
    # Map node index to list of asset names under that node
    node_to_assets = {}
    for i in range(N):
        node_to_assets[i] = [assets[i]]
        
    for i in range(N - 1):
        left_child = int(Z[i, 0])
        right_child = int(Z[i, 1])
        node_idx = N + i
        node_to_assets[node_idx] = node_to_assets[left_child] + node_to_assets[right_child]
        
    # Queue stores node indices, starting with the root node (2*N - 2)
    queue = [2 * N - 2]
    
    while len(queue) > 0:
        current_node = queue.pop(0)
        if current_node < N:
            continue
            
        left_child = int(Z[current_node - N, 0])
        right_child = int(Z[current_node - N, 1])
        
        left_assets = node_to_assets[left_child]
        right_assets = node_to_assets[right_child]
        
        # Calculate left cluster variance
        cov_left = cov.loc[left_assets, left_assets]
        diag_left = np.diag(cov_left)
        diag_left_safe = np.where(diag_left == 0, 1e-8, diag_left)
        w_left = 1.0 / diag_left_safe
        w_left /= w_left.sum()
        var_left = np.dot(w_left, np.dot(cov_left, w_left))
        
        # Calculate right cluster variance
        cov_right = cov.loc[right_assets, right_assets]
        diag_right = np.diag(cov_right)
        diag_right_safe = np.where(diag_right == 0, 1e-8, diag_right)
        w_right = 1.0 / diag_right_safe
        w_right /= w_right.sum()
        var_right = np.dot(w_right, np.dot(cov_right, w_right))
        
        # Calculate split factor alpha
        if var_left + var_right > 0:
            alpha = 1.0 - var_left / (var_left + var_right)
        else:
            alpha = 0.5
            
        # Scale weights of each sub-cluster
        w[left_assets] *= alpha
        w[right_assets] *= (1.0 - alpha)
        
        # Queue children for further bisection
        queue.append(left_child)
        queue.append(right_child)
        
    return w

def optimize_hrp(cov: pd.DataFrame, linkage_method: str = 'single', bisection_method: str = 'tree') -> pd.Series:
    """
    Main HRP optimization entry point.
    Computes the distance matrix, performs hierarchical clustering,
    reorders the covariance matrix (if index-based), and allocates weights.
    
    Parameters:
    - cov: Covariance DataFrame (N x N)
    - linkage_method: 'single', 'complete', or 'ward'
    - bisection_method: 'index' (original Lopez de Prado) or 'tree' (tree-based)
    """
    assets = list(cov.columns)
    N = len(assets)
    if N == 0:
        return pd.Series(dtype=float)
    if N == 1:
        return pd.Series(1.0, index=assets)
        
    # Calculate standard deviations
    std = np.sqrt(np.diag(cov))
    std_safe = np.where(std == 0, 1e-8, std)
    
    # Calculate correlation matrix
    corr = cov / np.outer(std_safe, std_safe)
    corr = np.clip(corr, -1.0, 1.0)
    
    # Convert correlation to distance matrix: D = sqrt(0.5 * (1 - rho))
    # This matches Lopez de Prado's original formulation exactly.
    dist = np.sqrt(0.5 * np.clip(1.0 - corr.values, 0.0, 2.0))
    
    # Compute Euclidean distance between columns of the distance matrix (fully vectorized)
    dist_cols = squareform(pdist(dist, metric='euclidean'))
    
    # Compute linkage matrix
    condensed_dist = squareform(dist_cols)
    Z = sch.linkage(condensed_dist, method=linkage_method)
    
    # Compute weights via recursive bisection
    if bisection_method == 'tree':
        w = recursive_bisection_tree(cov, Z, assets)
    else:
        # Get the quasi-diagonalized order
        ordered_indices = get_quasi_diag(Z)
        sort_items = [assets[i] for i in ordered_indices]
        w = recursive_bisection_index(cov, sort_items)
    
    # Reindex to match the original assets order
    w = w.reindex(assets)
    
    return w

