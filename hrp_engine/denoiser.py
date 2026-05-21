import numpy as np
import pandas as pd
from typing import Tuple

def denoise_covariance(cov_emp: np.ndarray, n_obs: int) -> np.ndarray:
    """
    Denoises the empirical covariance matrix using the Marchenko-Pastur theorem.
    
    Parameters:
    - cov_emp: Empirical covariance matrix (N x N)
    - n_obs: Number of historical observations used to compute cov_emp (T_obs)
    
    Returns:
    - denoised_cov: Denoised covariance matrix (N x N)
    """
    N = cov_emp.shape[0]
    if N <= 1:
        return cov_emp.copy()
        
    # Extract standard deviations
    std = np.sqrt(np.diag(cov_emp))
    # Replace zeros with epsilon to avoid division by zero
    std_safe = np.where(std == 0, 1e-8, std)
    
    # Calculate empirical correlation matrix C_emp
    corr_emp = cov_emp / np.outer(std_safe, std_safe)
    # Clip to avoid floating point anomalies out of [-1, 1]
    corr_emp = np.clip(corr_emp, -1.0, 1.0)
    
    # Eigen decomposition (using eigh since corr_emp is symmetric)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_emp)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Isolate noise variance sigma^2 by finding the variance of eigenvalues <= 1.0
    eigenvalues_lte_1 = eigenvalues[eigenvalues <= 1.0]
    if len(eigenvalues_lte_1) > 0:
        sigma2 = np.mean(eigenvalues_lte_1)
    else:
        sigma2 = 1.0
        
    # Avoid zero variance
    if sigma2 <= 0:
        sigma2 = 1e-8
        
    # Compute the analytical noise cutoff boundary lambda_max
    q = N / n_obs
    lambda_max = sigma2 * (1.0 + np.sqrt(q))**2
    
    # Find noise-dominated eigenvalues (eigenvalues <= lambda_max)
    is_noise = eigenvalues <= lambda_max
    n_noise = np.sum(is_noise)
    
    if n_noise > 0:
        # Replace noise-dominated eigenvalues with their average, preserving the trace
        average_noise_eigenvalue = np.mean(eigenvalues[is_noise])
        eigenvalues_denoised = eigenvalues.copy()
        eigenvalues_denoised[is_noise] = average_noise_eigenvalue
    else:
        eigenvalues_denoised = eigenvalues.copy()
        
    # Reconstruct denoised correlation matrix
    corr_denoised_raw = np.dot(eigenvectors * eigenvalues_denoised, eigenvectors.T)
    
    # Rescale diagonal to 1.0 to preserve correlation matrix properties.
    # Clip diagonal elements to a small positive epsilon (1e-8) to prevent division by zero or NaN due to precision.
    diag_val = np.diag(corr_denoised_raw)
    diag_val_safe = np.clip(diag_val, 1e-8, None)
    diag_inv_sqrt = 1.0 / np.sqrt(diag_val_safe)
    corr_denoised = corr_denoised_raw * np.outer(diag_inv_sqrt, diag_inv_sqrt)
    corr_denoised = np.clip(corr_denoised, -1.0, 1.0)
    
    # Reconstruct denoised covariance matrix using original standard deviations
    cov_denoised = corr_denoised * np.outer(std, std)
    
    return cov_denoised
