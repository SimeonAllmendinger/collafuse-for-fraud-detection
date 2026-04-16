import numpy as np
from scipy.stats import ks_2samp, entropy
from sklearn.metrics.pairwise import rbf_kernel


def evaluate_ks(real_samples: np.ndarray, generated_samples: np.ndarray):
    """
    Computes Kolmogorov-Smirnov (KS) statistic for each feature
    between real and generated samples.
    Returns a dictionary of {feature_index: ks_statistic}.
    """
    assert real_samples.shape[1] == generated_samples.shape[1], f"Feature dimensions must match. Got {
        real_samples.shape[1]} and {generated_samples.shape[1]}."
    ks_results = {}
    for i in range(real_samples.shape[1]):
        stat, _ = ks_2samp(real_samples[:, i], generated_samples[:, i])
        ks_results[i] = stat
    return ks_results


def evaluate_kl(real_samples: np.ndarray, generated_samples: np.ndarray, num_bins: int = 50):
    """
    Computes KL divergence for each feature between real and generated samples
    using discretized histograms.
    Returns a dictionary of {feature_index: kl_divergence}.
    """

    assert real_samples.shape[1] == generated_samples.shape[1], f"Feature dimensions must match. Got {
        real_samples.shape[1]} and {generated_samples.shape[1]}."
    kl_results = {}
    for i in range(real_samples.shape[1]):
        real_hist, bins = np.histogram(real_samples[:, i], bins=num_bins, density=True)
        gen_hist, _ = np.histogram(generated_samples[:, i], bins=bins, density=True)

        real_hist += 1e-8
        gen_hist += 1e-8

        kl_div = entropy(real_hist, gen_hist)
        kl_results[i] = kl_div
    return kl_results


def evaluate_mmd(
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    batch_size: int,
    seed: int,
    kernel_gamma: float = None
):
    """
    Computes MMD between real and generated samples using an RBF kernel.
    Automatically switches to batched/subsampled mode for large inputs.

    Args:
        real_samples (np.ndarray): Real data samples (N_real, D_features)
        generated_samples (np.ndarray): Generated data samples (N_gen, D_features)
        kernel_gamma (float, optional): Gamma parameter for the RBF kernel.
                                        If None, uses heuristic (1 / num_features).
        batch_size (int): Number of samples per batch for approximation.
        seed (int): Random seed for reproducibility.

    Returns:
        float: Estimated MMD value.
    """
    assert real_samples.shape[1] == generated_samples.shape[1], f"Feature dimensions must match. Got {
        real_samples.shape[1]} and {generated_samples.shape[1]}."

    if kernel_gamma is None:
        # Heuristic: 1 / num_features
        kernel_gamma = 1.0 / real_samples.shape[1]
    n_real, n_gen = len(real_samples), len(generated_samples)
    # For clients that has data points less than batch_size
    if n_real <= batch_size and n_gen <= batch_size:
        K_xx = rbf_kernel(real_samples, real_samples, gamma=kernel_gamma)
        K_yy = rbf_kernel(generated_samples, generated_samples, gamma=kernel_gamma)
        K_xy = rbf_kernel(real_samples, generated_samples, gamma=kernel_gamma)
        mmd_squared = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
        return float(np.sqrt(max(0, mmd_squared)))
    rng = np.random.default_rng(seed)
    mmd_values = []

    idx_real = rng.choice(n_real, size=min(batch_size, n_real))
    idx_gen = rng.choice(n_gen, size=min(batch_size, n_gen))
    real_sub = real_samples[idx_real]
    gen_sub = generated_samples[idx_gen]
    K_xx = rbf_kernel(real_sub, real_sub, gamma=kernel_gamma)
    K_yy = rbf_kernel(gen_sub, gen_sub, gamma=kernel_gamma)
    K_xy = rbf_kernel(real_sub, gen_sub, gamma=kernel_gamma)

    mmd_squared = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
    mmd_values.append(np.sqrt(max(0, mmd_squared)))

    return float(np.mean(mmd_values))


def summarize_statistics(ks_results: dict, kl_results: dict, mmd_value: float):
    """
    Print mean, max, and min of KS and KL statistics
    """
    print(
        f"KS Mean: {np.mean(
            list(ks_results.values())
            ):.4f}, Max: {np.max(
                list(ks_results.values())
                ):.4f}, Min: {np.min(
                    list(ks_results.values())
                    ):.4f}")
    print(f"KL Mean: {np.mean(
        list(kl_results.values())
        ):.4f}, Max: {np.max(
            list(kl_results.values())
            ):.4f}, Min: {np.min(
                list(kl_results.values())
                ):.4f}")
    print(f"MMD Score: {mmd_value:.4f}")
