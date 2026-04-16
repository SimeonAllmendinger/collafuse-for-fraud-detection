import torch
import torch.nn.functional as F


def mse_loss(predicted_noise, true_noise):
    return F.mse_loss(predicted_noise, true_noise)


def normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))


def prior_loss(predicted_noise, mu_nf, sigma_nf, eps=1e-8):

    """
    Probability-based loss. Encourages predicted errors to match non-fraud noise prior.
    - predicted_noise: (batch_size, num_features)
    - mu_nf: (num_features,) mean of non-fraud noise
    - sigma_nf: (num_features,) std dev (diagonal covariance)
    """
    sigma_nf = sigma_nf + eps

    # z-score: (pred - mu) / sigma
    z_scores = (predicted_noise - mu_nf) / sigma_nf

    # Compute 2 * P(Z ≤ |z|) - 1 = 1 - 2 * P(Z ≥ |z|)
    abs_z = torch.abs(z_scores)
    probs = 2 * normal_cdf(abs_z) - 1
    loss = 1 - probs  # We minimize the complement

    return loss.mean()


def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Triplet loss to keep generated samples closer to real frauds and far from non-frauds.
    - anchor: generated fraud sample (batch_size, features)
    - positive: real fraud sample (batch_size, features)
    - negative: real non-fraud sample (batch_size, features)
    """
    d_pos = F.pairwise_distance(anchor, positive)
    d_neg = F.pairwise_distance(anchor, negative)
    return F.relu(d_pos - d_neg + margin).mean()


def fraud_diffuse_loss(predicted_noise, true_noise,
                       mu_nf, sigma_nf,
                       anchor=None, positive=None, negative=None,
                       w1=0.1, w2=0.0):
    """
    Final loss function combining all three components.
    """
    l_norm = mse_loss(predicted_noise, true_noise)
    l_prior = prior_loss(predicted_noise, mu_nf, sigma_nf)
    l_triplet = 0.0

    if anchor is not None and positive is not None and negative is not None:
        l_triplet = triplet_loss(anchor, positive, negative)

    return l_norm + w1 * l_prior + w2 * l_triplet, {
        "L_norm": l_norm.item(),
        "L_prior": l_prior.item(),
        "L_triplet": l_triplet.item() if isinstance(l_triplet, torch.Tensor) else l_triplet
    }
