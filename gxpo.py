"""
GXPO (Group-based eXploration Policy Optimization)
====================================================
Evolution of GRPO that introduces group-level exploration mechanisms.

Key innovation: Advantage = GRPO relative advantage + exploration bonus
"""

import torch
import torch.nn.functional as F


def gxpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    output_entropies: torch.Tensor = None,
    ref_log_probs: torch.Tensor = None,
    beta: float = 0.04,
    clip_eps: float = 0.2,
    lambda_explore: float = 0.1,
    gamma_entropy: float = 0.01,
) -> torch.Tensor:
    """
    GXPO Loss implementation.

    Builds on GRPO by adding:
    1. Exploration bonus based on output entropy
    2. Variance-aware exploration weighting
    3. Policy entropy regularization

    Args:
        log_probs: Current policy log probabilities, [B, G, L]
        old_log_probs: Old policy log probabilities, [B, G, L]
        rewards: Rewards for each output, [B, G]
        output_entropies: Entropy of each output, [B, G]
        ref_log_probs: Reference model log probs
        beta: KL penalty coefficient
        clip_eps: PPO clipping epsilon
        lambda_explore: Exploration bonus weight
        gamma_entropy: Policy entropy regularization coefficient

    Returns:
        Scalar loss tensor
    """
    # === GRPO base advantage ===
    mean_r = rewards.mean(dim=-1, keepdim=True)
    std_r = rewards.std(dim=-1, keepdim=True) + 1e-8
    grpo_adv = (rewards - mean_r) / std_r  # [B, G]

    # === GXPO exploration bonus ===
    if output_entropies is not None:
        # Variance-aware exploration: higher weight when reward variance is large
        group_var = rewards.var(dim=-1, keepdim=True).expand(-1, rewards.size(-1))
        explore_bonus = lambda_explore * (output_entropies + group_var)
    else:
        explore_bonus = 0.0

    # Combined advantage
    advantages = grpo_adv + explore_bonus  # [B, G]
    advantages = advantages.unsqueeze(-1)  # [B, G, 1]

    # Importance sampling ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    pg_loss = -torch.min(surr1, surr2).mean()

    # KL regularization
    if ref_log_probs is not None:
        kl_div = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
        kl_loss = beta * kl_div.mean()
    else:
        log_ratio = log_probs - old_log_probs
        kl_loss = beta * (torch.exp(log_ratio) - log_ratio - 1).mean()

    # GXPO extra: policy entropy regularization (encourages exploration)
    if output_entropies is not None:
        entropy_loss = -gamma_entropy * output_entropies.mean()
    else:
        entropy_loss = 0.0

    return pg_loss + kl_loss + entropy_loss


if __name__ == "__main__":
    # Example usage
    batch_size, group_size, seq_len = 4, 8, 128
    log_probs = torch.randn(batch_size, group_size, seq_len)
    old_log_probs = torch.randn(batch_size, group_size, seq_len)
    rewards = torch.randn(batch_size, group_size)
    entropies = torch.rand(batch_size, group_size) * 2.0

    loss = gxpo_loss(log_probs, old_log_probs, rewards, entropies)
    print(f"GXPO Loss: {loss.item():.4f}")
