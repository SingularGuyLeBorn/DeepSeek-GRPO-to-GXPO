"""
GRPO (Group Relative Policy Optimization)
===========================================
Core algorithm behind DeepSeek-R1.
Eliminates the need for a Critic network by using group-relative advantages.

Reference: DeepSeek-R1 (https://arxiv.org/abs/2501.12948)
"""

import torch
import torch.nn.functional as F


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    ref_log_probs: torch.Tensor = None,
    beta: float = 0.04,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """
    GRPO Loss implementation.

    Args:
        log_probs: Current policy log probabilities, shape [B, G, L]
        old_log_probs: Old policy log probabilities, shape [B, G, L]
        rewards: Rewards for each output, shape [B, G]
        ref_log_probs: Reference model log probs for KL regularization
        beta: KL penalty coefficient
        clip_eps: PPO clipping epsilon

    Returns:
        Scalar loss tensor
    """
    # Importance sampling ratio
    ratio = torch.exp(log_probs - old_log_probs)  # [B, G, L]

    # Group-relative advantage normalization
    mean_r = rewards.mean(dim=-1, keepdim=True)   # [B, 1]
    std_r = rewards.std(dim=-1, keepdim=True) + 1e-8
    advantages = (rewards - mean_r) / std_r       # [B, G]
    advantages = advantages.unsqueeze(-1)         # [B, G, 1]

    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    pg_loss = -torch.min(surr1, surr2).mean()

    # KL divergence regularization
    if ref_log_probs is not None:
        kl_div = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
        kl_loss = beta * kl_div.mean()
    else:
        # Approximate KL (when no reference model is available)
        log_ratio = log_probs - old_log_probs
        kl_loss = beta * (torch.exp(log_ratio) - log_ratio - 1).mean()

    return pg_loss + kl_loss


if __name__ == "__main__":
    # Example usage
    batch_size, group_size, seq_len = 4, 8, 128
    log_probs = torch.randn(batch_size, group_size, seq_len)
    old_log_probs = torch.randn(batch_size, group_size, seq_len)
    rewards = torch.randn(batch_size, group_size)

    loss = grpo_loss(log_probs, old_log_probs, rewards)
    print(f"GRPO Loss: {loss.item():.4f}")
