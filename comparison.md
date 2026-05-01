# GRPO vs GXPO: Detailed Comparison

## Overview

| Dimension | GRPO (DeepSeek-R1) | GXPO (Evolved) |
|-----------|-------------------|----------------|
| **Advantage Function** | $A_i = (r_i - \mu_r)/\sigma_r$ | $A_i = (r_i - \mu_r)/\sigma_r + \lambda \mathcal{E}(o_i, \mathcal{G})$ |
| **Exploration Mechanism** | No explicit exploration, relies on sampling stochasticity | Entropy bonus + variance-aware exploration |
| **Model Architecture** | Critic-free | Critic-free |
| **Computational Cost** | Low (single forward pass) | Slightly higher (entropy + variance computation) |
| **KL Regularization** | $\beta \cdot D_{KL}$ | $\beta \cdot D_{KL} + \gamma \cdot \mathcal{H}$ |
| **Sample Efficiency** | Moderate | Higher (more thorough exploration) |
| **Convergence Stability** | Good | Better (exploration-exploitation balance) |
| **Best For** | Tasks with clear reward functions | Sparse/complex long-reasoning tasks |

## Mathematical Comparison

### GRPO Objective

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q, \{o_i\}} \frac{1}{G} \sum_{i=1}^G \left[ \min(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref}) \right]$$

### GXPO Objective

$$\mathcal{J}_{GXPO}(\theta) = \mathbb{E}_{q, \{o_i\}} \frac{1}{G} \sum_{i=1}^G \left[ \min(\rho_i A_i^{GXPO}, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i^{GXPO}) - \beta \cdot D_{KL} \right] + \gamma \cdot \mathbb{E}_{o \sim \pi_\theta}[\mathcal{H}(\pi_\theta(\cdot|o))]$$

where $\rho_i = \pi_\theta(o_i|q) / \pi_{\theta_{old}}(o_i|q)$ is the importance sampling ratio.

## Key Insight

The evolution from GRPO to GXPO represents a paradigm shift from **"efficiently exploiting the known"** to **"actively exploring the unknown"**. GRPO proved that "group-level comparison" can replace the value function, while GXPO further demonstrates that introducing **exploration mechanisms within the group comparison framework** can significantly improve policy generalization on complex reasoning tasks.
