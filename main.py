# main.py
# Availability-oriented demo:
# - Instantiates models
# - Runs one forward pass for WGAN-GP and BNN
# - Computes representative loss terms (no training loop)

from __future__ import annotations

import torch

from models import (
    Generator,
    Critic,
    BayesianNeuralNetwork,
    get_mixed_noise,
    gradient_penalty,
)
from losses import (
    consistency_loss,
    feature_loss,
    center_loss,
)


def main() -> None:
    # -----------------------------
    # Device & seed (as in study)
    # -----------------------------
    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Dimensions (aligned to your original setting)
    # joint = 3 process params + 2 geometry = 5
    # -----------------------------
    latent_dim = 32
    hidden_dim = 64
    input_dim_gan = 3
    output_dim_gan = 2
    joint_dim = input_dim_gan + output_dim_gan

    # BNN dimensions used in your code
    bnn_in = input_dim_gan
    bnn_out = output_dim_gan
    bnn_h1, bnn_h2, bnn_h3 = 32, 64, 32

    # WGAN-GP hyperparams used in demo
    lambda_gp = 2.0
    w_cons, w_feat, w_center = 0.4, 0.7, 0.1

    # -----------------------------
    # Instantiate models
    # -----------------------------
    G = Generator(latent_dim, hidden_dim, joint_dim).to(device)
    D = Critic(joint_dim, hidden_dim).to(device)
    bnn = BayesianNeuralNetwork(bnn_in, bnn_h1, bnn_h2, bnn_h3, bnn_out).to(device)

    # -----------------------------
    # Create dummy "real" samples in joint (x,y) space
    # (no experimental data is shipped in this repo)
    # -----------------------------
    batch_size = 1  # consistent with extremely small-sample setting
    real_joint = torch.randn(batch_size, joint_dim, device=device)

    # -----------------------------
    # One-step WGAN-GP forward + losses
    # -----------------------------
    z = get_mixed_noise(batch_size, latent_dim, device=device, std=2.0, ratio=0.5)
    fake_joint = G(z)

    real_score, real_feat = D(real_joint, return_feature=True)
    fake_score, fake_feat = D(fake_joint.detach(), return_feature=True)

    gp = gradient_penalty(D, real_joint, fake_joint.detach(), lambda_gp=lambda_gp)

    # Critic loss (as in your code)
    d_loss = -torch.mean(real_score) + torch.mean(fake_score) + gp

    # Generator loss (as in your code)
    fake_score_g, fake_feat_g = D(fake_joint, return_feature=True)
    real_score_g, real_feat_g = D(real_joint, return_feature=True)

    l_cons = consistency_loss(fake_joint, real_joint)
    l_feat = feature_loss(fake_feat_g, real_feat_g)
    l_center = center_loss(fake_joint, real_joint)

    g_loss = -torch.mean(fake_score_g) + w_cons * l_cons + w_feat * l_feat + w_center * l_center

    # -----------------------------
    # One-step BNN forward + KL-like regularizer
    # -----------------------------
    x = torch.randn(batch_size, bnn_in, device=device)
    y_hat = bnn(x)
    kl_reg = bnn.kl_like()

    # -----------------------------
    # Print (sanity + transparency)
    # -----------------------------
    print("=== Demo run (availability-oriented) ===")
    print(f"Device: {device}")
    print(f"Generator output shape: {tuple(fake_joint.shape)}")
    print(f"Critic output shape: {tuple(real_score.shape)}")
    print(f"BNN output shape: {tuple(y_hat.shape)}")
    print("--- Loss terms (single forward pass, no training) ---")
    print(f"d_loss: {float(d_loss):.6f}  (includes GP)")
    print(f"g_loss: {float(g_loss):.6f}")
    print(f"  - consistency: {float(l_cons):.6f}")
    print(f"  - feature:     {float(l_feat):.6f}")
    print(f"  - center:      {float(l_center):.6f}")
    print(f"BNN KL-like regularizer: {float(kl_reg):.6f}")


if __name__ == "__main__":
    main()
