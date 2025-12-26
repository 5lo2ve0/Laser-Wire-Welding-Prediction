# Laser-Wire-Welding-Prediction
# Simulation-informed Bayesian Transfer Learning for Melt Pool Geometry Prediction

This repository provides a **representative implementation** of the core model architectures proposed in the paper:

**Few-shot Transfer Learning for Laser Welding Prediction**

The repository is intended to support **methodological transparency** and to clarify the availability of code resources referenced in the manuscript.
It is **not intended to serve as a complete reproduction package**.

---

## Repository Contents

The repository includes the following components:

- **models.py**
Definitions of the key neural network architectures used in the study, including:
- Bayesian neural network (BNN) layers with Gaussian weight parameterization
- Generator and critic architectures employed for WGAN-GP–based data augmentation

- **main.py**
A minimal runnable example demonstrating model instantiation and forward propagation.
This script is provided as a **functional reference**, rather than a full training or evaluation pipeline.

---

## Scope and Limitations

The released code focuses on the **core model structures and learning logic** described in the paper.
For clarity, the following components are **not included** in this repository:

- High-fidelity CFD simulation codes
- Complete training pipelines, hyperparameter tuning procedures, or evaluation scripts

These elements are intentionally omitted to avoid misinterpretation of the repository as a fully reproducible benchmark.

---

## Data and Code Availability

The manuscript presents the experimental data in tabulated form and provides a high-level pseudocode description of the overall network workflow.
Other data, detailed training configurations, and additional implementation details are available from the corresponding author upon reasonable request.

The present repository therefore serves as a **partial but faithful implementation** aligned with the scope of the published study.

---

## Usage Note

This code is intended for academic and research purposes only.
Users interested in reproducing or extending the full workflow are encouraged to contact the authors for further information.

---

## License

This project is released for non-commercial research use.
