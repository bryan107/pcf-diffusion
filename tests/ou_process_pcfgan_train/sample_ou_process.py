import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def sample_ou_process(
    N: int,
    L: int,
    D: int,
    theta: np.ndarray,
    mu_func: callable,
    sigma: np.ndarray,
    X0: np.ndarray = None,
) -> np.ndarray:
    """
    Samples a multidimensional Ornstein-Uhlenbeck process with a non-linear mu.

    Args:
        N (int): Number of sequences to sample.
        L (int): Length of each sequence (number of time steps).
        D (int): Dimensionality of each OU process.
        theta (np.ndarray): Rate of mean reversion of shape (D,).
        mu_func (callable): Function returning long-term mean (mu) as a vector of shape (D,).
        sigma (np.ndarray): Volatility of shape (D,).
        delta_t (float): Time step size (default: 1.0).
        X0 (np.ndarray): Initial state (optional, if None, defaults to `mu_func(0)`).

    Returns:
        np.ndarray: OU process of shape (N, L, D).
    """
    theta = np.asarray(theta).reshape(1, D)
    sigma = np.asarray(sigma).reshape(1, D)

    # Initialize X0 using the mu function at t=0 if not provided
    if X0 is None:
        X0 = mu_func(0)

    # Pre-allocate output array
    X = np.zeros((N, L, D))
    X[:, 0, :] = X0

    tt = np.linspace(0, 1, L)
    delta_t = tt[1] - tt[0]
    for i in range(1, L):
        noise = np.random.normal(0, 1, (N, D))
        mu_t = mu_func(tt[i]).reshape(1, D)  # Get time-dependent mu
        X[:, i, :] = (
            X[:, i - 1, :]
            + theta * (mu_t - X[:, i - 1, :]) * delta_t
            + sigma * math.sqrt(delta_t) * noise
        )

    return X


def plot_ou_process(ou_process: np.ndarray, mu_func: callable) -> None:
    """
    Plot sample paths from the OU process, the mean trend, and some sample paths with transparency.
    """
    N, L, D = ou_process.shape
    tt = np.linspace(0, 1, L)

    data = {
        "Time": np.tile(np.repeat(tt, D), N),
        "Value": ou_process.flatten(),
        "Dimension": np.tile(np.arange(D), L * N),
        "Sample": np.repeat(np.arange(N), L * D),
    }
    df = pd.DataFrame(data)

    # Plot sample paths with mean and confidence intervals
    sns.lineplot(
        data=df,
        x="Time",
        y="Value",
        hue="Dimension",
        ci="sd",
        estimator="mean",
        palette="tab10",
    )

    # Plot a few sample paths with transparency
    for sample_idx in range(min(50, N)):  # Plot up to 5 sample paths
        for d in range(D):
            plt.plot(
                tt,
                ou_process[sample_idx, :, d],
                color=f"C{d}",
                alpha=0.15,
            )

    # Plot the non-linear mean trend (mu)
    mu_values = np.array([mu_func(t) for t in tt])
    for d in range(D):
        plt.plot(
            tt,
            mu_values[:, d],
            color="red",
            lw=2,
            label=f"Non-linear Mean (Dim {d+1})",
            linestyle="--",
        )

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("OU Process: Sample Paths, Non-linear Mean, and Mean Trend")
    plt.legend()
    plt.show()


# Non-linear mu function example
def non_linear_mu(t: float) -> np.ndarray:
    return np.array([0.5 * np.sin(2.0 * t), 0.5 * np.cos(2.0 * t)])


np.random.seed(42)

# Example usage
N, L, D = 50, 30, 2
theta = np.array([25, 15])
sigma = np.array([0.2, 0.5])

ou_process = sample_ou_process(N, L, D, theta, non_linear_mu, sigma)
plot_ou_process(ou_process, non_linear_mu)
