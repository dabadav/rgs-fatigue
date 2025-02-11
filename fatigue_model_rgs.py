import numpy as np
import pandas as pd
import scipy.optimize as opt
from pathlib import Path

def softmax(x, beta):
    """ Compute softmax probability for binary choices """
    return 1 / (1 + np.exp(-beta * x))

def subjective_value(reward, effort, fatigue, k):
    """ Compute subjective value of a work option """
    return reward - (fatigue * k * effort ** 2)

def negative_log_likelihood(params, efforts, choices, prev_rewards, prev_success):
    """
    Compute negative log-likelihood for given parameters.

    params: [k, beta, alpha, delta, theta] - model parameters
    rewards: array of rewards offered
    efforts: array of efforts required
    choices: array of binary choices (1 = work, 0 = rest)
    """
    k, beta, alpha, delta, theta = params  # Unpack parameters
    num_trials = len(efforts)

    # Initialize fatigue states
    RF = 0.5  # Recoverable Fatigue
    UF = 0.5  # Unrecoverable Fatigue

    nll = 0  # Negative log-likelihood

    for t in range(num_trials):

        reward = expected_reward(prev_rewards[:t], prev_success[:t])

        fatigue = RF + UF  # Total fatigue
        sv_work = subjective_value(reward, efforts[t], fatigue, k)
        sv_rest = 1  # Rest is always worth 1 credit

        # Compute probability of choosing work
        p_work = softmax(sv_work - sv_rest, beta)

        # Avoid log(0) issues
        p_work = np.clip(p_work, 1e-6, 1 - 1e-6)

        # Compute log-likelihood for observed choice
        if choices[t] == 1:  # If participant chose work
            nll -= np.log(p_work)
        else:  # If participant chose rest
            nll -= np.log(1 - p_work)

        # Update fatigue states
        if choices[t] == 1:
            RF += alpha * efforts[t]  # Increase recoverable fatigue
            UF += theta * efforts[t]  # Increase unrecoverable fatigue
        else:
            RF -= delta  # Recoverable fatigue decreases with rest
            RF = max(0, RF)  # RF cannot be negative

    return nll  # We minimize this value

def fit_model(rewards, efforts, choices, initial_guess=[0.1, 1, 0.1, 0.1, 0.1]):
    """ Fit the model to the data using maximum likelihood estimation """
    result = opt.minimize(
        negative_log_likelihood,
        initial_guess,
        args=(rewards, efforts, choices)
    )
    fitted_params = result.x

    print(f"Fitted Parameters: k={fitted_params[0]:.3f}, beta={fitted_params[1]:.3f}, "
      f"alpha={fitted_params[2]:.3f}, delta={fitted_params[3]:.3f}, theta={fitted_params[4]:.3f}")

    return fitted_params

if __name__ == "__main__":
    # Path: rgs-fatigue/fatigue_model.py
    data_path = Path("../data")
    output_path = Path("../results")
    rgs_mode = "app"
    data_file = data_path / f"rgs_data_{rgs_mode}_fatigue.csv"

    # Generate synthetic data
    df = pd.read_csv(data_file)

    rewards = df["EXPECTED_REWARD"]
    efforts = df["SESSION_PRESCRIBED"]
    choices = df["ADHERENCE_BINARY"]

    # Load rgs data and fit the model
    fitted_params = fit_model(rewards, efforts, choices)

    # Save the fitted parameters
    np.save("fitted_params.npy", output_path / fitted_params)
