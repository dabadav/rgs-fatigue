import numpy as np
import scipy.optimize as opt
from pathlib import Path

def expected_reward(prev_rewards, prev_success, lambda_r=0.7, gamma_s=0.3):
    """
    Compute expected reward based on past rewards and success rate.

    prev_rewards: List of previous session scores.
    prev_success: List of binary success (1 = completed, 0 = not completed).
    lambda_r: Weight for past rewards.
    gamma_s: Weight for past success rate.
    """
    if len(prev_rewards) == 0:
        return 1  # Default small reward if no prior data

    avg_reward = np.mean(prev_rewards)
    success_rate = np.mean(prev_success)

    return lambda_r * avg_reward + gamma_s * success_rate

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

    # Generate synthetic data
    np.random.seed(42)
    num_trials = 100
    rewards = np.random.randint(1, 5, num_trials)
    efforts = np.random.randint(1, 5, num_trials)
    choices = np.random.binomial(1, 0.5, num_trials)

    # Load rgs data and fit the model
    fitted_params = fit_model(rewards, efforts, choices)

    # Save the fitted parameters
    np.save("fitted_params.npy", output_path / fitted_params)
