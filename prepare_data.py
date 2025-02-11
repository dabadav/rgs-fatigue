import pandas as pd
import numpy as np
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

def compute_expected_reward(df, output_file = "rgs_data_fatigue.csv"):
    """
    The fatigue model is fitted using the following data:
    - Reward: Here as reward is not presented a priori to the patient before performing a task, we estimate it based on previous success rate and performance.
    - Effort: We consider effort the prescribed session duration to the patient
    - Choice: If the patient decides or not to finish the session
    """

    df = df.copy()
    df["ADHERENCE_BINARY"] = df["ADHERENCE"] == 1

    df = df.sort_values(by=["PATIENT_ID", "PROTOCOL_ID", "SESSION_ID"])
    df["EXPECTED_REWARD"] = np.nan

    for (patient_id, protocol_id), patient_protocol_data in df.groupby(["PATIENT_ID", "PROTOCOL_ID"]):
        prev_rewards = []  # Stores past scores for this protocol
        prev_success = []  # Stores past success for this protocol

        for idx in patient_protocol_data.index:
            # Compute expected reward using past data
            df.at[idx, "EXPECTED_REWARD"] = expected_reward(prev_rewards, prev_success)

            # Update history with the current session data
            prev_rewards.append(df.at[idx, "SCORE"])  # Save current session score
            prev_success.append(df.at[idx, "ADHERENCE_BINARY"])  # Save whether patient completed task

    # Fill any remaining NaNs with default reward
    df["EXPECTED_REWARD"].fillna(1, inplace=True)

    df.to_csv(output_file)
    print(f"Saving fatigue df to {output_file}...")

    return df

if __name__ == "__main__":
    RGS_MODE = "app"
    DATA_PATH = Path("../data")
    DATA_FILE = DATA_PATH / F"rgs_{RGS_MODE}.csv"

    data = pd.read_csv(DATA_FILE)
    data_fatigue = compute_expected_reward(data, DATA_PATH / f"rgs_{RGS_MODE}_fatigue.csv")
