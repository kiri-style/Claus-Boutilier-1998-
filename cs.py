import numpy as np
import matplotlib.pyplot as plt

# ----- Penalty Game Matrix -----
REWARD_MATRIX = np.array([
    [-2, -10],  # A0
    [-10, 10]    # A1
])
ACTIONS = [0, 1]

# ----- Parameters -----
EPISODES = 70
ALPHA = 0.1
GAMMA = 0.9
INITIAL_TEMP = 16.0
DECAY_RATE = 0.995

# ----- Helper: Numerically stable softmax -----
def softmax(q_values, temperature):
    q = q_values / temperature
    q -= np.max(q)  # for numerical stability
    exp_q = np.exp(q)
    return exp_q / np.sum(exp_q)

# ----- Helper: Sliding average -----
def moving_average(data, window=5):
    return np.convolve(data, np.ones(window) / window, mode='valid')

# ----- Normal Boltzmann -----
def train_normal_boltzmann():
    q1 = np.zeros(2)
    q2 = np.zeros(2)
    rewards = []

    for t in range(EPISODES):
        temp = max(INITIAL_TEMP * (DECAY_RATE ** t), 0.1)
        probs1 = softmax(q1, temp)
        probs2 = softmax(q2, temp)

        a1 = np.random.choice(ACTIONS, p=probs1)
        a2 = np.random.choice(ACTIONS, p=probs2)

        reward = REWARD_MATRIX[a1, a2]
        rewards.append(reward)

        q1[a1] += ALPHA * (reward + GAMMA * np.max(q1) - q1[a1])
        q2[a2] += ALPHA * (reward + GAMMA * np.max(q2) - q2[a2])

    return rewards

# ----- Optimistic Boltzmann -----
def train_optimistic_boltzmann():
    q1 = np.zeros((2, 2))
    q2 = np.zeros((2, 2))
    rewards = []

    for t in range(EPISODES):
        temp = max(INITIAL_TEMP * (DECAY_RATE ** t), 0.1)

        q1_optimistic = [np.max(q1[a1]) for a1 in ACTIONS]
        q2_optimistic = [np.max(q2[a2]) for a2 in ACTIONS]

        probs1 = softmax(np.array(q1_optimistic), temp)
        probs2 = softmax(np.array(q2_optimistic), temp)

        a1 = np.random.choice(ACTIONS, p=probs1)
        a2 = np.random.choice(ACTIONS, p=probs2)

        reward = REWARD_MATRIX[a1, a2]
        rewards.append(reward)

        q1[a1, a2] += ALPHA * (reward + GAMMA * np.max(q1[a1]) - q1[a1, a2])
        q2[a2, a1] += ALPHA * (reward + GAMMA * np.max(q2[a2]) - q2[a2, a1])

    return rewards

# ----- Weighted Optimistic Boltzmann -----
def train_weighted_ob():
    q1 = np.zeros((2, 2))
    q2 = np.zeros((2, 2))
    rewards = []

    for t in range(EPISODES):
        temp = max(INITIAL_TEMP * (DECAY_RATE ** t), 0.1)

        q2_optimistic = [np.max(q2[a2]) for a2 in ACTIONS]
        probs2 = softmax(np.array(q2_optimistic), temp)

        ev1 = [sum(q1[a1, a2] * probs2[a2] for a2 in ACTIONS) for a1 in ACTIONS]
        probs1 = softmax(np.array(ev1), temp)

        q1_optimistic = [np.max(q1[a1]) for a1 in ACTIONS]
        probs1_for_agent2 = softmax(np.array(q1_optimistic), temp)

        ev2 = [sum(q2[a2, a1] * probs1_for_agent2[a1] for a1 in ACTIONS) for a2 in ACTIONS]
        probs2_final = softmax(np.array(ev2), temp)

        a1 = np.random.choice(ACTIONS, p=probs1)
        a2 = np.random.choice(ACTIONS, p=probs2_final)

        reward = REWARD_MATRIX[a1, a2]
        rewards.append(reward)

        q1[a1, a2] += ALPHA * (reward + GAMMA * np.max(q1[a1]) - q1[a1, a2])
        q2[a2, a1] += ALPHA * (reward + GAMMA * np.max(q2[a2]) - q2[a2, a1])

    return rewards

# ----- Combined Strategy (β-weighted OB + WOB) -----
def train_combined_strategy(beta=0.5):
    q1 = np.zeros((2, 2))
    q2 = np.zeros((2, 2))
    rewards = []

    for t in range(EPISODES):
        temp = max(INITIAL_TEMP * (DECAY_RATE ** t), 0.1)

        q2_optimistic = [np.max(q2[a2]) for a2 in ACTIONS]
        probs2 = softmax(np.array(q2_optimistic), temp)

        combined1 = []
        for a1 in ACTIONS:
            max_q = np.max(q1[a1])
            ev_q = sum(q1[a1, a2] * probs2[a2] for a2 in ACTIONS)
            combined_value = beta * max_q + (1 - beta) * ev_q
            combined1.append(combined_value)
        probs1 = softmax(np.array(combined1), temp)

        q1_optimistic = [np.max(q1[a1]) for a1 in ACTIONS]
        probs1_for_agent2 = softmax(np.array(q1_optimistic), temp)

        combined2 = []
        for a2 in ACTIONS:
            max_q = np.max(q2[a2])
            ev_q = sum(q2[a2, a1] * probs1_for_agent2[a1] for a1 in ACTIONS)
            combined_value = beta * max_q + (1 - beta) * ev_q
            combined2.append(combined_value)
        probs2_final = softmax(np.array(combined2), temp)

        a1 = np.random.choice(ACTIONS, p=probs1)
        a2 = np.random.choice(ACTIONS, p=probs2_final)

        reward = REWARD_MATRIX[a1, a2]
        rewards.append(reward)

        q1[a1, a2] += ALPHA * (reward + GAMMA * np.max(q1[a1]) - q1[a1, a2])
        q2[a2, a1] += ALPHA * (reward + GAMMA * np.max(q2[a2]) - q2[a2, a1])

    return rewards

# ----- Run and Plot All Strategies -----
nb_rewards = train_normal_boltzmann()
ob_rewards = train_optimistic_boltzmann()
wob_rewards = train_weighted_ob()
cb_rewards = train_combined_strategy(beta=0.5)

plt.plot(moving_average(nb_rewards), label="Normal Boltzmann (NB)")
plt.plot(moving_average(ob_rewards), label="Optimistic Boltzmann (OB)")
plt.plot(moving_average(wob_rewards), label="Weighted OB (WOB)")
plt.plot(moving_average(cb_rewards), label="Combined Strategy (β=0.5)")
plt.xlabel("Episode")
plt.ylabel("Average Reward (window=5)")
plt.title("Sliding Average Reward – Penalty Game")
plt.grid()
plt.legend()
plt.show()
