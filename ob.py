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
def moving_average(data, window=10):
    return np.convolve(data, np.ones(window) / window, mode='valid')

# ----- Normal Boltzmann Exploration -----
def train_normal_boltzmann(episodes=EPISODES, initial_temp=INITIAL_TEMP, decay_rate=DECAY_RATE, alpha=ALPHA, gamma=GAMMA):
    q1 = np.zeros(2)
    q2 = np.zeros(2)
    rewards = []

    for t in range(episodes):
        temp = max(initial_temp * (decay_rate ** t), 0.1)

        probs1 = softmax(q1, temp)
        probs2 = softmax(q2, temp)

        a1 = np.random.choice(ACTIONS, p=probs1)
        a2 = np.random.choice(ACTIONS, p=probs2)

        reward = REWARD_MATRIX[a1, a2]
        rewards.append(reward)

        q1[a1] += alpha * (reward + gamma * np.max(q1) - q1[a1])
        q2[a2] += alpha * (reward + gamma * np.max(q2) - q2[a2])

    return rewards

# ----- Optimistic Boltzmann Exploration -----
def train_optimistic_boltzmann(episodes=EPISODES, initial_temp=INITIAL_TEMP, decay_rate=DECAY_RATE, alpha=ALPHA, gamma=GAMMA):
    q1 = np.zeros((2, 2))  # Q1[a1, a2]
    q2 = np.zeros((2, 2))  # Q2[a2, a1]
    rewards = []

    for t in range(episodes):
        temp = max(initial_temp * (decay_rate ** t), 0.1)

        # Optimistic Q-values
        q1_optimistic = np.array([np.max(q1[a1]) for a1 in ACTIONS])
        q2_optimistic = np.array([np.max(q2[a2]) for a2 in ACTIONS])

        probs1 = softmax(q1_optimistic, temp)
        probs2 = softmax(q2_optimistic, temp)

        a1 = np.random.choice(ACTIONS, p=probs1)
        a2 = np.random.choice(ACTIONS, p=probs2)

        reward = REWARD_MATRIX[a1, a2]
        rewards.append(reward)

        q1[a1, a2] += alpha * (reward + gamma * np.max(q1[a1]) - q1[a1, a2])
        q2[a2, a1] += alpha * (reward + gamma * np.max(q2[a2]) - q2[a2, a1])

    return rewards

# ----- Run Training -----
nb_rewards = train_normal_boltzmann()
ob_rewards = train_optimistic_boltzmann()

nb_smoothed = moving_average(nb_rewards, window=5)
ob_smoothed = moving_average(ob_rewards, window=5)

# ----- Plotting -----
plt.plot(nb_smoothed, label="Normal Boltzmann (NB)")
plt.plot(ob_smoothed, label="Optimistic Boltzmann (OB)")
plt.xlabel("Episode")
plt.ylabel("Average Reward (window=10)")
plt.title("Sliding Average Reward – Penalty Game")
plt.grid()
plt.legend()
plt.show()
