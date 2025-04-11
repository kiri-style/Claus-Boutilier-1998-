import numpy as np
import matplotlib.pyplot as plt

# ----- Penalty Game Matrix -----
REWARD_MATRIX = np.array([
    [10, 0],  # A0
    [0, 2]    # A1
])
ACTIONS = [0, 1]

# ----- Parameters -----
EPISODES = 5000
ALPHA = 0.1
GAMMA = 0.9
TEMPERATURE = 1.0

# ----- Helper: Softmax sampling -----
def softmax(q_values, temperature):
    exp_q = np.exp(q_values / temperature)
    return exp_q / np.sum(exp_q)

# ----- Training function: Normal Boltzmann exploration -----
def train_normal_boltzmann(episodes=EPISODES, temperature=TEMPERATURE, alpha=ALPHA, gamma=GAMMA):
    q1 = np.zeros(2)  # Q-values for agent 1
    q2 = np.zeros(2)  # Q-values for agent 2
    rewards = []

    for _ in range(episodes):
        # Softmax action selection
        probs1 = softmax(q1, temperature)
        probs2 = softmax(q2, temperature)

        a1 = np.random.choice(ACTIONS, p=probs1)
        a2 = np.random.choice(ACTIONS, p=probs2)

        reward = REWARD_MATRIX[a1, a2]
        rewards.append(reward)

        # Q-learning updates
        q1[a1] += alpha * (reward + gamma * np.max(q1) - q1[a1])
        q2[a2] += alpha * (reward + gamma * np.max(q2) - q2[a2])

    return rewards

# ----- Helper: Sliding average reward -----
def moving_average(data, window=10):
    return np.convolve(data, np.ones(window) / window, mode='valid')

# ----- Run and plot -----
nb_rewards = train_normal_boltzmann()
nb_smoothed = moving_average(nb_rewards, window=10)

plt.plot(nb_smoothed, label="Normal Boltzmann (NB)")
plt.xlabel("Episode")
plt.ylabel("Average Reward (window=10)")
plt.title("Sliding Average Reward – Penalty Game")
plt.grid()
plt.legend()
plt.show()
