import numpy as np
import matplotlib.pyplot as plt

# ----- Penalty Game Matrix -----
REWARD_MATRIX = np.array([
    [10, 0],  # A0
    [0, 2]    # A1
])
ACTIONS = [0, 1]

# ----- Parameters -----
EPISODES = 50
ALPHA = 0.1
GAMMA = 0.9
TEMPERATURE = 16.0

# ----- Helper: Softmax sampling -----
def softmax(q_values, temperature):
    q = q_values / temperature
    q -= np.max(q)  # subtract max to stabilize (avoid overflow)
    exp_q = np.exp(q)
    return exp_q / np.sum(exp_q)

# ----- Training function: Normal Boltzmann exploration -----
def train_normal_boltzmann(episodes=5000, initial_temp=16.0, decay_rate=0.995, alpha=0.1, gamma=0.9):
    q1 = np.zeros(2)
    q2 = np.zeros(2)
    rewards = []

    for t in range(episodes):
        temp = max(initial_temp * (decay_rate ** t), 0.1)  # Ensure temp doesn’t go to 0

        # Boltzmann action selection with current temperature
        probs1 = softmax(q1, temp)
        probs2 = softmax(q2, temp)

        a1 = np.random.choice([0, 1], p=probs1)
        a2 = np.random.choice([0, 1], p=probs2)

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
