import numpy as np
import matplotlib.pyplot as plt

# ----- Environment: Coordination Game -----
REWARD_MATRIX = np.array([
    [8, 0],
    [0, 8]
])

ACTIONS = [0, 1]  # Action 0 and 1
episodes = 5000
epsilon = 0.1
alpha = 0.1
gamma = 0.9

# ----- Function: Independent Q-Learning -----
def train_iql():
    q1 = np.zeros(2)
    q2 = np.zeros(2)
    coordination_history = []

    for _ in range(episodes):
        # ε-greedy action selection
        a1 = np.random.choice(ACTIONS) if np.random.rand() < epsilon else np.argmax(q1)
        a2 = np.random.choice(ACTIONS) if np.random.rand() < epsilon else np.argmax(q2)

        reward = REWARD_MATRIX[a1, a2]

        # Q-learning updates
        q1[a1] = q1[a1] + alpha * (reward + gamma * np.max(q1) - q1[a1])
        q2[a2] = q2[a2] + alpha * (reward + gamma * np.max(q2) - q2[a2])

        coordination_history.append(1 if a1 == a2 else 0)

    return coordination_history

# ----- Function: Joint Action Learners -----
def train_jal():
    q1 = np.zeros((2, 2))  # Q[a1, a2]
    q2 = np.zeros((2, 2))  # Q[a2, a1]
    coordination_history = []

    for _ in range(episodes):
        # ε-greedy selection based on joint Q
        probs1 = np.full(2, epsilon / 2)
        probs2 = np.full(2, epsilon / 2)

        best_a1 = np.argmax([np.max(q1[a1]) for a1 in ACTIONS])
        best_a2 = np.argmax([np.max(q2[a2]) for a2 in ACTIONS])
        probs1[best_a1] += 1 - epsilon
        probs2[best_a2] += 1 - epsilon

        a1 = np.random.choice(ACTIONS, p=probs1)
        a2 = np.random.choice(ACTIONS, p=probs2)

        reward = REWARD_MATRIX[a1, a2]

        # Q-learning update using full joint action
        q1[a1, a2] += alpha * (reward + gamma * np.max(q1[a1]) - q1[a1, a2])
        q2[a2, a1] += alpha * (reward + gamma * np.max(q2[a2]) - q2[a2, a1])

        coordination_history.append(1 if a1 == a2 else 0)

    return coordination_history

# ----- Train both and plot -----
iql_history = train_iql()
jal_history = train_jal()

# Smooth data
def smooth(data, window=100):
    return [np.mean(data[i:i+window]) for i in range(len(data) - window)]

plt.plot(smooth(iql_history), label="Independent Q-Learning")
plt.plot(smooth(jal_history), label="Joint Action Learner")
plt.xlabel("Episode")
plt.ylabel("Coordination Rate (Moving Avg.)")
plt.title("Coordination Comparison: IQL vs JAL")
plt.legend()
plt.grid()
plt.show()
