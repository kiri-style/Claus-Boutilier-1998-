import numpy as np
import matplotlib.pyplot as plt

# ----- Penalty Game Matrix (as in the paper) -----
REWARD_MATRIX = np.array([
    [-2, -10],  # A0
    [-10, 10]   # A1
])
ACTIONS = [0, 1]

# ----- Parameters -----
EPISODES = 60
ALPHA = 0.1
GAMMA = 0.9
INITIAL_TEMP = 16.0
DECAY_RATE = 0.995
WINDOW = 10  # sliding average window

# ----- Softmax helper -----
def softmax(q_values, temperature):
    q = q_values / temperature
    q -= np.max(q)
    exp_q = np.exp(q)
    return exp_q / np.sum(exp_q)

# ----- Moving average helper -----
def moving_average(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# ----- Train functions for each strategy -----

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
            combined1.append(beta * max_q + (1 - beta) * ev_q)
        probs1 = softmax(np.array(combined1), temp)
        q1_optimistic = [np.max(q1[a1]) for a1 in ACTIONS]
        probs1_for_agent2 = softmax(np.array(q1_optimistic), temp)
        combined2 = []
        for a2 in ACTIONS:
            max_q = np.max(q2[a2])
            ev_q = sum(q2[a2, a1] * probs1_for_agent2[a1] for a1 in ACTIONS)
            combined2.append(beta * max_q + (1 - beta) * ev_q)
        probs2_final = softmax(np.array(combined2), temp)
        a1 = np.random.choice(ACTIONS, p=probs1)
        a2 = np.random.choice(ACTIONS, p=probs2_final)
        reward = REWARD_MATRIX[a1, a2]
        rewards.append(reward)
        q1[a1, a2] += ALPHA * (reward + GAMMA * np.max(q1[a1]) - q1[a1, a2])
        q2[a2, a1] += ALPHA * (reward + GAMMA * np.max(q2[a2]) - q2[a2, a1])
    return rewards

# ----- Run base strategies -----
nb = train_normal_boltzmann()
ob = train_optimistic_boltzmann()
wob = train_weighted_ob()

# ----- Plot base strategies -----
plt.plot(moving_average(np.cumsum(nb), window=WINDOW), label="NB strategy")
plt.plot(moving_average(np.cumsum(ob), window=WINDOW), label="OB strategy")
plt.plot(moving_average(np.cumsum(wob), window=WINDOW), label="WOB strategy")

# ----- Run and plot combined strategies for multiple β -----
for beta in [0.25, 0.5, 0.75]:
    cb = train_combined_strategy(beta=beta)
    label = f"Combined strategy (β={beta})"
    plt.plot(moving_average(np.cumsum(cb), window=WINDOW), label=label)

# ----- Plot config -----
plt.axhline(0, color='black', linestyle='--')
plt.title("Sliding avg. reward – Penalty game (Figure 6 reproduction)")
plt.xlabel("Number of interactions")
plt.ylabel("Accumulated reward (moving avg)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("figure6_multi_beta.png", dpi=300)
plt.show()
