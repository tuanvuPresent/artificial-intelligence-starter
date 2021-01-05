import random

import gym
import numpy as np


class TaxiReinforcement:
    def __init__(self):
        self.env = gym.make('Taxi-v3')
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

    def train(self, total_episodes, alpha=0.1, gamma=0.6, epsilon=0.1):
        for i in range(1, total_episodes):
            state = self.env.reset()
            done = False
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample()  # Explore action space
                else:
                    action = np.argmax(self.q_table[state])  # Exploit learned values
                next_state, reward, done, info = self.env.step(action)
                self.q_table[state, action] = (1 - alpha) * self.q_table[state, action] \
                                              + alpha * (reward + gamma * np.max(self.q_table[next_state]))
                state = next_state
        np.save('q_table', self.q_table)
        print("Training finished.\n")

    def test(self, episodes):
        total_epochs, total_penalties = 0, 0
        for _ in range(episodes):
            state = self.env.reset()
            epochs, penalties, reward = 0, 0, 0
            done = False
            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, done, info = self.env.step(action)
                print(self.env.render())
                if reward == -10:
                    penalties += 1
                epochs += 1
            total_penalties += penalties
            total_epochs += epochs
        print(f"Average timesteps per episode: {total_epochs / episodes}")
        print(f"Average penalties per episode: {total_penalties / episodes}")

    def load_model(self, file_name):
        self.q_table = np.load(file_name)


taxi = TaxiReinforcement()
# taxi.train(total_episodes=200000)
taxi.load_model('q_table.npy')
taxi.test(1)
