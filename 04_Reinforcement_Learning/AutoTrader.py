from collections import deque

import tensorflow as tf
import numpy as np


class AutoTrader:

    def __init__(self, state_size=8, action_space=3, model_name="AutoTrader"):
        self.state_size = state_size
        self.action_space = action_space
        self.model_name = model_name
        self.memory = deque(maxlen=2000)

        self.model = self.build_nn()

        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

    def build_nn(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def trade(self, state):

        actions = self.model.predict(state)
        return np.argmax(actions[0])

    def train_batch(self, batch_size):
        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])
        for state, action, reward, next_state, done in batch:
            if not done:
                reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target = self.model.predict(state)
            target[0][action] = reward

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
