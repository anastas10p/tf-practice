import numpy as np
from tensorflow import sigmoid
from tqdm import tqdm


class Trainer:
    def __init__(self, interval_size=10, episodes=1000, batch_size=32):
        self.interval_size = interval_size
        self.episodes = episodes
        self.batch_size = batch_size

    def create_state(self, step, interval_size, close):
        start_id = step - interval_size + 1
        if start_id >= 0:
            interval_data = close[start_id:step + 1]
        else:
            interval_data = - start_id * [close[0]] + list(close[0:step + 1])

        state = []
        for i in range(self.interval_size - 2):
            state.append(sigmoid(interval_data[i + 1] - interval_data[i]))
        state = (np.array(state)).reshape(-1, 8)
        return state

    def train(self, preprocessor, trader):

        data_samples = len(preprocessor.close) - 1

        for episode in range(1, self.episodes + 1):

            print("Episode: {}/{}".format(episode, self.episodes))

            state = self.create_state(0, self.interval_size + 1, preprocessor.close)

            total_profit = 0
            trader.inventory = []

            for t in tqdm(range(data_samples)):

                action = trader.trade(state)

                next_state = self.create_state(t + 1, self.interval_size + 1, preprocessor.close)
                reward = 0

                if action == 1:  # Buy
                    trader.inventory.append(preprocessor.close[t])
                    print("AI Trader bought: ", preprocessor.format_price(preprocessor.close[t]))

                elif action == 2 and len(trader.inventory) > 0:  # Sell
                    buy_price = trader.inventory.pop(0)

                    reward = max(preprocessor.close[t] - buy_price, 0)
                    total_profit += preprocessor.close[t] - buy_price
                    print("AI Trader sold: ", preprocessor.format_price(preprocessor.close[t]),
                          " Profit: " + preprocessor.format_price(preprocessor.close[t] - buy_price))

                if t == data_samples - 1:
                    done = True
                else:
                    done = False

                trader.memory.append((state, action, reward, next_state, done))

                state = next_state

                if done:
                    print("########################")
                    print("TOTAL PROFIT: {}".format(total_profit))
                    print("########################")

                if len(trader.memory) > self.batch_size:
                    trader.train_batch(self.batch_size)

            if episode % 10 == 0:
                trader.model.save("ai_trader_{}.h5".format(episode))
