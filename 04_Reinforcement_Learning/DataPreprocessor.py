import pandas_datareader as pdr


class DataPreprocessor:
    def __init__(self, asset='BTC-EUR', step=1, interval_size=10):
        self.asset = asset
        self.close = self.load_data()
        self.step = step
        self.interval_size = interval_size

    def load_data(self):
        data = pdr.get_data_yahoo(self.asset)
        close = data['Close']
        return close

    @staticmethod
    def format_price(price):
        # To make the prices readable
        return "{0:2f}".format(price)
