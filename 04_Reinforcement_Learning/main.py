from AutoTrader import AutoTrader
from DataPreprocessor import DataPreprocessor
from Trainer import Trainer

if __name__ == '__main__':
    trader = AutoTrader()
    preprocessor = DataPreprocessor()
    trainer = Trainer()

    trainer.train(preprocessor, trader)


