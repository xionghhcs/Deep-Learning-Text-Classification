import tensorflow as tf
from abc import ABCMeta, abstractmethod, abstractclassmethod

class BasicModel(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractclassmethod
    def build_model(self):
        pass

    @abstractclassmethod
    def fit(self):
        pass

    @abstractclassmethod
    def predict(self):
        pass

    def save_weight(self, file_path):
        self.saver.save(self.sess, file_path)

    def load_weight(self, file_path):
        self.saver.restore(self.sess, file_path)
