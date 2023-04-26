"""Base class for memory providers."""
import abc

from autogpt.config import Config
from autogpt.singleton import AbstractSingleton

cfg = Config()


def save_vec_num(vec_num):
    with open("vec_num.txt", "w") as file:
        file.write(str(vec_num))


def load_vec_num():
    try:
        with open("vec_num.txt", "r") as file:
            vec_num = int(file.read())
            return vec_num
    except FileNotFoundError:
        return 0


class MemoryProviderSingleton(AbstractSingleton):
    @abc.abstractmethod
    def add(self, data):
        pass

    @abc.abstractmethod
    def get(self, data):
        pass

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def get_relevant(self, data, num_relevant=5):
        pass

    @abc.abstractmethod
    def get_stats(self):
        pass
