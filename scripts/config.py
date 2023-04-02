class Singleton(type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=Singleton):
    """
    Configuration class to store the state of bools for different scripts access.
    """

    def __init__(self):
        self.continuous_mode = False
        self.speak_mode = False

    def set_continuous_mode(self, value: bool):
        self.continuous_mode = value

    def set_speak_mode(self, value: bool):
        self.speak_mode = value
