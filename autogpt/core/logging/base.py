import abc
class LogFormat(Enum):
  NONE       = 0
  MARKDOWN   = 1
  JSON       = 2
  TEXT       = 3
  YAML       = 4

@dataclasses.dataclass
class LogMessage:
  title: str
  message: str
  format: LogFormat

class LogLevels(Enum):
  NONE  = 0
  ERROR = 1
  WARN  = 2
  INFO  = 3
  DEBUG = 4
  SILLY = 5

class Logger(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *arg, **kwargs):
        pass

    @abc.abstractmethod
    def debug(self, message: LogMessage):
      pass

    @abc.abstractmethod
    def info(self, message: LogMessage):
      pass

    @abc.abstractmethod
    def warn(self, message: LogMessage):
      pass

    @abc.abstractmethod
    def error(self, message: LogMessage):
      pass

    @abc.abstractmethod
    def set_log_level(self, level: LogLevels):
      pass
