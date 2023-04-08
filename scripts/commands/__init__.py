import os
import importlib

class Command(object):
	"""
	An interface for all Auto-GPT commands
	"""
	def __init__(self):
		self.name = self.__class__.__name__.lower()

	def execute(self, **kwargs):
		raise NotImplementedError()

class CommandManager():
	_registry = {}

	@staticmethod
	def register(command):
		if not issubclass(command.__class__, Command):
			raise ValueError('You may only register sub classes of Command')

		CommandManager._registry[command.name] = command

	@staticmethod
	def get(command_name):
		return CommandManager._registry[command_name]

"""
Load every module in the commands directory that subclasses
the `Command` interface and register it to the `CommandManager`
"""
for filename in os.listdir(os.path.dirname(__file__)):
	if filename.endswith('.py') and filename != '__init__.py':
		# get just the module name without the .py extension
		module_name = filename[:-3] 

		# load the module
		module = importlib.import_module(f'.{module_name}', package=__name__)

		for name in dir(module):
			clazz = getattr(module, name)

			if isinstance(clazz, type) and issubclass(clazz, Command) and clazz != Command:
				CommandManager.register(clazz())