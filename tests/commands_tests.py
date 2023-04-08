import unittest
import os
import importlib
import sys
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath('./scripts'))

from commands import CommandManager, Command

class Mock_Command(Command):
	def __init__(self):
		super().__init__()

	def execute(self, **kwargs):
		pass

class TestCommandManager(unittest.TestCase):

	def test_invalid_registration(self):

		mock_command = MagicMock()

		with self.assertRaises(ValueError):
			CommandManager.register(mock_command)

	def test_valid_registration(self):

		CommandManager.register(Mock_Command())

		command = CommandManager.get('mock_command')

		self.assertIsNotNone(command)
		self.assertIsInstance(command, Command)


	def test_commands_loaded(self):
		# test the ALL the commands in the command directory are loaded
		for filename in os.listdir(os.path.abspath('./scripts/commands')):
			if filename.endswith('.py') and filename != '__init__.py':
				# get just the module name without the .py extension
				module_name = filename[:-3] 
				# load the module
				module = importlib.import_module(f'.{module_name}', package='commands')

				for name in dir(module):
					clazz = getattr(module, name)

					if isinstance(clazz, type) and issubclass(clazz, Command) and clazz != Command:
						command = CommandManager.get(clazz.__name__.lower())

						self.assertIsNotNone(command)
						self.assertIsInstance(command, clazz)


if __name__ == '__main__':
    unittest.main()