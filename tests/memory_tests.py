import unittest
from unittest import mock
import sys
import os

sys.path.append(os.path.abspath('./scripts'))

from factory import MemoryFactory
from providers.weaviate import WeaviateMemory
from providers.pinecone import PineconeMemory

class TestMemoryFactory(unittest.TestCase):

	def test_invalid_memory_provider(self):
		
		with self.assertRaises(ValueError):
			memory = MemoryFactory.get_memory('Thanos')

	def test_create_pinecone_provider(self):

		# mock the init function of the provider to bypass
		# connection to the external pinecone service
		def __init__(self):
			pass

		with mock.patch.object(PineconeMemory, '__init__', __init__):
			memory = MemoryFactory.get_memory('pinecone')
			self.assertIsInstance(memory, PineconeMemory)

	def test_create_weaviate_provider(self):

		# mock the init function of the provider to bypass
		# connection to the external weaviate service
		def __init__(self):
			pass

		with mock.patch.object(WeaviateMemory, '__init__', __init__):
			memory = MemoryFactory.get_memory('weaviate')
			self.assertIsInstance(memory, WeaviateMemory)

	def test_provider_is_singleton(self):

		def __init__(self):
			pass

		with mock.patch.object(WeaviateMemory, '__init__', __init__):
			instance = MemoryFactory.get_memory('weaviate')
			other_instance = MemoryFactory.get_memory('weaviate')

			self.assertIs(instance, other_instance)


if __name__ == '__main__':
    unittest.main()

