import os
import sys
# Probably a better way:
sys.path.append(os.path.abspath('../scripts'))
from memory.base import get_embedding

def MockConfig():
    return type('MockConfig', (object,), {
        'debug_mode': False,
        'continuous_mode': False,
        'speak_mode': False,
        'memory_embedder': 'sbert'
    })

class TestMemoryEmbedder(unittest.TestCase):
    def setUp(self): 
        self.cfg = MockConfig()

    def test_ada(self):
        self.cfg.memory_embedder = "ada"
        text = "Sample text"
        result = get_embedding(text)
        self.assertEqual(result.shape, (1536,))

    def test_sbert(self):
        self.cfg.memory_embedder = "sbert"
        text = "Sample text"
        result = get_embedding(text)
        self.assertEqual(result.shape, (768,))


if __name__ == '__main__':
    unittest.main()
 