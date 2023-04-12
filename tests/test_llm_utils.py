import unittest

from config import Config
from dotenv import load_dotenv
import logging
from scripts.llm_utils import create_chat_completion

load_dotenv()

cfg = Config()


class TestLLMUtils(unittest.TestCase):

    def test_create_completion(self):
        messages = [{"role": "user", "content": "Give a brief overview of GPT Large Language Models"}, ]
        result = create_chat_completion(model=cfg.fast_llm_model, messages=messages)
        logging.info(result)
        assert len(result) > 200
