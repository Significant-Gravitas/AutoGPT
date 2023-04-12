import asyncio
import pprint
import unittest
import logging
from dotenv import load_dotenv
from scripts.search import google_official_search, get_text_summaries, get_text_summary

load_dotenv()


class TestSearch(unittest.TestCase):

    def test_google_official_search(self):
        links = google_official_search('What is the situation with Trump indictments?', 3)
        assert len(links) == 3

    def test_search_and_summarize(self):
        question = 'What is the situation with the Trump indictments?'
        links = google_official_search(question, 3)
        answer = asyncio.run(get_text_summary(links[1], question=question))

        assert "Result" in answer

    def test_search_and_summarize_multiple_sources(self):
        question = 'What is the situation with the Trump indictments?'
        links = google_official_search(question, num_results=5)
        answers = asyncio.run(get_text_summaries(links, question=question, num_sources=3))
        logging.info(pprint.pformat(answers))
        assert len(answers) == 3
