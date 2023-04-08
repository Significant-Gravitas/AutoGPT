from operator import itemgetter

import browse
from commands import Command

class Get_Text_Summary(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        url, question = itemgetter('url', 'question')(kwargs)

        text = browse.scrape_text(url)
        summary = browse.summarize_text(text, question)
    
        return """ "Result" : """ + summary