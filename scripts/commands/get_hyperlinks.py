from operator import itemgetter
import browse

from commands import Command

class Get_Hyperlinks(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        url = itemgetter('url')(kwargs)

        link_list = browse.scrape_links(url)
    
        return link_list