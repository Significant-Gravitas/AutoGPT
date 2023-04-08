from operator import itemgetter
from commands import Command, CommandManager

class Browse_Website(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        url, question = itemgetter('url', 'question')(kwargs)

        get_text_summary = CommandManager.get('get_text_summary')
        get_hyperlinks = CommandManager.get('get_hyperlinks')

        summary = get_text_summary.execute(url=url, question=question)
        links = get_hyperlinks.execute(url=url)

        # Limit links to 5
        if len(links) > 5:
            links = links[:5]

        result = f"""Website Content Summary: {summary}\n\nLinks: {links}"""

        return result