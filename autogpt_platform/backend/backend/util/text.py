from jinja2 import BaseLoader
from jinja2.sandbox import SandboxedEnvironment


class TextFormatter:
    def __init__(self):
        # Create a sandboxed environment
        self.env = SandboxedEnvironment(loader=BaseLoader(), autoescape=True)

        # Clear any registered filters, tests, and globals to minimize attack surface
        self.env.filters.clear()
        self.env.tests.clear()
        self.env.globals.clear()

    def format_string(self, template_str: str, values=None, **kwargs) -> str:
        template = self.env.from_string(template_str)
        return template.render(values or {}, **kwargs)
