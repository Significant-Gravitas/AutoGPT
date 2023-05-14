import os
from pathlib import Path

from autogpt.workspace import WORKSPACE_PATH

TEMPLATES_PATH = WORKSPACE_PATH / "templates"

if not os.path.exists(TEMPLATES_PATH):
    os.makedirs(TEMPLATES_PATH)

class TemplateManager:
    _instance = None
    current_template = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.templates = {}
            cls._instance.current_template = None
            cls._instance.load_templates()
        return cls._instance

    def load_templates(self):
        for template_file in TEMPLATES_PATH.glob("*.md"):
            template_name = template_file.stem
            with open(template_file, "r", encoding="utf-8") as file:
                self.templates[template_name] = file.read()

    def list_templates(self) -> list[str]:
        return list(self.templates.keys())

    def read_template(self, template_name: str) -> str:
        if template_name in self.templates:
            return self.templates[template_name]
        else:
            return f"Template '{template_name}' not found."

    def create_template(self, template_name: str, content: str) -> str:
        if template_name in self.templates:
            return f"Template '{template_name}' already exists."
        template_path = TEMPLATES_PATH / f"{template_name}.md"
        with open(template_path, "w", encoding="utf-8") as file:
            file.write(content)
        self.templates[template_name] = content
        return f"Template '{template_name}' created at {template_path}"

    def set_current_template(self, template_name: str) -> str:
        if template_name in self.templates:
            self.current_template = self.templates[template_name]
            return f"Current template set to '{template_name}'."
        else:
            available_templates = ', '.join(self.list_templates())
            return f"Template '{template_name}' not found. Available templates: {available_templates}"

    def delete_template(self, template_name: str) -> str:
        if template_name in self.templates:
            template_path = TEMPLATES_PATH / f"{template_name}.md"
            os.remove(template_path)
            del self.templates[template_name]
            if self.current_template == template_name:
                self.current_template = None
            return f"Template '{template_name}' deleted."
        else:
            return f"Template '{template_name}' not found."

    def get_default_template(self) -> str:
        if self.current_template:
            return self.templates[self.current_template]
        else:
            return "No template is currently selected."

    def get_context_template(self, template_name=None):
        if template_name:
            template = self.read_template(template_name)
            if template:
                return template
            else:
                return f"Error: Template '{template_name}' not found."
        else:
            # If no template_name is provided, return the default context template
            with open(self.context_template_file, "r") as file:
                return file.read()
