
import os

from jinja2 import Environment, FileSystemLoader
# from Cheetah.Template import Template as CheetahTemplate

class TemplateAdapter:
    def __init__(self, template_path_or_string, template_type='jinja2'):
        self.template_type = template_type
        if os.path.isfile(template_path_or_string):
            self.template_path = template_path_or_string
            self.template_string = None
        else:
            self.template_path = None
            self.template_string = template_path_or_string

    def render(self, context):
        if self.template_type == 'jinja2':
            if self.template_path:
                env = Environment(loader=FileSystemLoader(os.path.dirname(self.template_path)))
                template = env.get_template(os.path.basename(self.template_path))
            else:
                template = Environment().from_string(self.template_string)
            return template.render(context)
        elif self.template_type == 'cheetah':
            if self.template_path:
                template = CheetahTemplate(file=self.template_path, searchList=[context])
            else:
                template = CheetahTemplate(source=self.template_string)
            return str(template)
        else:
            raise ValueError('Invalid template type: {}'.format(self.template_type))



