from jinja2 import Template as JinjaTemplate
# from Cheetah.Template import Template as CheetahTemplate

class TemplateAdapter:
    def __init__(self, template_string, template_type='jinja2'):
        self.template_string = template_string
        self.template_type = template_type

    def render(self, context):
        if self.template_type == 'jinja2':
            template = JinjaTemplate(self.template_string)
            return template.render(context)
        elif self.template_type == 'cheetah':
            template = CheetahTemplate(source=self.template_string)
            return str(template(searchList=[context]))
        else:
            raise ValueError('Invalid template type: {}'.format(self.template_type))

template_string = 'Hello, {{ name }}!'
context = {'name': 'World'}

adapter = TemplateAdapter(template_string, template_type='jinja2')
output = adapter.render(context)
print(output)  # Output: Hello, World!

#adapter = TemplateAdapter(template_string, template_type='cheetah')
#output = adapter.render(context)
#print(output)  # Output: Hello, World!

