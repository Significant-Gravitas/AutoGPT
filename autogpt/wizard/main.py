import click

from QtWizard import QtAdapter 
from FileTemplating import TemplateAdapter

import argparse
import json

###
# TODO: the JSON structure is likely to evolve as needed:
# - optional/required fields
# - input validation
# - execution of commands (top-level cmd mgr), e.g. for validating HTML/XML files generated procedurally
# - probably need to support some state machine stuff for each step ?

@click.command()
@click.argument('filename',default='cmakelists.txt')
# @click.option('--filename', prompt='wizard file', help='The wizard to execute')
def main(filename):
    # Your main code goes here
    with open(filename+'.wizard') as f:
        data = json.load(f)

    ##
    # read out some meta data
    version = data['version']
    name = data['name']
    description = data['description']
    questions = data['questions']

    ##
    # show some meta info
    info = click.style(f"filename: {filename}, version:{version}, name:{name}", fg='green', bg='black', bold=True)
    click.echo(info)


    ResponseAdapter = QtAdapter() # or PyInquirerAdapter()

    # and run the prompt loop
    answers = ResponseAdapter.prompt(questions)

    #context = {'version':'1.0.0','project': 'Demo','binary':'myDemo','sources':'main.cxx'}
    # Using a file-based Jinja2 template
    template_path = filename+'.template'
 
    SubstitutionAdapter = TemplateAdapter(template_path, template_type='jinja2')
 
    output = SubstitutionAdapter.render(answers)
    print(output)  

if __name__ == '__main__':
    main()


