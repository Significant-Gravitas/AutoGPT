from typing import Any, Dict
from abc import ABC, abstractmethod


import argparse

from FileTemplating import TemplateAdapter

##
# TODO: should support multiple wizard engines here to be able to tinker around
from questionary import prompt as questionary_prompt
#from PyInquirer import prompt as pyinquirer_prompt

import json

###
# NOTE: This is just a proof-of-concept to illustrate the underlying idea
# Given the ongoing re-arch it's probably a good idea to wait with major coding efforts
# However, coding up something in standalone mode using a few stubs and helper classes should
# be useful in and of itself.
#

class PromptAdapter(ABC):
    @abstractmethod
    def prompt(self, questions: Dict[str, Any]) -> Dict[str, Any]:
        pass

class QuestionaryAdapter(PromptAdapter):
    def prompt(self, questions: Dict[str, Any]) -> Dict[str, Any]:
        return questionary_prompt(questions)

class PyInquirerAdapter(PromptAdapter):
    def prompt(self, questions: Dict[str, Any]) -> Dict[str, Any]:
        return pyinquirer_prompt(questions)

##
# TODO new adapter for Agents (inter-agent messaging)

ResponseAdapter = QuestionaryAdapter() # or PyInquirerAdapter()


###
# TODO: the JSON structure is likely to evolve as needed:
# - optional/required fields
# - input validation
# - execution of commands (top-level cmd mgr), e.g. for validating HTML/XML files generated procedurally
# - probably need to support some state machine stuff for each step ?

def main(args):
    # Your main code goes here
    #print(args)
    filename = args.filename # 'webAppGenerator.wizard'
    with open(filename) as f:
        data = json.load(f)

    ##
    # read out some meta data
    version = data['version']
    name = data['name']
    description = data['description']
    questions = data['questions']

    ##
    # show some meta info
    print(f"filename: {filename}, version:{version}, name:{name}")

    # and run the prompt loop
    answers = ResponseAdapter.prompt(questions)

    #context = {'version':'1.0.0','project': 'Demo','binary':'myDemo','sources':'main.cxx'}
    # Using a file-based Jinja2 template
    template_path = 'cmakelists.txt.template'
 
    SubstitutionAdapter = TemplateAdapter(template_path, template_type='jinja2')
 
    output = SubstitutionAdapter.render(answers)
    print(output)  



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WizardMode proof-of-concept for Auto-GPT')
    parser.add_argument('filename', type=str, nargs='?', help='Path to .wizard file', default='cmakelists.txt.wizard')
    args = parser.parse_args()

    main(args)


