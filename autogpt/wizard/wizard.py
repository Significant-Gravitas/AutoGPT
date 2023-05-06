from typing import Any, Dict
from abc import ABC, abstractmethod

from questionary import prompt as questionary_prompt
#from PyInquirer import prompt as pyinquirer_prompt

import json

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

adapter = QuestionaryAdapter() # or PyInquirerAdapter()

filename = 'webAppGenerator.wizard'
with open(filename) as f:
    data = json.load(f)

version = data['version']
name = data['name']
description = data['description']
questions = data['questions']

print(f"filename: {filename}, version:{version}, name:{name}")

answers = adapter.prompt(questions)

print(answers)
