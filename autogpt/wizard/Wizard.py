from typing import Any, Dict, List
from abc import ABC, abstractmethod


##
# TODO: should support multiple wizard engines here to be able to tinker around
from questionary import prompt as questionary_prompt
#from PyInquirer import prompt as pyinquirer_prompt

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



