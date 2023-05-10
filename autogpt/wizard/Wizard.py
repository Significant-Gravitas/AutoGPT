from typing import Any, Dict
from abc import ABC, abstractmethod


##
# TODO: should support multiple wizard engines here to be able to tinker around
from questionary import prompt as questionary_prompt
#from PyInquirer import prompt as pyinquirer_prompt

from typing import Dict, Any
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QDialogButtonBox


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

class PromptDialog(QDialog):
    def __init__(self, questions: Dict[str, Any]):
        super().__init__()

        # Determine the title of the dialog
        if isinstance(questions, dict):
            title = questions.get('name', 'Prompt Dialog')
            questions_list = questions.get('questions', [])
        else:
            title = 'Prompt Dialog'
            questions_list = questions

        self.setWindowTitle(title)

        # Create a layout for the dialog
        layout = QVBoxLayout()

        # Add labels and line edits for each question
        self.answers = {}
        for question in questions_list:
            label = QLabel(question.get('message', ''))
            layout.addWidget(label)
            line_edit = QLineEdit()
            self.answers[question.get('name', '')] = line_edit
            layout.addWidget(line_edit)

        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_answers(self) -> Dict[str, Any]:
        # Return a dictionary of answers, keyed by question name
        return {name: line_edit.text() for name, line_edit in self.answers.items()}


class QtAdapter(PromptAdapter):
    def prompt(self, questions: Dict[str, Any]) -> Dict[str, Any]:
        app = QApplication([])

        # Create a dialog to prompt the user for input
        dialog = PromptDialog(questions)

        # Display the dialog and wait for the user to close it
        if dialog.exec_() == QDialog.Accepted:
            # Return the user's answers
            return dialog.get_answers()
        else:
            # User canceled the dialog, so return an empty dictionary
            return {}
##
# TODO new adapter for Agents (inter-agent messaging)


ResponseAdapter = QtAdapter() # or PyInquirerAdapter()

