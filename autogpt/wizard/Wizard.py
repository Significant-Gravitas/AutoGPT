from typing import Any, Dict
from abc import ABC, abstractmethod

import sys

##
# TODO: should support multiple wizard engines here to be able to tinker around
from questionary import prompt as questionary_prompt
#from PyInquirer import prompt as pyinquirer_prompt

from PyQt5.QtCore import Qt
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

        if isinstance(questions, dict):
            title = questions.get('name', 'Auto-GPT Wizard')
            questions_list = questions.get('questions', [])
        else:
            title = 'Auto-GPT Wizard'
            questions_list = questions

        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowContextHelpButtonHint)
        self.setWindowTitle(title)

        # Create a layout for the dialog
        layout = QVBoxLayout()

        # Add labels, line edits, and tooltips for each question
        self.answers = {}
        self.tooltips = {}
        self.validators = {}
        for question in questions_list:
            label = QLabel(question.get('message', ''))
            layout.addWidget(label)
            line_edit = QLineEdit()
            self.answers[question.get('name', '')] = line_edit
            layout.addWidget(line_edit)

            tooltip = question.get('tooltip', None)
            if tooltip:
                self.tooltips[question.get('name', '')] = tooltip

            validation = question.get('validation', None)
            if validation:
                regex = validation.get('regex', '')
                error_message = validation.get('error_message', '')
                self.validators[question.get('name', '')] = (regex, error_message)

        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def validate_and_accept(self):
        # Check for any validation errors before accepting the dialog
        for name, (regex, error_message) in self.validators.items():
            line_edit = self.answers[name]
            text = line_edit.text()
            if not re.match(regex, text):
                QMessageBox.warning(self, 'Validation Error', error_message)
                return

        # If there are no validation errors, accept the dialog
        self.accept()

    def get_answers(self) -> Dict[str, Any]:
        # Return a dictionary of answers, keyed by question name
        return {name: line_edit.text() for name, line_edit in self.answers.items()}

    def showEvent(self, event):
        # Add tooltips to line edits after the dialog is shown
        super().showEvent(event)
        for name, tooltip in self.tooltips.items():
            line_edit = self.answers[name]
            line_edit.setToolTip(tooltip)


class QtAdapter:
    def prompt(self, questions: Dict[str, Any]) -> Dict[str, Any]:
        app = QApplication(sys.argv)
        dialog = PromptDialog(questions)
        result = dialog.exec_()
        answers = dialog.get_answers()
        return answers if result == QDialog.Accepted else {}

##
# TODO new adapter for Agents (inter-agent messaging)


ResponseAdapter = QtAdapter() # or PyInquirerAdapter()

