from typing import Any, Dict, List
from abc import ABC, abstractmethod


import sys

from Wizard import PromptAdapter

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QDialogButtonBox

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

        # Add labels and line edits for each question
        self.answers = {}
        line_edits = create_line_edits(questions_list)
        for line_edit in line_edits:
            name = line_edit.objectName()
            message = line_edit.property('message')
            layout.addWidget(QLabel(message))
            layout.addWidget(line_edit)
            self.answers[name] = line_edit

        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_answers(self) -> Dict[str, Any]:
        # Return a dictionary of answers, keyed by question name
        return {name: line_edit.text() for name, line_edit in self.answers.items()}


def create_line_edits(questions_list: List[Dict[str, Any]]) -> List[QLineEdit]:
    line_edits = []
    for question in questions_list:
        widget = question.get('widget', 'text').lower()
        if widget == 'text':
            line_edit = QLineEdit()
            line_edit.setObjectName(question.get('name', ''))
            line_edit.setProperty('message', question.get('message', ''))
            line_edits.append(line_edit)

    return line_edits


class QtAdapter:
    def prompt(self, questions: Dict[str, Any]) -> Dict[str, Any]:
        app = QApplication(sys.argv)
        dialog = PromptDialog(questions)

        # Set the focus traversal order for the line edits
        line_edits = dialog.findChildren(QLineEdit)
        for i in range(len(line_edits) - 1):
            dialog.setTabOrder(line_edits[i], line_edits[i + 1])

        result = dialog.exec_()
        answers = dialog.get_answers()
        return answers if result == QDialog.Accepted else {}


def pyqt_prompt(questions: Dict[str, Any]) -> Dict[str, Any]:
    adapter = QtAdapter()
    return adapter.prompt(questions)


