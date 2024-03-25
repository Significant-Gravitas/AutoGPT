class WizardManager:
    def __init__(self):
        self.wizards = []

    def list_wizards(self):
        for wizard in self.wizards:
            print(wizard.name)

    def start_wizard(self, wizard_name):
        for wizard in self.wizards:
            if wizard.name == wizard_name:
                wizard.execute()
                break

    def add_wizard(self, wizard):
        self.wizards.append(wizard)

