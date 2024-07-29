import tkinter as tk
from tkinter import scrolledtext
from autogpt import Agent

class AutoGPTGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto-GPT Interface")

        self.label = tk.Label(root, text="Nome do Agente:")
        self.label.pack(pady=10)

        self.entry = tk.Entry(root, width=30)
        self.entry.pack(pady=10)

        self.run_button = tk.Button(root, text="Executar Agente", command=self.run_agent)
        self.run_button.pack(pady=10)

        self.output_text = scrolledtext.ScrolledText(root, width=40, height=10)
        self.output_text.pack(pady=10)

    def run_agent(self):
        agent_name = self.entry.get()
        self.output_text.insert(tk.END, f"Iniciando o agente: {agent_name}\n")
        agent = Agent(name=agent_name)
        agent.run()
        self.output_text.insert(tk.END, f"Agente {agent_name} terminou a execução.\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoGPTGUI(root)
    root.mainloop()
