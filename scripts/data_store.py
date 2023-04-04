import os
import shelve
import memory
import message_history

class DataStore():

    def persist_message_history(self, id):
      return True

    def load_message_history(self, id):
        return True

    def persist_memory(self ,id):
        return True

    def load_memory(self, id):
        return True

instance = DataStore()

class ShelfDataStore(DataStore):

    def __init__(self, path):
        self.path = path
        # Ensure path ends with a slash
        if self.path[-1] != "/":
            self.path += "/"
        self.message_history = []
        self.memory = []

    def persist_message_history(self, id):
        os.makedirs(f"{self.path}{id}", exist_ok=True)
        with shelve.open(f"{self.path}{id}/store") as f:
            f["message_history"] = message_history.message_history
        return True

    def load_message_history(self, id):
        try:
            with shelve.open(f"{self.path}{id}/store") as f:
                global message_history
                message_history.set_history(f["message_history"])
            return True
        except Exception as e:
            return e

    def persist_memory(self, id):
        os.makedirs(f"{self.path}{id}", exist_ok=True)
        with shelve.open(f"{self.path}/{id}/store") as f:
            f["memory"] = memory.permanent_memory
        return True

    def load_memory(self, id):
        try:
            with shelve.open(f"{self.path}{id}/store") as f:
                memory.permanent_memory = f["memory"]
            return True
        except Exception as e:
            return e
