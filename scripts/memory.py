class Memory:
    _instance = None
    permanent_memory = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def commit_memory(self, string) -> str:
        _text = f"""Committing memory with string "{string}" """
        self.permanent_memory.append(string)
        return _text

    def delete_memory(self, index) -> str:
        if index >= 0 and index < len(self.permanent_memory):
            _text = "Deleting memory with index " + str(index)
            del self.permanent_memory[index]
            print(_text)
            return _text
        else:
            print("Invalid index, cannot delete memory.")
            return None

    def overwrite_memory(self, index, string) -> str:
        if int(index) >= 0 and index < len(self.permanent_memory):
            _text = "Overwriting memory with index " + \
                str(index) + " and string " + string
            self.permanent_memory[index] = string
            print(_text)
            return _text
        else:
            print("Invalid index, cannot overwrite memory.")
            return None

    def get_memory(self, index) -> str:
        if int(index) >= 0 and index < len(self.permanent_memory):
            return self.permanent_memory[index]
        else:
            print("Invalid index, cannot get memory.")
            return None

    def get_all_memory(self) -> list:
        return self.permanent_memory

    def get_all_memory_as_string(self) -> str:
        return "\n".join(self.permanent_memory)


if __name__ == "__main__":
    mem = Memory()
    mem2 = Memory()
    print(mem is mem2 and "Memory is a singleton")