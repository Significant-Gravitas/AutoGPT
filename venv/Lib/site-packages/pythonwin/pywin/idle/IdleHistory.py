class History:
    def __init__(self, text, output_sep="\n"):
        self.text = text
        self.history = []
        self.history_prefix = None
        self.history_pointer = None
        self.output_sep = output_sep
        text.bind("<<history-previous>>", self.history_prev)
        text.bind("<<history-next>>", self.history_next)

    def history_next(self, event):
        self.history_do(0)
        return "break"

    def history_prev(self, event):
        self.history_do(1)
        return "break"

    def _get_source(self, start, end):
        # Get source code from start index to end index.  Lines in the
        # text control may be separated by sys.ps2 .
        lines = self.text.get(start, end).split(self.output_sep)
        return "\n".join(lines)

    def _put_source(self, where, source):
        output = self.output_sep.join(source.split("\n"))
        self.text.insert(where, output)

    def history_do(self, reverse):
        nhist = len(self.history)
        pointer = self.history_pointer
        prefix = self.history_prefix
        if pointer is not None and prefix is not None:
            if (
                self.text.compare("insert", "!=", "end-1c")
                or self._get_source("iomark", "end-1c") != self.history[pointer]
            ):
                pointer = prefix = None
        if pointer is None or prefix is None:
            prefix = self._get_source("iomark", "end-1c")
            if reverse:
                pointer = nhist
            else:
                pointer = -1
        nprefix = len(prefix)
        while 1:
            if reverse:
                pointer = pointer - 1
            else:
                pointer = pointer + 1
            if pointer < 0 or pointer >= nhist:
                self.text.bell()
                if self._get_source("iomark", "end-1c") != prefix:
                    self.text.delete("iomark", "end-1c")
                    self._put_source("iomark", prefix)
                pointer = prefix = None
                break
            item = self.history[pointer]
            if item[:nprefix] == prefix and len(item) > nprefix:
                self.text.delete("iomark", "end-1c")
                self._put_source("iomark", item)
                break
        self.text.mark_set("insert", "end-1c")
        self.text.see("insert")
        self.text.tag_remove("sel", "1.0", "end")
        self.history_pointer = pointer
        self.history_prefix = prefix

    def history_store(self, source):
        source = source.strip()
        if len(source) > 2:
            # avoid duplicates
            try:
                self.history.remove(source)
            except ValueError:
                pass
            self.history.append(source)
        self.history_pointer = None
        self.history_prefix = None

    def recall(self, s):
        s = s.strip()
        self.text.tag_remove("sel", "1.0", "end")
        self.text.delete("iomark", "end-1c")
        self.text.mark_set("insert", "end-1c")
        self.text.insert("insert", s)
        self.text.see("insert")
