# config.py - deals with loading configuration information.

# Loads config data from a .cfg file.  Also caches the compiled
# data back into a .cfc file.

# If you are wondering how to avoid needing .cfg files (eg,
# if you are freezing Pythonwin etc) I suggest you create a
# .py file, and put the config info in a docstring.  Then
# pass a CStringIO file (rather than a filename) to the
# config manager.
import glob
import importlib.util
import marshal
import os
import stat
import sys
import traceback
import types

import pywin
import win32api

from . import keycodes

debugging = 0
if debugging:
    import win32traceutil  # Some trace statements fire before the interactive window is open.

    def trace(*args):
        sys.stderr.write(" ".join(map(str, args)) + "\n")

else:
    trace = lambda *args: None

compiled_config_version = 3


def split_line(line, lineno):
    comment_pos = line.find("#")
    if comment_pos >= 0:
        line = line[:comment_pos]
    sep_pos = line.rfind("=")
    if sep_pos == -1:
        if line.strip():
            print("Warning: Line %d: %s is an invalid entry" % (lineno, repr(line)))
            return None, None
        return "", ""
    return line[:sep_pos].strip(), line[sep_pos + 1 :].strip()


def get_section_header(line):
    # Returns the section if the line is a section header, else None
    if line[0] == "[":
        end = line.find("]")
        if end == -1:
            end = len(line)
        rc = line[1:end].lower()
        try:
            i = rc.index(":")
            return rc[:i], rc[i + 1 :]
        except ValueError:
            return rc, ""
    return None, None


def find_config_file(f):
    return os.path.join(pywin.__path__[0], f + ".cfg")


def find_config_files():
    return [
        os.path.split(x)[1]
        for x in [
            os.path.splitext(x)[0]
            for x in glob.glob(os.path.join(pywin.__path__[0], "*.cfg"))
        ]
    ]


class ConfigManager:
    def __init__(self, f):
        self.filename = "unknown"
        self.last_error = None
        self.key_to_events = {}
        b_close = False
        if hasattr(f, "readline"):
            fp = f
            self.filename = "<config string>"
            compiled_name = None
        else:
            try:
                f = find_config_file(f)
                src_stat = os.stat(f)
            except os.error:
                self.report_error("Config file '%s' not found" % f)
                return
            self.filename = f
            self.basename = os.path.basename(f)
            trace("Loading configuration", self.basename)
            compiled_name = os.path.splitext(f)[0] + ".cfc"
            try:
                cf = open(compiled_name, "rb")
                try:
                    ver = marshal.load(cf)
                    ok = compiled_config_version == ver
                    if ok:
                        kblayoutname = marshal.load(cf)
                        magic = marshal.load(cf)
                        size = marshal.load(cf)
                        mtime = marshal.load(cf)
                        if (
                            magic == importlib.util.MAGIC_NUMBER
                            and win32api.GetKeyboardLayoutName() == kblayoutname
                            and src_stat[stat.ST_MTIME] == mtime
                            and src_stat[stat.ST_SIZE] == size
                        ):
                            self.cache = marshal.load(cf)
                            trace("Configuration loaded cached", compiled_name)
                            return  # We are ready to roll!
                finally:
                    cf.close()
            except (os.error, IOError, EOFError):
                pass
            fp = open(f)
            b_close = True
        self.cache = {}
        lineno = 1
        line = fp.readline()
        while line:
            # Skip to the next section (maybe already there!)
            section, subsection = get_section_header(line)
            while line and section is None:
                line = fp.readline()
                if not line:
                    break
                lineno = lineno + 1
                section, subsection = get_section_header(line)
            if not line:
                break

            if section == "keys":
                line, lineno = self._load_keys(subsection, fp, lineno)
            elif section == "extensions":
                line, lineno = self._load_extensions(subsection, fp, lineno)
            elif section == "idle extensions":
                line, lineno = self._load_idle_extensions(subsection, fp, lineno)
            elif section == "general":
                line, lineno = self._load_general(subsection, fp, lineno)
            else:
                self.report_error(
                    "Unrecognised section header '%s:%s'" % (section, subsection)
                )
                line = fp.readline()
                lineno = lineno + 1
        if b_close:
            fp.close()
        # Check critical data.
        if not self.cache.get("keys"):
            self.report_error("No keyboard definitions were loaded")
        if not self.last_error and compiled_name:
            try:
                cf = open(compiled_name, "wb")
                marshal.dump(compiled_config_version, cf)
                marshal.dump(win32api.GetKeyboardLayoutName(), cf)
                marshal.dump(importlib.util.MAGIC_NUMBER, cf)
                marshal.dump(src_stat[stat.ST_SIZE], cf)
                marshal.dump(src_stat[stat.ST_MTIME], cf)
                marshal.dump(self.cache, cf)
                cf.close()
            except (IOError, EOFError):
                pass  # Ignore errors - may be read only.

    def configure(self, editor, subsections=None):
        # Execute the extension code, and find any events.
        # First, we "recursively" connect any we are based on.
        if subsections is None:
            subsections = []
        subsections = [""] + subsections
        general = self.get_data("general")
        if general:
            parents = general.get("based on", [])
            for parent in parents:
                trace("Configuration based on", parent, "- loading.")
                parent = self.__class__(parent)
                parent.configure(editor, subsections)
                if parent.last_error:
                    self.report_error(parent.last_error)

        bindings = editor.bindings
        codeob = self.get_data("extension code")
        if codeob is not None:
            ns = {}
            try:
                exec(codeob, ns)
            except:
                traceback.print_exc()
                self.report_error("Executing extension code failed")
                ns = None
            if ns:
                num = 0
                for name, func in list(ns.items()):
                    if type(func) == types.FunctionType and name[:1] != "_":
                        bindings.bind(name, func)
                        num = num + 1
                trace("Configuration Extension code loaded", num, "events")
        # Load the idle extensions
        for subsection in subsections:
            for ext in self.get_data("idle extensions", {}).get(subsection, []):
                try:
                    editor.idle.IDLEExtension(ext)
                    trace("Loaded IDLE extension", ext)
                except:
                    self.report_error("Can not load the IDLE extension '%s'" % ext)

        # Now bind up the key-map (remembering a reverse map
        subsection_keymap = self.get_data("keys")
        num_bound = 0
        for subsection in subsections:
            keymap = subsection_keymap.get(subsection, {})
            bindings.update_keymap(keymap)
            num_bound = num_bound + len(keymap)
        trace("Configuration bound", num_bound, "keys")

    def get_key_binding(self, event, subsections=None):
        if subsections is None:
            subsections = []
        subsections = [""] + subsections

        subsection_keymap = self.get_data("keys")
        for subsection in subsections:
            map = self.key_to_events.get(subsection)
            if map is None:  # Build it
                map = {}
                keymap = subsection_keymap.get(subsection, {})
                for key_info, map_event in list(keymap.items()):
                    map[map_event] = key_info
                self.key_to_events[subsection] = map

            info = map.get(event)
            if info is not None:
                return keycodes.make_key_name(info[0], info[1])
        return None

    def report_error(self, msg):
        self.last_error = msg
        print("Error in %s: %s" % (self.filename, msg))

    def report_warning(self, msg):
        print("Warning in %s: %s" % (self.filename, msg))

    def _readline(self, fp, lineno, bStripComments=1):
        line = fp.readline()
        lineno = lineno + 1
        if line:
            bBreak = (
                get_section_header(line)[0] is not None
            )  # A new section is starting
            if bStripComments and not bBreak:
                pos = line.find("#")
                if pos >= 0:
                    line = line[:pos] + "\n"
        else:
            bBreak = 1
        return line, lineno, bBreak

    def get_data(self, name, default=None):
        return self.cache.get(name, default)

    def _save_data(self, name, data):
        self.cache[name] = data
        return data

    def _load_general(self, sub_section, fp, lineno):
        map = {}
        while 1:
            line, lineno, bBreak = self._readline(fp, lineno)
            if bBreak:
                break

            key, val = split_line(line, lineno)
            if not key:
                continue
            key = key.lower()
            l = map.get(key, [])
            l.append(val)
            map[key] = l
        self._save_data("general", map)
        return line, lineno

    def _load_keys(self, sub_section, fp, lineno):
        # Builds a nested dictionary of
        # (scancode, flags) = event_name
        main_map = self.get_data("keys", {})
        map = main_map.get(sub_section, {})
        while 1:
            line, lineno, bBreak = self._readline(fp, lineno)
            if bBreak:
                break

            key, event = split_line(line, lineno)
            if not event:
                continue
            sc, flag = keycodes.parse_key_name(key)
            if sc is None:
                self.report_warning("Line %d: Invalid key name '%s'" % (lineno, key))
            else:
                map[sc, flag] = event
        main_map[sub_section] = map
        self._save_data("keys", main_map)
        return line, lineno

    def _load_extensions(self, sub_section, fp, lineno):
        start_lineno = lineno
        lines = []
        while 1:
            line, lineno, bBreak = self._readline(fp, lineno, 0)
            if bBreak:
                break
            lines.append(line)
        try:
            c = compile(
                "\n" * start_lineno + "".join(lines),  # produces correct tracebacks
                self.filename,
                "exec",
            )
            self._save_data("extension code", c)
        except SyntaxError as details:
            errlineno = details.lineno + start_lineno
            # Should handle syntax errors better here, and offset the lineno.
            self.report_error(
                "Compiling extension code failed:\r\nFile: %s\r\nLine %d\r\n%s"
                % (details.filename, errlineno, details.msg)
            )
        return line, lineno

    def _load_idle_extensions(self, sub_section, fp, lineno):
        extension_map = self.get_data("idle extensions")
        if extension_map is None:
            extension_map = {}
        extensions = []
        while 1:
            line, lineno, bBreak = self._readline(fp, lineno)
            if bBreak:
                break
            line = line.strip()
            if line:
                extensions.append(line)
        extension_map[sub_section] = extensions
        self._save_data("idle extensions", extension_map)
        return line, lineno


def test():
    import time

    start = time.clock()
    f = "default"
    cm = ConfigManager(f)
    map = cm.get_data("keys")
    took = time.clock() - start
    print("Loaded %s items in %.4f secs" % (len(map), took))


if __name__ == "__main__":
    test()
