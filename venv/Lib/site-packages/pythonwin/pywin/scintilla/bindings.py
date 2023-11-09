import traceback

import win32api
import win32con
import win32ui

from . import IDLEenvironment, keycodes

HANDLER_ARGS_GUESS = 0
HANDLER_ARGS_NATIVE = 1
HANDLER_ARGS_IDLE = 2
HANDLER_ARGS_EXTENSION = 3

next_id = 5000

event_to_commands = {}  # dict of integer IDs to event names.
command_to_events = {}  # dict of event names to int IDs


def assign_command_id(event, id=0):
    global next_id
    if id == 0:
        id = event_to_commands.get(event, 0)
        if id == 0:
            id = next_id
            next_id = next_id + 1
        # Only map the ones we allocated - specified ones are assumed to have a handler
        command_to_events[id] = event
    event_to_commands[event] = id
    return id


class SendCommandHandler:
    def __init__(self, cmd):
        self.cmd = cmd

    def __call__(self, *args):
        win32ui.GetMainFrame().SendMessage(win32con.WM_COMMAND, self.cmd)


class Binding:
    def __init__(self, handler, handler_args_type):
        self.handler = handler
        self.handler_args_type = handler_args_type


class BindingsManager:
    def __init__(self, parent_view):
        self.parent_view = parent_view
        self.bindings = {}  # dict of Binding instances.
        self.keymap = {}

    def prepare_configure(self):
        self.keymap = {}

    def complete_configure(self):
        for id in command_to_events.keys():
            self.parent_view.HookCommand(self._OnCommand, id)

    def close(self):
        self.parent_view = self.bindings = self.keymap = None

    def report_error(self, problem):
        try:
            win32ui.SetStatusText(problem, 1)
        except win32ui.error:
            # No status bar!
            print(problem)

    def update_keymap(self, keymap):
        self.keymap.update(keymap)

    def bind(self, event, handler, handler_args_type=HANDLER_ARGS_GUESS, cid=0):
        if handler is None:
            handler = SendCommandHandler(cid)
        self.bindings[event] = self._new_binding(handler, handler_args_type)
        self.bind_command(event, cid)

    def bind_command(self, event, id=0):
        "Binds an event to a Windows control/command ID"
        id = assign_command_id(event, id)
        return id

    def get_command_id(self, event):
        id = event_to_commands.get(event)
        if id is None:
            # See if we even have an event of that name!?
            if event not in self.bindings:
                return None
            id = self.bind_command(event)
        return id

    def _OnCommand(self, id, code):
        event = command_to_events.get(id)
        if event is None:
            self.report_error("No event associated with event ID %d" % id)
            return 1
        return self.fire(event)

    def _new_binding(self, event, handler_args_type):
        return Binding(event, handler_args_type)

    def _get_IDLE_handler(self, ext, handler):
        try:
            instance = self.parent_view.idle.IDLEExtension(ext)
            name = handler.replace("-", "_") + "_event"
            return getattr(instance, name)
        except (ImportError, AttributeError):
            msg = "Can not find event '%s' in IDLE extension '%s'" % (handler, ext)
            self.report_error(msg)
            return None

    def fire(self, event, event_param=None):
        # Fire the specified event.  Result is native Pythonwin result
        # (ie, 1==pass one, 0 or None==handled)

        # First look up the event directly - if there, we are set.
        binding = self.bindings.get(event)
        if binding is None:
            # If possible, find it!
            # A native method name
            handler = getattr(self.parent_view, event + "Event", None)
            if handler is None:
                # Can't decide if I should report an error??
                self.report_error("The event name '%s' can not be found." % event)
                # Either way, just let the default handlers grab it.
                return 1
            binding = self._new_binding(handler, HANDLER_ARGS_NATIVE)
            # Cache it.
            self.bindings[event] = binding

        handler_args_type = binding.handler_args_type
        # Now actually fire it.
        if handler_args_type == HANDLER_ARGS_GUESS:
            # Can't be native, as natives are never added with "guess".
            # Must be extension or IDLE.
            if event[0] == "<":
                handler_args_type = HANDLER_ARGS_IDLE
            else:
                handler_args_type = HANDLER_ARGS_EXTENSION
        try:
            if handler_args_type == HANDLER_ARGS_EXTENSION:
                args = self.parent_view.idle, event_param
            else:
                args = (event_param,)
            rc = binding.handler(*args)
            if handler_args_type == HANDLER_ARGS_IDLE:
                # Convert to our return code.
                if rc in (None, "break"):
                    rc = 0
                else:
                    rc = 1
        except:
            message = "Firing event '%s' failed." % event
            print(message)
            traceback.print_exc()
            self.report_error(message)
            rc = 1  # Let any default handlers have a go!
        return rc

    def fire_key_event(self, msg):
        key = msg[2]
        keyState = 0
        if win32api.GetKeyState(win32con.VK_CONTROL) & 0x8000:
            keyState = (
                keyState | win32con.RIGHT_CTRL_PRESSED | win32con.LEFT_CTRL_PRESSED
            )
        if win32api.GetKeyState(win32con.VK_SHIFT) & 0x8000:
            keyState = keyState | win32con.SHIFT_PRESSED
        if win32api.GetKeyState(win32con.VK_MENU) & 0x8000:
            keyState = keyState | win32con.LEFT_ALT_PRESSED | win32con.RIGHT_ALT_PRESSED
        keyinfo = key, keyState
        # Special hacks for the dead-char key on non-US keyboards.
        # (XXX - which do not work :-(
        event = self.keymap.get(keyinfo)
        if event is None:
            return 1
        return self.fire(event, None)
