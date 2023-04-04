"""
Tkinter GUI progressbar decorator for iterators.

Usage:
>>> from tqdm.tk import trange, tqdm
>>> for i in trange(10):
...     ...
"""
import re
import sys
import tkinter
import tkinter.ttk as ttk
from warnings import warn

from .std import TqdmExperimentalWarning, TqdmWarning
from .std import tqdm as std_tqdm

__author__ = {"github.com/": ["richardsheridan", "casperdcl"]}
__all__ = ['tqdm_tk', 'ttkrange', 'tqdm', 'trange']


class tqdm_tk(std_tqdm):  # pragma: no cover
    """
    Experimental Tkinter GUI version of tqdm!

    Note: Window interactivity suffers if `tqdm_tk` is not running within
    a Tkinter mainloop and values are generated infrequently. In this case,
    consider calling `tqdm_tk.refresh()` frequently in the Tk thread.
    """

    # TODO: @classmethod: write()?

    def __init__(self, *args, **kwargs):
        """
        This class accepts the following parameters *in addition* to
        the parameters accepted by `tqdm`.

        Parameters
        ----------
        grab  : bool, optional
            Grab the input across all windows of the process.
        tk_parent  : `tkinter.Wm`, optional
            Parent Tk window.
        cancel_callback  : Callable, optional
            Create a cancel button and set `cancel_callback` to be called
            when the cancel or window close button is clicked.
        """
        kwargs = kwargs.copy()
        kwargs['gui'] = True
        # convert disable = None to False
        kwargs['disable'] = bool(kwargs.get('disable', False))
        self._warn_leave = 'leave' in kwargs
        grab = kwargs.pop('grab', False)
        tk_parent = kwargs.pop('tk_parent', None)
        self._cancel_callback = kwargs.pop('cancel_callback', None)
        super(tqdm_tk, self).__init__(*args, **kwargs)

        if self.disable:
            return

        if tk_parent is None:  # Discover parent widget
            try:
                tk_parent = tkinter._default_root
            except AttributeError:
                raise AttributeError(
                    "`tk_parent` required when using `tkinter.NoDefaultRoot()`")
            if tk_parent is None:  # use new default root window as display
                self._tk_window = tkinter.Tk()
            else:  # some other windows already exist
                self._tk_window = tkinter.Toplevel()
        else:
            self._tk_window = tkinter.Toplevel(tk_parent)

        warn("GUI is experimental/alpha", TqdmExperimentalWarning, stacklevel=2)
        self._tk_dispatching = self._tk_dispatching_helper()

        self._tk_window.protocol("WM_DELETE_WINDOW", self.cancel)
        self._tk_window.wm_title(self.desc)
        self._tk_window.wm_attributes("-topmost", 1)
        self._tk_window.after(0, lambda: self._tk_window.wm_attributes("-topmost", 0))
        self._tk_n_var = tkinter.DoubleVar(self._tk_window, value=0)
        self._tk_text_var = tkinter.StringVar(self._tk_window)
        pbar_frame = ttk.Frame(self._tk_window, padding=5)
        pbar_frame.pack()
        _tk_label = ttk.Label(pbar_frame, textvariable=self._tk_text_var,
                              wraplength=600, anchor="center", justify="center")
        _tk_label.pack()
        self._tk_pbar = ttk.Progressbar(
            pbar_frame, variable=self._tk_n_var, length=450)
        if self.total is not None:
            self._tk_pbar.configure(maximum=self.total)
        else:
            self._tk_pbar.configure(mode="indeterminate")
        self._tk_pbar.pack()
        if self._cancel_callback is not None:
            _tk_button = ttk.Button(pbar_frame, text="Cancel", command=self.cancel)
            _tk_button.pack()
        if grab:
            self._tk_window.grab_set()

    def close(self):
        if self.disable:
            return

        self.disable = True

        with self.get_lock():
            self._instances.remove(self)

        def _close():
            self._tk_window.after('idle', self._tk_window.destroy)
            if not self._tk_dispatching:
                self._tk_window.update()

        self._tk_window.protocol("WM_DELETE_WINDOW", _close)

        # if leave is set but we are self-dispatching, the left window is
        # totally unresponsive unless the user manually dispatches
        if not self.leave:
            _close()
        elif not self._tk_dispatching:
            if self._warn_leave:
                warn("leave flag ignored if not in tkinter mainloop",
                     TqdmWarning, stacklevel=2)
            _close()

    def clear(self, *_, **__):
        pass

    def display(self, *_, **__):
        self._tk_n_var.set(self.n)
        d = self.format_dict
        # remove {bar}
        d['bar_format'] = (d['bar_format'] or "{l_bar}<bar/>{r_bar}").replace(
            "{bar}", "<bar/>")
        msg = self.format_meter(**d)
        if '<bar/>' in msg:
            msg = "".join(re.split(r'\|?<bar/>\|?', msg, 1))
        self._tk_text_var.set(msg)
        if not self._tk_dispatching:
            self._tk_window.update()

    def set_description(self, desc=None, refresh=True):
        self.set_description_str(desc, refresh)

    def set_description_str(self, desc=None, refresh=True):
        self.desc = desc
        if not self.disable:
            self._tk_window.wm_title(desc)
            if refresh and not self._tk_dispatching:
                self._tk_window.update()

    def cancel(self):
        """
        `cancel_callback()` followed by `close()`
        when close/cancel buttons clicked.
        """
        if self._cancel_callback is not None:
            self._cancel_callback()
        self.close()

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        if hasattr(self, '_tk_pbar'):
            if total is None:
                self._tk_pbar.configure(maximum=100, mode="indeterminate")
            else:
                self._tk_pbar.configure(maximum=total, mode="determinate")
        super(tqdm_tk, self).reset(total=total)

    @staticmethod
    def _tk_dispatching_helper():
        """determine if Tkinter mainloop is dispatching events"""
        codes = {tkinter.mainloop.__code__, tkinter.Misc.mainloop.__code__}
        for frame in sys._current_frames().values():
            while frame:
                if frame.f_code in codes:
                    return True
                frame = frame.f_back
        return False


def ttkrange(*args, **kwargs):
    """Shortcut for `tqdm.tk.tqdm(range(*args), **kwargs)`."""
    return tqdm_tk(range(*args), **kwargs)


# Aliases
tqdm = tqdm_tk
trange = ttkrange
