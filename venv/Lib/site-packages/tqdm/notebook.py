"""
IPython/Jupyter Notebook progressbar decorator for iterators.
Includes a default `range` iterator printing to `stderr`.

Usage:
>>> from tqdm.notebook import trange, tqdm
>>> for i in trange(10):
...     ...
"""
# import compatibility functions and utilities
import re
import sys
from weakref import proxy

# to inherit from the tqdm class
from .std import tqdm as std_tqdm

if True:  # pragma: no cover
    # import IPython/Jupyter base widget and display utilities
    IPY = 0
    try:  # IPython 4.x
        import ipywidgets
        IPY = 4
    except ImportError:  # IPython 3.x / 2.x
        IPY = 32
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message=".*The `IPython.html` package has been deprecated.*")
            try:
                import IPython.html.widgets as ipywidgets  # NOQA: F401
            except ImportError:
                pass

    try:  # IPython 4.x / 3.x
        if IPY == 32:
            from IPython.html.widgets import HTML
            from IPython.html.widgets import FloatProgress as IProgress
            from IPython.html.widgets import HBox
            IPY = 3
        else:
            from ipywidgets import HTML
            from ipywidgets import FloatProgress as IProgress
            from ipywidgets import HBox
    except ImportError:
        try:  # IPython 2.x
            from IPython.html.widgets import HTML
            from IPython.html.widgets import ContainerWidget as HBox
            from IPython.html.widgets import FloatProgressWidget as IProgress
            IPY = 2
        except ImportError:
            IPY = 0
            IProgress = None
            HBox = object

    try:
        from IPython.display import display  # , clear_output
    except ImportError:
        pass

    # HTML encoding
    try:  # Py3
        from html import escape
    except ImportError:  # Py2
        from cgi import escape

__author__ = {"github.com/": ["lrq3000", "casperdcl", "alexanderkuk"]}
__all__ = ['tqdm_notebook', 'tnrange', 'tqdm', 'trange']
WARN_NOIPYW = ("IProgress not found. Please update jupyter and ipywidgets."
               " See https://ipywidgets.readthedocs.io/en/stable"
               "/user_install.html")


class TqdmHBox(HBox):
    """`ipywidgets.HBox` with a pretty representation"""
    def _json_(self, pretty=None):
        pbar = getattr(self, 'pbar', None)
        if pbar is None:
            return {}
        d = pbar.format_dict
        if pretty is not None:
            d["ascii"] = not pretty
        return d

    def __repr__(self, pretty=False):
        pbar = getattr(self, 'pbar', None)
        if pbar is None:
            return super(TqdmHBox, self).__repr__()
        return pbar.format_meter(**self._json_(pretty))

    def _repr_pretty_(self, pp, *_, **__):
        pp.text(self.__repr__(True))


class tqdm_notebook(std_tqdm):
    """
    Experimental IPython/Jupyter Notebook widget using tqdm!
    """
    @staticmethod
    def status_printer(_, total=None, desc=None, ncols=None):
        """
        Manage the printing of an IPython/Jupyter Notebook progress bar widget.
        """
        # Fallback to text bar if there's no total
        # DEPRECATED: replaced with an 'info' style bar
        # if not total:
        #    return super(tqdm_notebook, tqdm_notebook).status_printer(file)

        # fp = file

        # Prepare IPython progress bar
        if IProgress is None:  # #187 #451 #558 #872
            raise ImportError(WARN_NOIPYW)
        if total:
            pbar = IProgress(min=0, max=total)
        else:  # No total? Show info style bar with no progress tqdm status
            pbar = IProgress(min=0, max=1)
            pbar.value = 1
            pbar.bar_style = 'info'
            if ncols is None:
                pbar.layout.width = "20px"

        ltext = HTML()
        rtext = HTML()
        if desc:
            ltext.value = desc
        container = TqdmHBox(children=[ltext, pbar, rtext])
        # Prepare layout
        if ncols is not None:  # use default style of ipywidgets
            # ncols could be 100, "100px", "100%"
            ncols = str(ncols)  # ipywidgets only accepts string
            try:
                if int(ncols) > 0:  # isnumeric and positive
                    ncols += 'px'
            except ValueError:
                pass
            pbar.layout.flex = '2'
            container.layout.width = ncols
            container.layout.display = 'inline-flex'
            container.layout.flex_flow = 'row wrap'

        return container

    def display(self, msg=None, pos=None,
                # additional signals
                close=False, bar_style=None, check_delay=True):
        # Note: contrary to native tqdm, msg='' does NOT clear bar
        # goal is to keep all infos if error happens so user knows
        # at which iteration the loop failed.

        # Clear previous output (really necessary?)
        # clear_output(wait=1)

        if not msg and not close:
            d = self.format_dict
            # remove {bar}
            d['bar_format'] = (d['bar_format'] or "{l_bar}<bar/>{r_bar}").replace(
                "{bar}", "<bar/>")
            msg = self.format_meter(**d)

        ltext, pbar, rtext = self.container.children
        pbar.value = self.n

        if msg:
            # html escape special characters (like '&')
            if '<bar/>' in msg:
                left, right = map(escape, re.split(r'\|?<bar/>\|?', msg, 1))
            else:
                left, right = '', escape(msg)

            # Update description
            ltext.value = left
            # never clear the bar (signal: msg='')
            if right:
                rtext.value = right

        # Change bar style
        if bar_style:
            # Hack-ish way to avoid the danger bar_style being overridden by
            # success because the bar gets closed after the error...
            if pbar.bar_style != 'danger' or bar_style != 'success':
                pbar.bar_style = bar_style

        # Special signal to close the bar
        if close and pbar.bar_style != 'danger':  # hide only if no error
            try:
                self.container.close()
            except AttributeError:
                self.container.visible = False
            self.container.layout.visibility = 'hidden'  # IPYW>=8

        if check_delay and self.delay > 0 and not self.displayed:
            display(self.container)
            self.displayed = True

    @property
    def colour(self):
        if hasattr(self, 'container'):
            return self.container.children[-2].style.bar_color

    @colour.setter
    def colour(self, bar_color):
        if hasattr(self, 'container'):
            self.container.children[-2].style.bar_color = bar_color

    def __init__(self, *args, **kwargs):
        """
        Supports the usual `tqdm.tqdm` parameters as well as those listed below.

        Parameters
        ----------
        display  : Whether to call `display(self.container)` immediately
            [default: True].
        """
        kwargs = kwargs.copy()
        # Setup default output
        file_kwarg = kwargs.get('file', sys.stderr)
        if file_kwarg is sys.stderr or file_kwarg is None:
            kwargs['file'] = sys.stdout  # avoid the red block in IPython

        # Initialize parent class + avoid printing by using gui=True
        kwargs['gui'] = True
        # convert disable = None to False
        kwargs['disable'] = bool(kwargs.get('disable', False))
        colour = kwargs.pop('colour', None)
        display_here = kwargs.pop('display', True)
        super(tqdm_notebook, self).__init__(*args, **kwargs)
        if self.disable or not kwargs['gui']:
            self.disp = lambda *_, **__: None
            return

        # Get bar width
        self.ncols = '100%' if self.dynamic_ncols else kwargs.get("ncols", None)

        # Replace with IPython progress bar display (with correct total)
        unit_scale = 1 if self.unit_scale is True else self.unit_scale or 1
        total = self.total * unit_scale if self.total else self.total
        self.container = self.status_printer(self.fp, total, self.desc, self.ncols)
        self.container.pbar = proxy(self)
        self.displayed = False
        if display_here and self.delay <= 0:
            display(self.container)
            self.displayed = True
        self.disp = self.display
        self.colour = colour

        # Print initial bar state
        if not self.disable:
            self.display(check_delay=False)

    def __iter__(self):
        try:
            it = super(tqdm_notebook, self).__iter__()
            for obj in it:
                # return super(tqdm...) will not catch exception
                yield obj
        # NB: except ... [ as ...] breaks IPython async KeyboardInterrupt
        except:  # NOQA
            self.disp(bar_style='danger')
            raise
        # NB: don't `finally: close()`
        # since this could be a shared bar which the user will `reset()`

    def update(self, n=1):
        try:
            return super(tqdm_notebook, self).update(n=n)
        # NB: except ... [ as ...] breaks IPython async KeyboardInterrupt
        except:  # NOQA
            # cannot catch KeyboardInterrupt when using manual tqdm
            # as the interrupt will most likely happen on another statement
            self.disp(bar_style='danger')
            raise
        # NB: don't `finally: close()`
        # since this could be a shared bar which the user will `reset()`

    def close(self):
        if self.disable:
            return
        super(tqdm_notebook, self).close()
        # Try to detect if there was an error or KeyboardInterrupt
        # in manual mode: if n < total, things probably got wrong
        if self.total and self.n < self.total:
            self.disp(bar_style='danger', check_delay=False)
        else:
            if self.leave:
                self.disp(bar_style='success', check_delay=False)
            else:
                self.disp(close=True, check_delay=False)

    def clear(self, *_, **__):
        pass

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Consider combining with `leave=True`.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        if self.disable:
            return super(tqdm_notebook, self).reset(total=total)
        _, pbar, _ = self.container.children
        pbar.bar_style = ''
        if total is not None:
            pbar.max = total
            if not self.total and self.ncols is None:  # no longer unknown total
                pbar.layout.width = None  # reset width
        return super(tqdm_notebook, self).reset(total=total)


def tnrange(*args, **kwargs):
    """Shortcut for `tqdm.notebook.tqdm(range(*args), **kwargs)`."""
    return tqdm_notebook(range(*args), **kwargs)


# Aliases
tqdm = tqdm_notebook
trange = tnrange
