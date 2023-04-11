from functools import partial

from dask.callbacks import Callback

from .auto import tqdm as tqdm_auto

__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['TqdmCallback']


class TqdmCallback(Callback):
    """Dask callback for task progress."""
    def __init__(self, start=None, pretask=None, tqdm_class=tqdm_auto,
                 **tqdm_kwargs):
        """
        Parameters
        ----------
        tqdm_class  : optional
            `tqdm` class to use for bars [default: `tqdm.auto.tqdm`].
        tqdm_kwargs  : optional
            Any other arguments used for all bars.
        """
        super(TqdmCallback, self).__init__(start=start, pretask=pretask)
        if tqdm_kwargs:
            tqdm_class = partial(tqdm_class, **tqdm_kwargs)
        self.tqdm_class = tqdm_class

    def _start_state(self, _, state):
        self.pbar = self.tqdm_class(total=sum(
            len(state[k]) for k in ['ready', 'waiting', 'running', 'finished']))

    def _posttask(self, *_, **__):
        self.pbar.update()

    def _finish(self, *_, **__):
        self.pbar.close()

    def display(self):
        """Displays in the current cell in Notebooks."""
        container = getattr(self.bar, 'container', None)
        if container is None:
            return
        from .notebook import display
        display(container)
