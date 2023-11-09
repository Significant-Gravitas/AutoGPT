import sys

__author__ = "github.com/casperdcl"
__all__ = ['tqdm_pandas']


def tqdm_pandas(tclass, **tqdm_kwargs):
    """
    Registers the given `tqdm` instance with
    `pandas.core.groupby.DataFrameGroupBy.progress_apply`.
    """
    from tqdm import TqdmDeprecationWarning

    if isinstance(tclass, type) or (getattr(tclass, '__name__', '').startswith(
            'tqdm_')):  # delayed adapter case
        TqdmDeprecationWarning(
            "Please use `tqdm.pandas(...)` instead of `tqdm_pandas(tqdm, ...)`.",
            fp_write=getattr(tqdm_kwargs.get('file', None), 'write', sys.stderr.write))
        tclass.pandas(**tqdm_kwargs)
    else:
        TqdmDeprecationWarning(
            "Please use `tqdm.pandas(...)` instead of `tqdm_pandas(tqdm(...))`.",
            fp_write=getattr(tclass.fp, 'write', sys.stderr.write))
        type(tclass).pandas(deprecated_t=tclass)
