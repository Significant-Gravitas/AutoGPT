import math as math
from typing import Any

from numpy._pytesttester import PytestTester

from numpy import (
    ndenumerate as ndenumerate,
    ndindex as ndindex,
)

from numpy.version import version

from numpy.lib import (
    format as format,
    mixins as mixins,
    scimath as scimath,
    stride_tricks as stride_tricks,
)

from numpy.lib._version import (
    NumpyVersion as NumpyVersion,
)

from numpy.lib.arraypad import (
    pad as pad,
)

from numpy.lib.arraysetops import (
    ediff1d as ediff1d,
    intersect1d as intersect1d,
    setxor1d as setxor1d,
    union1d as union1d,
    setdiff1d as setdiff1d,
    unique as unique,
    in1d as in1d,
    isin as isin,
)

from numpy.lib.arrayterator import (
    Arrayterator as Arrayterator,
)

from numpy.lib.function_base import (
    select as select,
    piecewise as piecewise,
    trim_zeros as trim_zeros,
    copy as copy,
    iterable as iterable,
    percentile as percentile,
    diff as diff,
    gradient as gradient,
    angle as angle,
    unwrap as unwrap,
    sort_complex as sort_complex,
    disp as disp,
    flip as flip,
    rot90 as rot90,
    extract as extract,
    place as place,
    vectorize as vectorize,
    asarray_chkfinite as asarray_chkfinite,
    average as average,
    bincount as bincount,
    digitize as digitize,
    cov as cov,
    corrcoef as corrcoef,
    msort as msort,
    median as median,
    sinc as sinc,
    hamming as hamming,
    hanning as hanning,
    bartlett as bartlett,
    blackman as blackman,
    kaiser as kaiser,
    trapz as trapz,
    i0 as i0,
    add_newdoc as add_newdoc,
    add_docstring as add_docstring,
    meshgrid as meshgrid,
    delete as delete,
    insert as insert,
    append as append,
    interp as interp,
    add_newdoc_ufunc as add_newdoc_ufunc,
    quantile as quantile,
)

from numpy.lib.histograms import (
    histogram_bin_edges as histogram_bin_edges,
    histogram as histogram,
    histogramdd as histogramdd,
)

from numpy.lib.index_tricks import (
    ravel_multi_index as ravel_multi_index,
    unravel_index as unravel_index,
    mgrid as mgrid,
    ogrid as ogrid,
    r_ as r_,
    c_ as c_,
    s_ as s_,
    index_exp as index_exp,
    ix_ as ix_,
    fill_diagonal as fill_diagonal,
    diag_indices as diag_indices,
    diag_indices_from as diag_indices_from,
)

from numpy.lib.nanfunctions import (
    nansum as nansum,
    nanmax as nanmax,
    nanmin as nanmin,
    nanargmax as nanargmax,
    nanargmin as nanargmin,
    nanmean as nanmean,
    nanmedian as nanmedian,
    nanpercentile as nanpercentile,
    nanvar as nanvar,
    nanstd as nanstd,
    nanprod as nanprod,
    nancumsum as nancumsum,
    nancumprod as nancumprod,
    nanquantile as nanquantile,
)

from numpy.lib.npyio import (
    savetxt as savetxt,
    loadtxt as loadtxt,
    genfromtxt as genfromtxt,
    recfromtxt as recfromtxt,
    recfromcsv as recfromcsv,
    load as load,
    save as save,
    savez as savez,
    savez_compressed as savez_compressed,
    packbits as packbits,
    unpackbits as unpackbits,
    fromregex as fromregex,
    DataSource as DataSource,
)

from numpy.lib.polynomial import (
    poly as poly,
    roots as roots,
    polyint as polyint,
    polyder as polyder,
    polyadd as polyadd,
    polysub as polysub,
    polymul as polymul,
    polydiv as polydiv,
    polyval as polyval,
    polyfit as polyfit,
    RankWarning as RankWarning,
    poly1d as poly1d,
)

from numpy.lib.shape_base import (
    column_stack as column_stack,
    row_stack as row_stack,
    dstack as dstack,
    array_split as array_split,
    split as split,
    hsplit as hsplit,
    vsplit as vsplit,
    dsplit as dsplit,
    apply_over_axes as apply_over_axes,
    expand_dims as expand_dims,
    apply_along_axis as apply_along_axis,
    kron as kron,
    tile as tile,
    get_array_wrap as get_array_wrap,
    take_along_axis as take_along_axis,
    put_along_axis as put_along_axis,
)

from numpy.lib.stride_tricks import (
    broadcast_to as broadcast_to,
    broadcast_arrays as broadcast_arrays,
    broadcast_shapes as broadcast_shapes,
)

from numpy.lib.twodim_base import (
    diag as diag,
    diagflat as diagflat,
    eye as eye,
    fliplr as fliplr,
    flipud as flipud,
    tri as tri,
    triu as triu,
    tril as tril,
    vander as vander,
    histogram2d as histogram2d,
    mask_indices as mask_indices,
    tril_indices as tril_indices,
    tril_indices_from as tril_indices_from,
    triu_indices as triu_indices,
    triu_indices_from as triu_indices_from,
)

from numpy.lib.type_check import (
    mintypecode as mintypecode,
    asfarray as asfarray,
    real as real,
    imag as imag,
    iscomplex as iscomplex,
    isreal as isreal,
    iscomplexobj as iscomplexobj,
    isrealobj as isrealobj,
    nan_to_num as nan_to_num,
    real_if_close as real_if_close,
    typename as typename,
    common_type as common_type,
)

from numpy.lib.ufunclike import (
    fix as fix,
    isposinf as isposinf,
    isneginf as isneginf,
)

from numpy.lib.utils import (
    issubclass_ as issubclass_,
    issubsctype as issubsctype,
    issubdtype as issubdtype,
    deprecate as deprecate,
    deprecate_with_doc as deprecate_with_doc,
    get_include as get_include,
    info as info,
    source as source,
    who as who,
    lookfor as lookfor,
    byte_bounds as byte_bounds,
    safe_eval as safe_eval,
    show_runtime as show_runtime,
)

from numpy.core.multiarray import (
    tracemalloc_domain as tracemalloc_domain,
)

__all__: list[str]
__path__: list[str]
test: PytestTester

__version__ = version
emath = scimath
