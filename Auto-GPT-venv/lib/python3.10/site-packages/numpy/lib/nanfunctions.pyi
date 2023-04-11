from numpy.core.fromnumeric import (
    amin,
    amax,
    argmin,
    argmax,
    sum,
    prod,
    cumsum,
    cumprod,
    mean,
    var,
    std
)

from numpy.lib.function_base import (
    median,
    percentile,
    quantile,
)

__all__: list[str]

# NOTE: In reaility these functions are not aliases but distinct functions
# with identical signatures.
nanmin = amin
nanmax = amax
nanargmin = argmin
nanargmax = argmax
nansum = sum
nanprod = prod
nancumsum = cumsum
nancumprod = cumprod
nanmean = mean
nanvar = var
nanstd = std
nanmedian = median
nanpercentile = percentile
nanquantile = quantile
