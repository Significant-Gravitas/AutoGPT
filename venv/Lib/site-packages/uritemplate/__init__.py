"""

uritemplate
===========

URI templates implemented as close to :rfc:`6570` as possible

See http://uritemplate.rtfd.org/ for documentation

:copyright:
    (c) 2013 Ian Stapleton Cordasco
:license:
    Modified BSD Apache License (Version 2.0), see LICENSE for more details
    and either LICENSE.BSD or LICENSE.APACHE for the details of those specific
    licenses

"""

__title__ = "uritemplate"
__author__ = "Ian Stapleton Cordasco"
__license__ = "Modified BSD or Apache License, Version 2.0"
__copyright__ = "Copyright 2013 Ian Stapleton Cordasco"
__version__ = "4.1.1"
__version_info__ = tuple(
    int(i) for i in __version__.split(".") if i.isdigit()
)

from uritemplate.api import (
    URITemplate,
    expand,
    partial,
    variables,
)

__all__ = ("URITemplate", "expand", "partial", "variables")
