# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2003-2007, 2009-2011 Nominum, Inc.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose with or without fee is hereby granted,
# provided that the above copyright notice and this permission notice
# appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND NOMINUM DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL NOMINUM BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import dns.rdtypes.mxbase
import dns.immutable


@dns.immutable.immutable
class AFSDB(dns.rdtypes.mxbase.UncompressedDowncasingMX):

    """AFSDB record"""

    # Use the property mechanism to make "subtype" an alias for the
    # "preference" attribute, and "hostname" an alias for the "exchange"
    # attribute.
    #
    # This lets us inherit the UncompressedMX implementation but lets
    # the caller use appropriate attribute names for the rdata type.
    #
    # We probably lose some performance vs. a cut-and-paste
    # implementation, but this way we don't copy code, and that's
    # good.

    @property
    def subtype(self):
        "the AFSDB subtype"
        return self.preference

    @property
    def hostname(self):
        "the AFSDB hostname"
        return self.exchange
