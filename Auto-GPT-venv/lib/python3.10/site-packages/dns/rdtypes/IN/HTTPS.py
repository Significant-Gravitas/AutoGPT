# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

import dns.rdtypes.svcbbase
import dns.immutable


@dns.immutable.immutable
class HTTPS(dns.rdtypes.svcbbase.SVCBBase):
    """HTTPS record"""
