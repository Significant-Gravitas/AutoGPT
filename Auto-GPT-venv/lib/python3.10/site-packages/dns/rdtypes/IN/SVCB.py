# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

import dns.rdtypes.svcbbase
import dns.immutable


@dns.immutable.immutable
class SVCB(dns.rdtypes.svcbbase.SVCBBase):
    """SVCB record"""
