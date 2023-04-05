#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Extensible Messaging and Presence Protocol (XMPP)
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc6120.txt
#

from pyasn1.type import char
from pyasn1.type import univ

from pyasn1_modules import rfc5280

MAX = float('inf')


# XmppAddr Identifier Type as specified in Section 13.7.1.4. of RFC 6120

id_pkix = rfc5280.id_pkix

id_on = id_pkix + (8, )

id_on_xmppAddr = id_on + (5, )


class XmppAddr(char.UTF8String):
    pass


# Map of Other Name OIDs to Other Name is added to the
# ones that are in rfc5280.py

_anotherNameMapUpdate = {
    id_on_xmppAddr: XmppAddr(),
}

rfc5280.anotherNameMap.update(_anotherNameMapUpdate)
