#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Expression of Service Names in X.509 Certificates
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc4985.txt
#

from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import univ

from pyasn1_modules import rfc5280

MAX = float('inf')


# As specified in Appendix A.2 of RFC 4985

id_pkix = rfc5280.id_pkix

id_on = id_pkix + (8, )

id_on_dnsSRV = id_on + (7, )


class SRVName(char.IA5String):
    subtypeSpec = constraint.ValueSizeConstraint(1, MAX)


srvName = rfc5280.AnotherName()
srvName['type-id'] = id_on_dnsSRV
srvName['value'] = SRVName()


# Map of Other Name OIDs to Other Name is added to the
# ones that are in rfc5280.py

_anotherNameMapUpdate = {
    id_on_dnsSRV: SRVName(),
}

rfc5280.anotherNameMap.update(_anotherNameMapUpdate)
