#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
from sys import version_info

if version_info[0] <= 2:
    int2oct = chr
    # noinspection PyPep8
    ints2octs = lambda s: ''.join([int2oct(x) for x in s])
    null = ''
    oct2int = ord
    # TODO: refactor to return a sequence of ints
    # noinspection PyPep8
    octs2ints = lambda s: [oct2int(x) for x in s]
    # noinspection PyPep8
    str2octs = lambda x: x
    # noinspection PyPep8
    octs2str = lambda x: x
    # noinspection PyPep8
    isOctetsType = lambda s: isinstance(s, str)
    # noinspection PyPep8
    isStringType = lambda s: isinstance(s, (str, unicode))
    # noinspection PyPep8
    ensureString = str
else:
    ints2octs = bytes
    # noinspection PyPep8
    int2oct = lambda x: ints2octs((x,))
    null = ints2octs()
    # noinspection PyPep8
    oct2int = lambda x: x
    # noinspection PyPep8
    octs2ints = lambda x: x
    # noinspection PyPep8
    str2octs = lambda x: x.encode('iso-8859-1')
    # noinspection PyPep8
    octs2str = lambda x: x.decode('iso-8859-1')
    # noinspection PyPep8
    isOctetsType = lambda s: isinstance(s, bytes)
    # noinspection PyPep8
    isStringType = lambda s: isinstance(s, str)
    # noinspection PyPep8
    ensureString = bytes
