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

"""A place to store TSIG keys."""

from typing import Any, Dict

import base64

import dns.name
import dns.tsig


def from_text(textring: Dict[str, Any]) -> Dict[dns.name.Name, dns.tsig.Key]:
    """Convert a dictionary containing (textual DNS name, base64 secret)
    pairs into a binary keyring which has (dns.name.Name, bytes) pairs, or
    a dictionary containing (textual DNS name, (algorithm, base64 secret))
    pairs into a binary keyring which has (dns.name.Name, dns.tsig.Key) pairs.
    @rtype: dict"""

    keyring = {}
    for (name, value) in textring.items():
        kname = dns.name.from_text(name)
        if isinstance(value, str):
            keyring[kname] = dns.tsig.Key(kname, value).secret
        else:
            (algorithm, secret) = value
            keyring[kname] = dns.tsig.Key(kname, secret, algorithm)
    return keyring


def to_text(keyring: Dict[dns.name.Name, Any]) -> Dict[str, Any]:
    """Convert a dictionary containing (dns.name.Name, dns.tsig.Key) pairs
    into a text keyring which has (textual DNS name, (textual algorithm,
    base64 secret)) pairs, or a dictionary containing (dns.name.Name, bytes)
    pairs into a text keyring which has (textual DNS name, base64 secret) pairs.
    @rtype: dict"""

    textring = {}

    def b64encode(secret):
        return base64.encodebytes(secret).decode().rstrip()

    for (name, key) in keyring.items():
        tname = name.to_text()
        if isinstance(key, bytes):
            textring[tname] = b64encode(key)
        else:
            if isinstance(key.secret, bytes):
                text_secret = b64encode(key.secret)
            else:
                text_secret = str(key.secret)

            textring[tname] = (key.algorithm.to_text(), text_secret)
    return textring
