import base64
import re

import pyparsing as pp

from .error import *


try:  # pyparsing>=3.0.0
    downcaseTokens = pp.common.downcaseTokens
except AttributeError:
    downcaseTokens = pp.downcaseTokens

UNQUOTE_PAIRS = re.compile(r"\\(.)")
unquote = lambda s, l, t: UNQUOTE_PAIRS.sub(r"\1", t[0][1:-1])

# https://tools.ietf.org/html/rfc7235#section-1.2
# https://tools.ietf.org/html/rfc7235#appendix-B
tchar = "!#$%&'*+-.^_`|~" + pp.nums + pp.alphas
token = pp.Word(tchar).setName("token")
token68 = pp.Combine(pp.Word("-._~+/" + pp.nums + pp.alphas) + pp.Optional(pp.Word("=").leaveWhitespace())).setName(
    "token68"
)

quoted_string = pp.dblQuotedString.copy().setName("quoted-string").setParseAction(unquote)
auth_param_name = token.copy().setName("auth-param-name").addParseAction(downcaseTokens)
auth_param = auth_param_name + pp.Suppress("=") + (quoted_string | token)
params = pp.Dict(pp.delimitedList(pp.Group(auth_param)))

scheme = token("scheme")
challenge = scheme + (params("params") | token68("token"))

authentication_info = params.copy()
www_authenticate = pp.delimitedList(pp.Group(challenge))


def _parse_authentication_info(headers, headername="authentication-info"):
    """https://tools.ietf.org/html/rfc7615
    """
    header = headers.get(headername, "").strip()
    if not header:
        return {}
    try:
        parsed = authentication_info.parseString(header)
    except pp.ParseException as ex:
        # print(ex.explain(ex))
        raise MalformedHeader(headername)

    return parsed.asDict()


def _parse_www_authenticate(headers, headername="www-authenticate"):
    """Returns a dictionary of dictionaries, one dict per auth_scheme."""
    header = headers.get(headername, "").strip()
    if not header:
        return {}
    try:
        parsed = www_authenticate.parseString(header)
    except pp.ParseException as ex:
        # print(ex.explain(ex))
        raise MalformedHeader(headername)

    retval = {
        challenge["scheme"].lower(): challenge["params"].asDict()
        if "params" in challenge
        else {"token": challenge.get("token")}
        for challenge in parsed
    }
    return retval
