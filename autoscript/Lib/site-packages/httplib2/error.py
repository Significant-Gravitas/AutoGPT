# All exceptions raised here derive from HttpLib2Error
class HttpLib2Error(Exception):
    pass


# Some exceptions can be caught and optionally
# be turned back into responses.
class HttpLib2ErrorWithResponse(HttpLib2Error):
    def __init__(self, desc, response, content):
        self.response = response
        self.content = content
        HttpLib2Error.__init__(self, desc)


class RedirectMissingLocation(HttpLib2ErrorWithResponse):
    pass


class RedirectLimit(HttpLib2ErrorWithResponse):
    pass


class FailedToDecompressContent(HttpLib2ErrorWithResponse):
    pass


class UnimplementedDigestAuthOptionError(HttpLib2ErrorWithResponse):
    pass


class UnimplementedHmacDigestAuthOptionError(HttpLib2ErrorWithResponse):
    pass


class MalformedHeader(HttpLib2Error):
    pass


class RelativeURIError(HttpLib2Error):
    pass


class ServerNotFoundError(HttpLib2Error):
    pass


class ProxiesUnavailableError(HttpLib2Error):
    pass
