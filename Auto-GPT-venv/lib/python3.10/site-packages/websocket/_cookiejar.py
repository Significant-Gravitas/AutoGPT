import http.cookies

"""
_cookiejar.py
websocket - WebSocket client library for Python

Copyright 2022 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


class SimpleCookieJar:
    def __init__(self):
        self.jar = dict()

    def add(self, set_cookie):
        if set_cookie:
            simpleCookie = http.cookies.SimpleCookie(set_cookie)

            for k, v in simpleCookie.items():
                domain = v.get("domain")
                if domain:
                    if not domain.startswith("."):
                        domain = "." + domain
                    cookie = self.jar.get(domain) if self.jar.get(domain) else http.cookies.SimpleCookie()
                    cookie.update(simpleCookie)
                    self.jar[domain.lower()] = cookie

    def set(self, set_cookie):
        if set_cookie:
            simpleCookie = http.cookies.SimpleCookie(set_cookie)

            for k, v in simpleCookie.items():
                domain = v.get("domain")
                if domain:
                    if not domain.startswith("."):
                        domain = "." + domain
                    self.jar[domain.lower()] = simpleCookie

    def get(self, host):
        if not host:
            return ""

        cookies = []
        for domain, simpleCookie in self.jar.items():
            host = host.lower()
            if host.endswith(domain) or host == domain[1:]:
                cookies.append(self.jar.get(domain))

        return "; ".join(filter(
            None, sorted(
                ["%s=%s" % (k, v.value) for cookie in filter(None, cookies) for k, v in cookie.items()]
            )))
