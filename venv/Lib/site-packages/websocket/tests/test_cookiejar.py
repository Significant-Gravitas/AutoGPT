import unittest
from websocket._cookiejar import SimpleCookieJar

"""
test_cookiejar.py
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


class CookieJarTest(unittest.TestCase):
    def testAdd(self):
        cookie_jar = SimpleCookieJar()
        cookie_jar.add("")
        self.assertFalse(cookie_jar.jar, "Cookie with no domain should not be added to the jar")

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b")
        self.assertFalse(cookie_jar.jar, "Cookie with no domain should not be added to the jar")

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; domain=.abc")
        self.assertTrue(".abc" in cookie_jar.jar)

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; domain=abc")
        self.assertTrue(".abc" in cookie_jar.jar)
        self.assertTrue("abc" not in cookie_jar.jar)

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; c=d; domain=abc")
        self.assertEqual(cookie_jar.get("abc"), "a=b; c=d")
        self.assertEqual(cookie_jar.get(None), "")

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; c=d; domain=abc")
        cookie_jar.add("e=f; domain=abc")
        self.assertEqual(cookie_jar.get("abc"), "a=b; c=d; e=f")

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; c=d; domain=abc")
        cookie_jar.add("e=f; domain=.abc")
        self.assertEqual(cookie_jar.get("abc"), "a=b; c=d; e=f")

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; c=d; domain=abc")
        cookie_jar.add("e=f; domain=xyz")
        self.assertEqual(cookie_jar.get("abc"), "a=b; c=d")
        self.assertEqual(cookie_jar.get("xyz"), "e=f")
        self.assertEqual(cookie_jar.get("something"), "")

    def testSet(self):
        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b")
        self.assertFalse(cookie_jar.jar, "Cookie with no domain should not be added to the jar")

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; domain=.abc")
        self.assertTrue(".abc" in cookie_jar.jar)

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; domain=abc")
        self.assertTrue(".abc" in cookie_jar.jar)
        self.assertTrue("abc" not in cookie_jar.jar)

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc")
        self.assertEqual(cookie_jar.get("abc"), "a=b; c=d")

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc")
        cookie_jar.set("e=f; domain=abc")
        self.assertEqual(cookie_jar.get("abc"), "e=f")

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc")
        cookie_jar.set("e=f; domain=.abc")
        self.assertEqual(cookie_jar.get("abc"), "e=f")

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc")
        cookie_jar.set("e=f; domain=xyz")
        self.assertEqual(cookie_jar.get("abc"), "a=b; c=d")
        self.assertEqual(cookie_jar.get("xyz"), "e=f")
        self.assertEqual(cookie_jar.get("something"), "")

    def testGet(self):
        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc.com")
        self.assertEqual(cookie_jar.get("abc.com"), "a=b; c=d")
        self.assertEqual(cookie_jar.get("x.abc.com"), "a=b; c=d")
        self.assertEqual(cookie_jar.get("abc.com.es"), "")
        self.assertEqual(cookie_jar.get("xabc.com"), "")

        cookie_jar.set("a=b; c=d; domain=.abc.com")
        self.assertEqual(cookie_jar.get("abc.com"), "a=b; c=d")
        self.assertEqual(cookie_jar.get("x.abc.com"), "a=b; c=d")
        self.assertEqual(cookie_jar.get("abc.com.es"), "")
        self.assertEqual(cookie_jar.get("xabc.com"), "")


if __name__ == "__main__":
    unittest.main()
