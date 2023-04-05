import unittest

import win32com.test.util
from win32com.client.dynamic import DumbDispatch
from win32com.client.gencache import EnsureDispatch


class RegexTest(win32com.test.util.TestCase):
    def _CheckMatches(self, match, expected):
        found = []
        for imatch in match:
            found.append(imatch.FirstIndex)
        self.assertEqual(list(found), list(expected))

    def _TestVBScriptRegex(self, re):
        StringToSearch = "Python python pYthon Python"
        re.Pattern = "Python"
        re.Global = True

        re.IgnoreCase = True
        match = re.Execute(StringToSearch)
        expected = 0, 7, 14, 21
        self._CheckMatches(match, expected)

        re.IgnoreCase = False
        match = re.Execute(StringToSearch)
        expected = 0, 21
        self._CheckMatches(match, expected)

    def testDynamic(self):
        re = DumbDispatch("VBScript.Regexp")
        self._TestVBScriptRegex(re)

    def testGenerated(self):
        re = EnsureDispatch("VBScript.Regexp")
        self._TestVBScriptRegex(re)


if __name__ == "__main__":
    unittest.main()
