import unittest

import win32net
import win32netcon


class TestCase(unittest.TestCase):
    def testGroupsGoodResume(self, server=None):
        res = 0
        level = 0  # setting it to 1 will provide more detailed info
        while True:
            (user_list, total, res) = win32net.NetGroupEnum(server, level, res)
            for i in user_list:
                pass
            if not res:
                break

    def testGroupsBadResume(self, server=None):
        res = 1  # Can't pass this first time round.
        self.assertRaises(win32net.error, win32net.NetGroupEnum, server, 0, res)


if __name__ == "__main__":
    unittest.main()
