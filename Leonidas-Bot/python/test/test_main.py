import asyncio

import discord
import discord.ext.test as dpytest


import utils
from messageTypes import MessageTypes


TEST_CLIENT_ID = "TEST_CLIENT_ID"

def test_message_parsing():
    # Here are some sample test cases to check normal parsing of messages
    assert utils.parse_message("Hello there!", TEST_CLIENT_ID) == ("Hello there!", MessageTypes.RESPOND)
    assert utils.parse_message("/private <@{}> How are you doing?".format(TEST_CLIENT_ID), TEST_CLIENT_ID) == ("How are you doing?", MessageTypes.GO_PRIVATE)
    assert utils.parse_message("/reset <@{}>".format(TEST_CLIENT_ID), TEST_CLIENT_ID) == ("", MessageTypes.CHAT_RESET)
    assert utils.parse_message("/reset <@{}> Hi!".format(TEST_CLIENT_ID), TEST_CLIENT_ID) == ("Hi!", MessageTypes.CHAT_RESET)
    assert utils.parse_message("/sleep <@{}>".format(TEST_CLIENT_ID), TEST_CLIENT_ID) == ("/sleep <@{}>".format(TEST_CLIENT_ID), MessageTypes.NO_RESPONSE)
    assert utils.parse_message("/sleep10 <@{}>".format(TEST_CLIENT_ID), TEST_CLIENT_ID) == ("/sleep10 <@{}>".format(TEST_CLIENT_ID), MessageTypes.NO_RESPONSE)
    assert utils.parse_message("!play-50 <@{}> Hi".format(TEST_CLIENT_ID), TEST_CLIENT_ID) == ("!play-50 <@{}> Hi".format(TEST_CLIENT_ID), MessageTypes.NO_RESPONSE)