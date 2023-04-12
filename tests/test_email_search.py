import unittest
import os
import sys
sys.path.append(os.path.abspath('scripts'))

from outlook_mapi_functions import search_sent_emails

search_sent_emails("welcome")