COM Test Suite Readme
---------------------

Running the test suite:
-----------------------
* Open a command prompt
* Change to the "win32com\test" directory.
* run "testall.py".  This will perform level 1 testing.
  You may specify 1, 2, or 3 on the command line ("testutil 3")
  to execute more tests.

In general, this should just run the best it can, utilizing what is available
on the machine.  It is likely some tests will refuse to run due to objects not
being locally available - this is normal.

The win32com source tree has source code to a C++ and VB component used purely
for testing.  You may like to build and register these, particularly if you 
are doing anything related to argument/result handling.
