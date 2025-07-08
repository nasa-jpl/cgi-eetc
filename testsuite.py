# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
testsuite.py

Name tests as ut_*.py in base-level directories (efc/, etc.) and the test
suite will find them.  Assumes the testsuite is in the top level directory, as
it uses its own position to initialize the search.

Optionally takes an additional command-line argument, where it will search
only for that specific file rather than for all unit-test files.

"""

import unittest
import os
import sys
import logging

if __name__ == "__main__":

    testsuite = unittest.TestSuite()

    localpath = os.path.dirname(os.path.abspath(__file__))
    tl = unittest.defaultTestLoader
    if len(sys.argv) > 1:
        testsuite.addTest(tl.discover(localpath, pattern=str(sys.argv[1])))
        pass
    else:
        testsuite.addTest(tl.discover(localpath, pattern='ut_*.py'))
        pass
    pass
    unittest.TextTestRunner(verbosity=1, buffer=True).run(testsuite)
