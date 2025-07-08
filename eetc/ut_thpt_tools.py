# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit test suite for thpt_tools module.
"""
import copy
import unittest

import numpy as np

from eetc.thpt_tools import ThptToolsException, calc_thpt, _interp_thptcurve


class TestThptTools(unittest.TestCase):
    """
    Unit tests for thpt_tools module.
    """

    def setUp(self):
        # Set up reference thptcurve
        self.lams = np.array(
            [
                4600.0,
                5500.0,
                6400.0,
                7100.0,  # input will not necessarily be evenly spaced
                8200.0
            ]
        )
        self.thpts = np.array(
            [
                0.,
                0.15,  # input will not necessarily be evenly spaced
                0.5,
                0.75,
                1.0
            ]
        )
        self.thptcurve = (self.lams, self.thpts)

        # Set up invalid thptcurve
        self.thpts_err = self.thpts.copy()
        self.thpts_err[0] = -0.1
        self.thpts_err[-1] = 1.1
        self.thptcurve_err = (self.lams, self.thpts_err)

        # Set up uniform coating_thptcurves
        self.coating_thptcurves1 = {
            'alum': (self.lams, np.ones_like(self.thpts)),
            'IOI': (self.lams, np.ones_like(self.thpts)),
            'lens_glass': (self.lams, np.ones_like(self.thpts)),
            'broadband_ar': (self.lams, np.ones_like(self.thpts)),
            'hrc': (self.lams, np.ones_like(self.thpts)),
            'emccd': (self.lams, np.ones_like(self.thpts))
        }
        # Set up unique coating_thptcurves (unique at thpts[0])
        self.coating_thptcurves = copy.deepcopy(self.coating_thptcurves1)
        self.coating_thptcurves['alum'][1][0] = 0.01
        self.coating_thptcurves['IOI'][1][0] = 0.02
        self.coating_thptcurves['lens_glass'][1][0] = 0.03
        self.coating_thptcurves['broadband_ar'][1][0] = 0.04
        self.coating_thptcurves['hrc'][1][0] = 0.06
        self.coating_thptcurves['emccd'][1][0] = 0.07

        # Set up uniform setting_thptcurves
        self.setting_thptcurves1 = {
            'dms': (self.lams, np.ones_like(self.thpts)),
            'spam': (self.lams, np.ones_like(self.thpts)),
            'fpam': (self.lams, np.ones_like(self.thpts)),
            'lsam': (self.lams, np.ones_like(self.thpts)),
            'fsam': (self.lams, np.ones_like(self.thpts)),
            'dpam': (self.lams, np.ones_like(self.thpts))
        }
        # Set up unique setting_thptcurves (unique at thpts[0])
        self.setting_thptcurves = copy.deepcopy(self.setting_thptcurves1)
        self.setting_thptcurves['dms'][1][0] = 0.07
        self.setting_thptcurves['spam'][1][0] = 0.08
        self.setting_thptcurves['fpam'][1][0] = 0.09
        self.setting_thptcurves['lsam'][1][0] = 0.10
        self.setting_thptcurves['fsam'][1][0] = 0.11
        self.setting_thptcurves['dpam'][1][0] = 0.12

        # Set up uniform coating_config
        self.coating_config1 = {
            'alum': 1,
            'IOI': 1,
            'lens_glass': 1,
            'broadband_ar': 1,
            'hrc': 1,
            'emccd': 1,
        }
        # Set up unique coating_config
        self.coating_config = {
            'alum': 2,
            'IOI': 3,
            'lens_glass': 4,
            'broadband_ar': 5,
            'hrc': 7,
            'emccd': 1,
        }
        # Lambda at lams[0]
        self.lam0 = self.lams[0]

    def test_coating_thptcurves(self):
        """
        Verify coating throughput curves are being interpolated, matched
        with correct coating config numbers, and multiplied.
        """
        ct = copy.deepcopy(self.coating_thptcurves)

        thpt_coatings = (
            ct['alum'][1][0] ** self.coating_config['alum']
            * ct['IOI'][1][0] ** self.coating_config['IOI']
            * ct['lens_glass'][1][0] ** self.coating_config['lens_glass']
            * ct['broadband_ar'][1][0] ** self.coating_config['broadband_ar']
            * ct['hrc'][1][0] ** self.coating_config['hrc']
        )
        ct['emccd'][1][0] = 1.

        thpt = calc_thpt(ct, self.setting_thptcurves1,
                         self.coating_config, self.lam0)

        self.assertEqual(thpt, thpt_coatings)

    def test_setting_thptcurves(self):
        """
        Verify element throughputs are being interpolated and multiplied
        together.
        """
        thpt_check = 1
        for _, thpts in self.setting_thptcurves.items():
            thpt_check *= thpts[1][0]

        thpt = calc_thpt(self.coating_thptcurves1, self.setting_thptcurves,
                         self.coating_config1, self.lam0)

        self.assertEqual(thpt, thpt_check)

    def test_emccd_thptcurves(self):
        """
        Verify emccd throughput curve is being interpolated.
        """
        coating_thptcurves = copy.deepcopy(self.coating_thptcurves1)
        coating_thptcurves['emccd'] = \
            self.coating_thptcurves['emccd']

        thpt = calc_thpt(coating_thptcurves, self.setting_thptcurves1,
                         self.coating_config1, self.lam0)

        self.assertEqual(thpt, self.coating_thptcurves['emccd'][1][0])

    def test_thpt(self):
        """
        Verify all sub-throughputs are multiplied together to get final
        throughput.
        """
        thpt_coating_check = 1
        for _, thpts in self.coating_thptcurves.items():
            thpt_coating_check *= thpts[1][0]

        thpt_elements_check = 1
        for _, thpts in self.setting_thptcurves.items():
            thpt_elements_check *= thpts[1][0]

        thpt = calc_thpt(self.coating_thptcurves, self.setting_thptcurves,
                         self.coating_config1, self.lam0)

        self.assertEqual(thpt, thpt_coating_check * thpt_elements_check)

    def test_interp_thptcurve_exact(self):
        """
        Verify correct linear interpolation when evaluating exactly at array
        value.
        """
        i = 1
        lam = self.lams[i]
        thpt_check = self.thpts[i]

        thpt = _interp_thptcurve(self.thptcurve, lam)

        self.assertEqual(thpt, thpt_check)

    def test_interp_thptcurve_between(self):
        """
        Verify correct linear interpolation when evaluating between array
        values.
        """
        i = 1
        lam = self.lams[i] + (self.lams[i+1] - self.lams[i]) / 2
        thpt_check = self.thpts[i] + (self.thpts[i+1] - self.thpts[i]) / 2

        thpt = _interp_thptcurve(self.thptcurve, lam)

        self.assertEqual(thpt, thpt_check)

    def test_interp_thptcurve_beyond(self):
        """
        Verify correct linear interpolation when evaluating beyond array
        values.
        """
        lam_less = self.lams[0] - 100
        with self.assertRaises(ThptToolsException):
            _interp_thptcurve(self.thptcurve, lam_less)

        lam_more = self.lams[-1] + 100
        with self.assertRaises(ThptToolsException):
            _interp_thptcurve(self.thptcurve, lam_more)

    def test_interp_thptcurve_invalid_bounds(self):
        """
        Verify invalid interpolated throughput fails as expected.
        """
        lam = self.lams[-1]
        with self.assertRaises(ThptToolsException):
            _interp_thptcurve(self.thptcurve_err, lam)

        lam = self.lams[0]
        with self.assertRaises(ThptToolsException):
            _interp_thptcurve(self.thptcurve_err, lam)


if __name__ == '__main__':
    unittest.main()
