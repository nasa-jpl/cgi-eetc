# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit test suite for load module.
"""
import unittest

from eetc.tools import get_signal_ratio, get_effective_wavelength
from eetc.util.ut_check import strlist, rslist
from eetc.cgi_eetc import CGIEETC

class TestGetSignalRatio(unittest.TestCase):
    """
    Unit tests for get_signal_ratio function.
    """

    def setUp(self):
        self.mag1 = 2.25
        self.spt1 = 'B2V'
        self.gain1 = 1
        self.mag2 = 5.0
        self.spt2 = 'G2V'
        self.gain2 = 10
        self.sequence_name = 'LOCAM_NFOV_DM'


    def test_success(self):
        """Given valid inputs, completes without error"""
        get_signal_ratio(
            mag1=self.mag1,
            spt1=self.spt1,
            gain1=self.gain1,
            mag2=self.mag2,
            spt2=self.spt2,
            gain2=self.gain2,
            sequence_name=self.sequence_name,
        )
        pass


    def test_ratio_direction(self):
        """Verify output ratio is star 2/star 1"""
        s0 = get_signal_ratio(
            mag1=self.mag1,
            spt1=self.spt1,
            gain1=self.gain1,
            mag2=self.mag2,
            spt2=self.spt2,
            gain2=self.gain2,
            sequence_name=self.sequence_name,
        )

        # gain lower on star 2
        s1 = get_signal_ratio(
            mag1=self.mag1,
            spt1=self.spt1,
            gain1=self.gain1,
            mag2=self.mag2,
            spt2=self.spt2,
            gain2=self.gain2*0.5,
            sequence_name=self.sequence_name,
        )

        # flux lower on star 2
        s2 = get_signal_ratio(
            mag1=self.mag1,
            spt1=self.spt1,
            gain1=self.gain1,
            mag2=self.mag2 + 2.0,
            spt2=self.spt2,
            gain2=self.gain2,
            sequence_name=self.sequence_name,
        )

        # gain higher on star 1
        s3 = get_signal_ratio(
            mag1=self.mag1,
            spt1=self.spt1,
            gain1=self.gain1*2.0,
            mag2=self.mag2,
            spt2=self.spt2,
            gain2=self.gain2,
            sequence_name=self.sequence_name,
        )

        # flux higher on star 1
        s4 = get_signal_ratio(
            mag1=self.mag1 - 2.0,
            spt1=self.spt1,
            gain1=self.gain1,
            mag2=self.mag2,
            spt2=self.spt2,
            gain2=self.gain2,
            sequence_name=self.sequence_name,
        )

        self.assertTrue(s0 > s1)
        self.assertTrue(s0 > s2)
        self.assertTrue(s0 > s3)
        self.assertTrue(s0 > s4)

        pass


    def test_exact_result_same_star_different_gains(self):
        """Verify numerical correctness when changing the gain on a star"""
        tol = 1e-13

        s0 = get_signal_ratio(
            mag1=self.mag1,
            spt1=self.spt1,
            gain1=self.gain1,
            mag2=self.mag2,
            spt2=self.spt2,
            gain2=self.gain2,
            sequence_name=self.sequence_name,
        )

        s1 = get_signal_ratio(
            mag1=self.mag1,
            spt1=self.spt1,
            gain1=self.gain1*2.0,
            mag2=self.mag2,
            spt2=self.spt2,
            gain2=self.gain2,
            sequence_name=self.sequence_name,
        )

        s2 = get_signal_ratio(
            mag1=self.mag1,
            spt1=self.spt1,
            gain1=self.gain1,
            mag2=self.mag2,
            spt2=self.spt2,
            gain2=self.gain2*0.5,
            sequence_name=self.sequence_name,
        )

        self.assertTrue(abs(0.5*s0 - s1) < tol)
        self.assertTrue(abs(0.5*s0 - s2) < tol)

        pass


    def test_exact_flux_down_gain_up(self):
        """Verify exact answer if flux and gain are changed together"""

        tol = 1e-13

        s0 = get_signal_ratio(
            mag1=self.mag1,
            spt1=self.spt1,
            gain1=self.gain1,
            mag2=self.mag2,
            spt2=self.spt2,
            gain2=self.gain2,
            sequence_name=self.sequence_name,
        )

        # delta-mag of 5 = 100x when not in log anymore
        s1 = get_signal_ratio(
            mag1=self.mag1 + 5.0,
            spt1=self.spt1,
            gain1=self.gain1*100,
            mag2=self.mag2,
            spt2=self.spt2,
            gain2=self.gain2,
            sequence_name=self.sequence_name,
        )

        s2 = get_signal_ratio(
            mag1=self.mag1,
            spt1=self.spt1,
            gain1=self.gain1,
            mag2=self.mag2 + 5.0,
            spt2=self.spt2,
            gain2=self.gain2*100,
            sequence_name=self.sequence_name,
        )

        self.assertTrue(abs(s1-s0) < tol)
        self.assertTrue(abs(s2-s0) < tol)

        pass


    def test_unity_self_consistency(self):
        """Running a signal ratio on the same star should give output = 1"""
        tol = 1e-13

        # use star 1 for both
        s0 = get_signal_ratio(
            mag1=self.mag1,
            spt1=self.spt1,
            gain1=self.gain1,
            mag2=self.mag1,
            spt2=self.spt1,
            gain2=self.gain1,
            sequence_name=self.sequence_name,
        )

        self.assertTrue(abs(s0 - 1.0) < tol)

        pass



    def test_invalid_mag1(self):
        """Verify invalid inputs fail as expected."""
        for perr in rslist:
            with self.assertRaises(TypeError):
                get_signal_ratio(
                    mag1=perr,
                    spt1=self.spt1,
                    gain1=self.gain1,
                    mag2=self.mag2,
                    spt2=self.spt2,
                    gain2=self.gain2,
                    sequence_name=self.sequence_name,
                )
            pass
        pass


    def test_invalid_spt1(self):
        """Verify invalid inputs fail as expected."""
        for perr in strlist:
            with self.assertRaises(TypeError):
                get_signal_ratio(
                    mag1=self.mag1,
                    spt1=perr,
                    gain1=self.gain1,
                    mag2=self.mag2,
                    spt2=self.spt2,
                    gain2=self.gain2,
                    sequence_name=self.sequence_name,
                )
            pass

        for perr in ['invalid_spt']:
            with self.assertRaises(KeyError):
                get_signal_ratio(
                    mag1=self.mag1,
                    spt1=perr,
                    gain1=self.gain1,
                    mag2=self.mag2,
                    spt2=self.spt2,
                    gain2=self.gain2,
                    sequence_name=self.sequence_name,
                )
            pass

        pass


    def test_invalid_gain1(self):
        """Verify invalid inputs fail as expected."""
        for perr in rslist:
            with self.assertRaises(TypeError):
                get_signal_ratio(
                    mag1=self.mag1,
                    spt1=self.spt1,
                    gain1=perr,
                    mag2=self.mag2,
                    spt2=self.spt2,
                    gain2=self.gain2,
                    sequence_name=self.sequence_name,
                )
            pass

        for perr in [0, 0.99]:
            with self.assertRaises(ValueError):
                get_signal_ratio(
                    mag1=self.mag1,
                    spt1=self.spt1,
                    gain1=perr,
                    mag2=self.mag2,
                    spt2=self.spt2,
                    gain2=self.gain2,
                    sequence_name=self.sequence_name,
                )
            pass

        pass



    def test_invalid_mag2(self):
        """Verify invalid inputs fail as expected."""
        for perr in rslist:
            with self.assertRaises(TypeError):
                get_signal_ratio(
                    mag1=self.mag1,
                    spt1=self.spt1,
                    gain1=self.gain1,
                    mag2=perr,
                    spt2=self.spt2,
                    gain2=self.gain2,
                    sequence_name=self.sequence_name,
                )
            pass
        pass


    def test_invalid_spt2(self):
        """Verify invalid inputs fail as expected."""
        for perr in strlist:
            with self.assertRaises(TypeError):
                get_signal_ratio(
                    mag1=self.mag1,
                    spt1=self.spt1,
                    gain1=self.gain1,
                    mag2=self.mag2,
                    spt2=perr,
                    gain2=self.gain2,
                    sequence_name=self.sequence_name,
                )
            pass

        for perr in ['invalid_spt']:
            with self.assertRaises(KeyError):
                get_signal_ratio(
                    mag1=self.mag1,
                    spt1=self.spt1,
                    gain1=self.gain1,
                    mag2=self.mag2,
                    spt2=perr,
                    gain2=self.gain2,
                    sequence_name=self.sequence_name,
                )
            pass

        pass


    def test_invalid_gain2(self):
        """Verify invalid inputs fail as expected."""
        for perr in rslist:
            with self.assertRaises(TypeError):
                get_signal_ratio(
                    mag1=self.mag1,
                    spt1=self.spt1,
                    gain1=self.gain1,
                    mag2=self.mag2,
                    spt2=self.spt2,
                    gain2=perr,
                    sequence_name=self.sequence_name,
                )
            pass

        for perr in [0, 0.99]:
            with self.assertRaises(ValueError):
                get_signal_ratio(
                    mag1=self.mag1,
                    spt1=self.spt1,
                    gain1=self.gain1,
                    mag2=self.mag2,
                    spt2=self.spt2,
                    gain2=perr,
                    sequence_name=self.sequence_name,
                )
            pass

        pass


    def test_invalid_sequence_name(self):
        """Verify invalid inputs fail as expected."""
        for perr in strlist:
            with self.assertRaises(TypeError):
                get_signal_ratio(
                    mag1=self.mag1,
                    spt1=self.spt1,
                    gain1=self.gain1,
                    mag2=self.mag2,
                    spt2=self.spt2,
                    gain2=self.gain2,
                    sequence_name=perr,
                )
            pass

        for perr in ['invalid_sequence_name']:
            with self.assertRaises(ValueError):
                get_signal_ratio(
                    mag1=self.mag1,
                    spt1=self.spt1,
                    gain1=self.gain1,
                    mag2=self.mag2,
                    spt2=self.spt2,
                    gain2=self.gain2,
                    sequence_name=perr,
                )
            pass

        pass


    def test_invalid_pointer_path(self):
        """Verify invalid inputs fail as expected."""
        for perr in strlist:
            with self.assertRaises(TypeError):
                get_signal_ratio(
                    mag1=self.mag1,
                    spt1=self.spt1,
                    gain1=self.gain1,
                    mag2=self.mag2,
                    spt2=self.spt2,
                    gain2=self.gain2,
                    sequence_name=self.sequence_name,
                    pointer_path=perr,
                )
            pass

        for perr in ['invalid_pointer_path']:
            with self.assertRaises(ValueError):
                get_signal_ratio(
                    mag1=self.mag1,
                    spt1=self.spt1,
                    gain1=self.gain1,
                    mag2=self.mag2,
                    spt2=self.spt2,
                    gain2=self.gain2,
                    sequence_name=self.sequence_name,
                    pointer_path=perr,
                )
            pass

        pass


class TestGetEffectiveWavelength(unittest.TestCase):
    """
    Unit tests for get_effective_wavelength function.

    x success
    x invalid inputs
    cycle through all valid cfam and spt options
    """

    def setUp(self):
        self.spt = 'B2V'
        self.cfam = ['1B']
        pass


    def test_success(self):
        """Given valid inputs, completes without error"""
        get_effective_wavelength(
            cfam=self.cfam,
            spt=self.spt,
        )
        pass


    def test_exercise_all_possible_values(self):
        """
        cfam and spt are drawn for finite lists.  cycle through all pairs of
        inputs from both lists and verify that they all complete without
        error
        """
        # dummy instance
        tmp_eetc = CGIEETC(mag=0, phot='v', spt='g2v')
        for spt in tmp_eetc.wave_grid_spts:
            get_effective_wavelength(
                cfam=list(tmp_eetc.eff_cfams.keys()),
                spt=spt,
            )
            pass
        pass


    def test_invalid_cfam(self):
        """Verify invalid inputs fail as expected."""
        for perr in strlist:
            with self.assertRaises(TypeError):
                get_effective_wavelength(
                    cfam=[perr],
                    spt=self.spt,
                )
            pass

        for perr in [['invalid_cfam']]:
            with self.assertRaises(ValueError):
                get_effective_wavelength(
                    cfam=perr,
                    spt=self.spt,
                )
            pass

        for perr in ['1B']: # not list
            with self.assertRaises(TypeError):
                get_effective_wavelength(
                    cfam=perr,
                    spt=self.spt,
                )
            pass

        pass


    def test_invalid_spt(self):
        """Verify invalid inputs fail as expected."""
        for perr in strlist:
            with self.assertRaises(TypeError):
                get_effective_wavelength(
                    cfam=self.cfam,
                    spt=perr,
                )
            pass

        for perr in ['invalid_spt']:
            with self.assertRaises(KeyError):
                get_effective_wavelength(
                    cfam=self.cfam,
                    spt=perr,
                )
            pass

        pass


    def test_invalid_pointer_path(self):
        """Verify invalid inputs fail as expected."""
        for perr in strlist:
            with self.assertRaises(TypeError):
                get_effective_wavelength(
                    cfam=self.cfam,
                    spt=self.spt,
                    pointer_path=perr,
                )
            pass

        for perr in ['invalid_pointer_path']:
            with self.assertRaises(ValueError):
                get_effective_wavelength(
                    cfam=self.cfam,
                    spt=self.spt,
                    pointer_path=perr,
                )
            pass

        pass





if __name__ == '__main__':
    unittest.main()
