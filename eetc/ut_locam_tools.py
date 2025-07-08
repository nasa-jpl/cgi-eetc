# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit test suite for locam_tools module.
"""
import unittest

import numpy as np

import eetc.util.ut_check as ut_check
from eetc.locam_tools import calc_locam_gain, LOCAMOptimizeException

class TestCalcLOCAMGain(unittest.TestCase):
    """
    Tests for calc_locam_gain function.
    """

    def setUp(self):
        self.fluxe_bright = 20/0.000441 # e-/pix/fr -> e-/s
        self.darke = 8.33e-4 # e-/sec
        self.cic = 0.02 # e-
        self.alpha0 = 0.70 # unitless
        self.fwc = 50000 # e-
        self.alpha1 = 0.70 # unitless
        self.fwc_em = 90000 # e-
        self.g_max_comm = 7500 # unitless
        self.g_max_age = 200 # unitless
        self.e_max_age = 12800 # e-
        self.tframe = 0.000441 # seconds
        self.n = 4 # unitless

        # also set up parameters that "turn off" things, to make it easier to
        # chase down branches by only activating one constraint
        self.fluxe_bright_disable = 0
        self.darke_disable = 0
        self.cic_disable = 0
        self.alpha1_disable = 1
        self.fwc_em_disable = np.iinfo(int).max
        self.g_max_comm_disable = 1e20
        self.g_max_age_disable = 1e20
        self.e_max_age_disable = np.iinfo(int).max
        self.n_disable = 0

        pass


    def test_success(self):
        """good inputs complete without exception"""
        calc_locam_gain(
            fluxe_bright=self.fluxe_bright,
            darke=self.darke,
            cic=self.cic,
            alpha0=self.alpha0,
            fwc=self.fwc,
            alpha1=self.alpha1,
            fwc_em=self.fwc_em,
            g_max_comm=self.g_max_comm,
            g_max_age=self.g_max_age,
            e_max_age=self.e_max_age,
            tframe=self.tframe,
            n=self.n,
        )
        pass


    def test_success_exact(self):
        """Verify we get the answer we expect for a nontrivial test"""
        tol = 1e-13

        e_max = 16384
        epersec = 100
        gain, _ = calc_locam_gain(
            fluxe_bright=epersec,
            darke=0,
            cic=0,
            alpha0=1.0,
            fwc=100000,
            alpha1=1.0,
            fwc_em=100000,
            g_max_comm=self.g_max_comm_disable,
            g_max_age=self.g_max_age_disable,
            e_max_age=e_max,
            tframe=1.0,
            n=0,
        )
        self.assertTrue(np.max(np.abs(gain-e_max/epersec)) < tol)
        pass


    def test_failure_too_bright(self):
        """
        Constraint 4 violation (too bright for a feasible solution) caught
        as expected
        """
        with self.assertRaises(LOCAMOptimizeException):
            calc_locam_gain(
                fluxe_bright=self.fluxe_bright*1e20,
                darke=self.darke,
                cic=self.cic,
                alpha0=self.alpha0,
                fwc=self.fwc,
                alpha1=self.alpha1,
                fwc_em=self.fwc_em,
                g_max_comm=self.g_max_comm,
                g_max_age=self.g_max_age,
                e_max_age=self.e_max_age,
                tframe=self.tframe,
                n=self.n,
            )
        pass

    def test_constraint_2_nonunity(self):
        """
        Verify that the function succeeds with code 2 if constraint 2 is
        driving
        """
        gain, code = calc_locam_gain(
            fluxe_bright=self.fluxe_bright,
            darke=self.darke,
            cic=self.cic,
            alpha0=self.alpha0,
            fwc=self.fwc,
            alpha1=self.alpha1_disable,
            fwc_em=self.fwc_em_disable,
            g_max_comm=self.g_max_comm,
            g_max_age=self.g_max_age_disable,
            e_max_age=self.e_max_age_disable,
            tframe=self.tframe,
            n=self.n,
        )
        self.assertTrue(gain == self.g_max_comm)
        self.assertTrue(code == 2)
        pass


    def test_constraint_2_and_3_tie(self):
        """
        Verify that the function succeeds with code 2 if constraint 2 and 3
        are both driving
        """
        gain, code = calc_locam_gain(
            fluxe_bright=self.fluxe_bright,
            darke=self.darke,
            cic=self.cic,
            alpha0=self.alpha0,
            fwc=self.fwc,
            alpha1=self.alpha1_disable,
            fwc_em=self.fwc_em_disable,
            g_max_comm=self.g_max_comm,
            g_max_age=self.g_max_comm, # same as 2
            e_max_age=self.e_max_age_disable,
            tframe=self.tframe,
            n=self.n,
        )
        self.assertTrue(gain == self.g_max_comm)
        self.assertTrue(code == 2) # smaller of the two
        pass



    def test_constraint_3_nonunity(self):
        """
        Verify that the function succeeds with code 3 if constraint 3 is
        driving
        """
        gain, code = calc_locam_gain(
            fluxe_bright=self.fluxe_bright,
            darke=self.darke,
            cic=self.cic,
            alpha0=self.alpha0,
            fwc=self.fwc,
            alpha1=self.alpha1_disable,
            fwc_em=self.fwc_em_disable,
            g_max_comm=self.g_max_comm_disable,
            g_max_age=self.g_max_age,
            e_max_age=self.e_max_age_disable,
            tframe=self.tframe,
            n=self.n,
        )
        self.assertTrue(gain == self.g_max_age)
        self.assertTrue(code == 3)
        pass


    def test_constraint_5_nonunity(self):
        """
        Verify that the function succeeds with code 5 if constraint 5 is
        driving
        """
        _, code = calc_locam_gain(
            fluxe_bright=self.fluxe_bright,
            darke=self.darke,
            cic=self.cic,
            alpha0=self.alpha0,
            fwc=self.fwc,
            alpha1=self.alpha1,
            fwc_em=self.fwc_em,
            g_max_comm=self.g_max_comm_disable,
            g_max_age=self.g_max_age_disable,
            e_max_age=self.e_max_age_disable,
            tframe=self.tframe,
            n=self.n,
        )
        self.assertTrue(code == 5)
        pass


    def test_constraint_6_nonunity(self):
        """
        Verify that the function succeeds with code 6 if constraint 6 is
        driving
        """
        _, code = calc_locam_gain(
            fluxe_bright=self.fluxe_bright,
            darke=self.darke,
            cic=self.cic,
            alpha0=self.alpha0,
            fwc=self.fwc,
            alpha1=self.alpha1_disable,
            fwc_em=self.fwc_em_disable,
            g_max_comm=self.g_max_comm_disable,
            g_max_age=self.g_max_age_disable,
            e_max_age=self.e_max_age,
            tframe=self.tframe,
            n=self.n,
        )
        self.assertTrue(code == 6)
        pass


    def test_success_noes(self):
        """Completes with code 2 or 3 if no electrons are present"""
        _, code = calc_locam_gain(
            fluxe_bright=self.fluxe_bright_disable,
            darke=self.darke_disable,
            cic=self.cic_disable,
            alpha0=self.alpha0,
            fwc=self.fwc,
            alpha1=self.alpha1,
            fwc_em=self.fwc_em,
            g_max_comm=self.g_max_comm,
            g_max_age=self.g_max_age,
            e_max_age=self.e_max_age,
            tframe=self.tframe,
            n=self.n_disable,
        )
        # when no electrons, 5 and 6 are disabled.  Only 2 and 3 left
        if self.g_max_age < self.g_max_comm:
            target = 3
        else:
            target = 2
            pass

        self.assertTrue(code == target)
        pass


    def test_gain_1_nonunity(self):
        """
        Completes with code != 1 and gain = 1 if the constraints only allow
        unity gain, but there are no aging considerations.
        """

        gain, code = calc_locam_gain(
            fluxe_bright=self.fluxe_bright,
            darke=self.darke,
            cic=self.cic,
            alpha0=self.alpha0,
            fwc=self.fwc,
            alpha1=self.alpha1,
            fwc_em=self.fwc_em,
            g_max_comm=1,
            g_max_age=self.g_max_age,
            e_max_age=self.e_max_age,
            tframe=self.tframe,
            n=self.n,
        )
        self.assertTrue(gain == 1)
        self.assertTrue(code != 1)
        pass


    def test_constraint_2_unity(self):
        """
        Completes with code 1 and gain = 1 if the constraints only allow unity
        gain, and there *are* aging considerations.  Constraint 2 case.
        """

        gain, code = calc_locam_gain(
            fluxe_bright=(self.e_max_age + 1000)/self.tframe,
            darke=self.darke,
            cic=self.cic,
            alpha0=self.alpha0,
            fwc=self.fwc,
            alpha1=self.alpha1,
            fwc_em=self.fwc_em,
            g_max_comm=1,
            g_max_age=self.g_max_age,
            e_max_age=self.e_max_age,
            tframe=self.tframe,
            n=self.n,
        )
        self.assertTrue(gain == 1)
        self.assertTrue(code == 1)
        pass


    def test_constraint_5_unity(self):
        """
        Completes with code 1 and gain = 1 if the constraints only allow unity
        gain, and there *are* aging considerations.  Constraint 5 case.
        """

        gain, code = calc_locam_gain(
            fluxe_bright=(self.e_max_age + 1000)/self.tframe,
            darke=self.darke,
            cic=self.cic,
            alpha0=self.alpha0,
            fwc=self.fwc,
            alpha1=self.alpha1,
            fwc_em=self.fwc_em,
            g_max_comm=self.g_max_comm_disable,
            g_max_age=self.g_max_age,
            e_max_age=self.e_max_age,
            tframe=self.tframe,
            n=self.n,
        )
        self.assertTrue(gain == 1)
        self.assertTrue(code == 1)
        pass


    def test_failure_infeasible(self):
        """
        Raises an exception if there is no feasible answer even at unity gain
        """
        # use constraint 5 to fail both (serial full well smaller than flux)
        # not realistic (serial FWC smaller than pixel FWC?!) but testable
        with self.assertRaises(LOCAMOptimizeException):
            calc_locam_gain(
                fluxe_bright=(self.e_max_age + 1000)/self.tframe,
                darke=self.darke,
                cic=self.cic,
                alpha0=self.alpha0,
                fwc=self.fwc,
                alpha1=1.0,
                fwc_em=self.e_max_age,
                g_max_comm=self.g_max_comm,
                g_max_age=self.g_max_age,
                e_max_age=self.e_max_age,
                tframe=self.tframe,
                n=self.n,
            )
        pass



    #------------------------
    # Input validation tests
    #------------------------

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""

        check_list = ut_check.rnslist

        # fluxe_bright
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_locam_gain(
                    fluxe_bright=perr,
                    darke=self.darke,
                    cic=self.cic,
                    alpha0=self.alpha0,
                    fwc=self.fwc,
                    alpha1=self.alpha1,
                    fwc_em=self.fwc_em,
                    g_max_comm=self.g_max_comm,
                    g_max_age=self.g_max_age,
                    e_max_age=self.e_max_age,
                    tframe=self.tframe,
                    n=self.n,
                )
            pass

        # darke
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_locam_gain(
                    fluxe_bright=self.fluxe_bright,
                    darke=perr,
                    cic=self.cic,
                    alpha0=self.alpha0,
                    fwc=self.fwc,
                    alpha1=self.alpha1,
                    fwc_em=self.fwc_em,
                    g_max_comm=self.g_max_comm,
                    g_max_age=self.g_max_age,
                    e_max_age=self.e_max_age,
                    tframe=self.tframe,
                    n=self.n,
                )
            pass

        # cic
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_locam_gain(
                    fluxe_bright=self.fluxe_bright,
                    darke=self.darke,
                    cic=perr,
                    alpha0=self.alpha0,
                    fwc=self.fwc,
                    alpha1=self.alpha1,
                    fwc_em=self.fwc_em,
                    g_max_comm=self.g_max_comm,
                    g_max_age=self.g_max_age,
                    e_max_age=self.e_max_age,
                    tframe=self.tframe,
                    n=self.n,
                )
            pass

        # tframe
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_locam_gain(
                    fluxe_bright=self.fluxe_bright,
                    darke=self.darke,
                    cic=self.cic,
                    alpha0=self.alpha0,
                    fwc=self.fwc,
                    alpha1=self.alpha1,
                    fwc_em=self.fwc_em,
                    g_max_comm=self.g_max_comm,
                    g_max_age=self.g_max_age,
                    e_max_age=self.e_max_age,
                    tframe=perr,
                    n=self.n,
                )
            pass

        # n
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_locam_gain(
                    fluxe_bright=self.fluxe_bright,
                    darke=self.darke,
                    cic=self.cic,
                    alpha0=self.alpha0,
                    fwc=self.fwc,
                    alpha1=self.alpha1,
                    fwc_em=self.fwc_em,
                    g_max_comm=self.g_max_comm,
                    g_max_age=self.g_max_age,
                    e_max_age=self.e_max_age,
                    tframe=self.tframe,
                    n=perr,
                )
            pass
        pass

    def test_invalid_postive_scalar_integer(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.psilist

        # fwc
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_locam_gain(
                    fluxe_bright=self.fluxe_bright,
                    darke=self.darke,
                    cic=self.cic,
                    alpha0=self.alpha0,
                    fwc=perr,
                    alpha1=self.alpha1,
                    fwc_em=self.fwc_em,
                    g_max_comm=self.g_max_comm,
                    g_max_age=self.g_max_age,
                    e_max_age=self.e_max_age,
                    tframe=self.tframe,
                    n=self.n,
                )
            pass

        # fwc_em
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_locam_gain(
                    fluxe_bright=self.fluxe_bright,
                    darke=self.darke,
                    cic=self.cic,
                    alpha0=self.alpha0,
                    fwc=self.fwc,
                    alpha1=self.alpha1,
                    fwc_em=perr,
                    g_max_comm=self.g_max_comm,
                    g_max_age=self.g_max_age,
                    e_max_age=self.e_max_age,
                    tframe=self.tframe,
                    n=self.n,
                )
            pass

        # e_max_age
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_locam_gain(
                    fluxe_bright=self.fluxe_bright,
                    darke=self.darke,
                    cic=self.cic,
                    alpha0=self.alpha0,
                    fwc=self.fwc,
                    alpha1=self.alpha1,
                    fwc_em=self.fwc_em,
                    g_max_comm=self.g_max_comm,
                    g_max_age=self.g_max_age,
                    e_max_age=perr,
                    tframe=self.tframe,
                    n=self.n,
                )
            pass
        pass

    def test_invalid_real_positive_scalar(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.rpslist

        # alpha0
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_locam_gain(
                    fluxe_bright=self.fluxe_bright,
                    darke=self.darke,
                    cic=self.cic,
                    alpha0=perr,
                    fwc=self.fwc,
                    alpha1=self.alpha1,
                    fwc_em=self.fwc_em,
                    g_max_comm=self.g_max_comm,
                    g_max_age=self.g_max_age,
                    e_max_age=self.e_max_age,
                    tframe=self.tframe,
                    n=self.n,
                )
            pass

        # alpha1
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_locam_gain(
                    fluxe_bright=self.fluxe_bright,
                    darke=self.darke,
                    cic=self.cic,
                    alpha0=self.alpha0,
                    fwc=self.fwc,
                    alpha1=perr,
                    fwc_em=self.fwc_em,
                    g_max_comm=self.g_max_comm,
                    g_max_age=self.g_max_age,
                    e_max_age=self.e_max_age,
                    tframe=self.tframe,
                    n=self.n,
                )
            pass

        # g_max_comm
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_locam_gain(
                    fluxe_bright=self.fluxe_bright,
                    darke=self.darke,
                    cic=self.cic,
                    alpha0=self.alpha0,
                    fwc=self.fwc,
                    alpha1=self.alpha1,
                    fwc_em=self.fwc_em,
                    g_max_comm=perr,
                    g_max_age=self.g_max_age,
                    e_max_age=self.e_max_age,
                    tframe=self.tframe,
                    n=self.n,
                )
            pass

        # g_max_age
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_locam_gain(
                    fluxe_bright=self.fluxe_bright,
                    darke=self.darke,
                    cic=self.cic,
                    alpha0=self.alpha0,
                    fwc=self.fwc,
                    alpha1=self.alpha1,
                    fwc_em=self.fwc_em,
                    g_max_comm=self.g_max_comm,
                    g_max_age=perr,
                    e_max_age=self.e_max_age,
                    tframe=self.tframe,
                    n=self.n,
                )
            pass

    def test_invalid_alpha_gt_1(self):
        """Invalid inputs fail as expected"""
        perr = 1.1
        with self.assertRaises(ValueError):
            calc_locam_gain(
                fluxe_bright=self.fluxe_bright,
                darke=self.darke,
                cic=self.cic,
                alpha0=perr,
                fwc=self.fwc,
                alpha1=self.alpha1,
                fwc_em=self.fwc_em,
                g_max_comm=self.g_max_comm,
                g_max_age=self.g_max_age,
                e_max_age=self.e_max_age,
                tframe=self.tframe,
                n=self.n,
            )

        perr = 1.1
        with self.assertRaises(ValueError):
            calc_locam_gain(
                fluxe_bright=self.fluxe_bright,
                darke=self.darke,
                cic=self.cic,
                alpha0=self.alpha0,
                fwc=self.fwc,
                alpha1=perr,
                fwc_em=self.fwc_em,
                g_max_comm=self.g_max_comm,
                g_max_age=self.g_max_age,
                e_max_age=self.e_max_age,
                tframe=self.tframe,
                n=self.n,
            )
        pass


    def test_invalid_gmax_lt_1(self):
        """Invalid inputs fail as expected"""
        perr = 0.9
        # g_max_comm
        with self.assertRaises(ValueError):
            calc_locam_gain(
                fluxe_bright=self.fluxe_bright,
                darke=self.darke,
                cic=self.cic,
                alpha0=self.alpha0,
                fwc=self.fwc,
                alpha1=self.alpha1,
                fwc_em=self.fwc_em,
                g_max_comm=perr,
                g_max_age=self.g_max_age,
                e_max_age=self.e_max_age,
                tframe=self.tframe,
                n=self.n,
            )

        # g_max_age
        with self.assertRaises(ValueError):
            calc_locam_gain(
                fluxe_bright=self.fluxe_bright,
                darke=self.darke,
                cic=self.cic,
                alpha0=self.alpha0,
                fwc=self.fwc,
                alpha1=self.alpha1,
                fwc_em=self.fwc_em,
                g_max_comm=self.g_max_comm,
                g_max_age=perr,
                e_max_age=self.e_max_age,
                tframe=self.tframe,
                n=self.n,
            )
        pass


if __name__ == '__main__':
    unittest.main()
