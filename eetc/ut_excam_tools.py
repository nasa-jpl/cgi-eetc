# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit test suite for excam_tools module.
"""
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import scipy.optimize

import eetc.util.ut_check as ut_check
from eetc.excam_tools import (calc_gain_exptime, calc_gain_fixed_time,
                              calc_gain_fixed_N, calc_gain_fixed_g,
                              calc_gain_fixed_Ntime, calc_pc, calc_pc_fixed_N,
                              calc_pc_gain_fixed_Ntime,
                              EXCAMOptimizeException)

class TestCalcGainExptime(unittest.TestCase):
    """
    Tests for calc_gain_exptime function.
    """

    def setUp(self):
        self.target_snr = 7 # unitless
        self.fluxe = 10 # e-/sec
        self.fluxe_bright = 100 # e-sec
        self.darke = 8.33e-4 # e-/sec
        self.cic = 0.02 # e-
        self.rn = 160 # e-
        self.X = 5e4 # hits/m^2/sec
        self.a = 1.69e-10 # m^2/pixel
        self.Lij = 512 # pixels
        self.alpha0 = 0.75 # unitless
        self.fwc = 50000 # e-
        self.alpha1 = 0.75 # unitless
        self.fwc_em = 90000 # e-
        self.Nmin = 1 # frames
        self.Nmax = 49 # frames
        self.tmin = 0.264 # seconds/frame
        self.tmax = 120. # seconds/frame
        self.gmax = 5000 # unitless
        self.overhead = 0 # seconds
        self.opt_choice = 0 # 0 or 1
        self.n = 4 # number of standard deviations below fwc
        self.Nem = 604 # number of gain register cells
        self.tol = 1e-30 # tolerance level used by optimizations
        self.delta_constr = 1e-4 # constraints satisfied up to this fraction

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """good inputs for opt_choice = 0 complete without an exception"""
        calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead)
        pass

    def test_opt_choice_success(self):
        """good inputs for opt_choice=1 complete without exception"""
        calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead, opt_choice=1)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_first_both(self, mock_min):
        """good inputs for opt_choice = 0 complete without an exception.
        res1_cond and res2_cond both true.  Ends after first optimization
         with res2 results since it has the shorter total exposure time."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools, and they definitely
        # satisfy rail, em_rail, and target_snr constraints in the optimization
        _g1 = 1
        _t1 = self.tmin
        _N1 = self.Nmin+1
        _N2 = self.Nmin  # so N2*tfr2 will be better

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g1, _t1, _N2])
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t1, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_exptime(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead)
        self.assertTrue(out[-1] == 0)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N2)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_first_one(self, mock_min):
        """good inputs for opt_choice = 0 complete without an exception.
        res1_cond true while res2_cond false.  Ends after first optimization
         with res1 results."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools, and they definitely
        # satisfy rail, em_rail, and target_snr constraints in the optimization
        _g1 = 1
        _t1 = self.tmin
        _N1 = self.Nmin+1
        _g2 = 1e30 # ensures that res2_cond is false
        _t2 = self.tmin
        _N2 = self.Nmin  # would give a shorter N2*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _t2, _N2])
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t1, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_exptime(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead)
        self.assertTrue(out[-1] == 0)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N1)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_second_both(self, mock_min):
        """bad inputs for opt_choice = 0.
        Ends after second optimization due to constraint disagreement in first
        optimization scheme.  res3_cond and res4_cond both true.  Gives res4
        outputs due to bigger SNR."""

        _g1 = 1
        _t1 = self.tmin
        _N1 = self.Nmin

        # constraints not met (can't reach target SNR of 1e15)
        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g1, _t1, _N1])

        #now for 2nd optimzation scheme, let the constraints be met
        _N4 = self.Nmin+1  # gives bigger snr since snr proportional to sqrt(N)
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t1, _N4])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_exptime(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N4)
        pass


    @patch('scipy.optimize.minimize')
    def test_end_after_second_one(self, mock_min):
        """bad inputs for opt_choice = 0.
        Ends after 2nd optimization due to constraint disagreement in first
        optimization scheme.  res3_cond true while res4_cond false.  Gives res3
        results."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools
        _g1 = 1e30 # res1_cond false
        _t1 = self.tmin
        _N1 = self.Nmin+1
        _g2 = 1e30 # ensures that res2_cond is false
        _t2 = self.tmin
        _N2 = self.Nmin  # would give a shorter N2*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _t2, _N2])

        _g3 = 1
        _g4 = 1e30 # breaks em_rail constraint, even though it would give
        #bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _t1, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_exptime(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N1)
        pass


    @patch('scipy.optimize.minimize')
    def test_both_fail(self, mock_min):
        """For opt_choice = 0, the 2nd optimization scheme fails due to
        constraint disagreement, and an exception is raised."""

         # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools
        _g1 = 1e30 # res1_cond false
        _t1 = self.tmin
        _N1 = self.Nmin+1
        _g2 = 1e30 # ensures that res2_cond is false
        _t2 = self.tmin
        _N2 = self.Nmin  # would give a shorter N2*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _t2, _N2])

        #now for 2nd optimzation scheme, let the constraints disagree
        _g3 = 1e30  # breaks em_rail constraint
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _t1, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_both(self, mock_min):
        """For opt_choice = 1, both res3_cond and res4_cond are true.  Gives
        res4 outputs due to bigger SNR."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = 1
        _t3 = self.tmin
        _N3 = self.Nmin
        _N4 = _N3+1  # gives the bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t3, _N3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _t3, _N4])

        mock_min.side_effect = [_res3, _res4, _res3]  # should stop after
                                                      # second mock

        # target_snr should have no bearing when opt_choice=1
        out = calc_gain_exptime(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead, opt_choice=1)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t3)
        self.assertTrue(out[2] == _N4)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_one(self, mock_min):
        """For opt_choice = 1, res3_cond is true and res4_cond is false.  Gives
        res3 outputs."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = 1
        _t3 = self.tmin
        _N3 = self.Nmin
        _g4 = 1e30 # breaks em_rail condition, even though gives bigger snr
        _N4 = _N3+1  # would give the bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t3, _N3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _t3, _N4])

        mock_min.side_effect = [_res3, _res4, _res4]  # should stop after
                                                      # second mock

        # target_snr should have no bearing when opt_choice=1
        out = calc_gain_exptime(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead, opt_choice=1)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t3)
        self.assertTrue(out[2] == _N3)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_both_fail(self, mock_min):
        """For opt_choice = 1, the 2nd optimization scheme fails due to
        constraint disagreement, and an exception is raised."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = 1e30  # break em_rail condition
        _t3 = self.tmin
        _N3 = self.Nmin
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t3, _N3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _t3, _N3])

        mock_min.side_effect = [_res3, _res4, _res4]  # should stop after
                                                      # second mock

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead, opt_choice=1)
        pass

    def test_successful_run_multiple_pixels(self):
        '''Successful run for values of num_pixels other than the default.
        Other checks comparing multi-pixel SNR with single-pixel SNR done in
        ut_cgi_eetc.py.'''
        values = [15, 0.5, 3.3] # non-integer values should work, too
        for val in values:
            calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead, num_pixels=val)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""

        check_list = ut_check.rnslist

        # target_snr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=perr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fluxe
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=perr,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fluxe_bright
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=perr,
                                  fluxe_bright=perr,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # darke
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright, darke=perr,
                                  cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                                  Lij=self.Lij, alpha0=self.alpha0,
                                  fwc=self.fwc, alpha1=self.alpha1,
                                  fwc_em=self.fwc_em, Nmin=self.Nmin,
                                  Nmax=self.Nmax, tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead)
            pass

        # cic
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=perr, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # rn
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=perr,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # X
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=perr, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # a
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=perr, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # tmin
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax, tmin=perr,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead)
            pass

        # overhead
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=perr)
            pass


        # n
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  n=perr)
            pass

        # tol
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  tol=perr)
            pass

        # delta_constr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  delta_constr=perr)
            pass
        pass

        # num_pixels
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  num_pixels=perr)
            pass
        pass

    def test_invalid_postive_scalar_integer(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.psilist

        # Lij
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=perr,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fwc
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=perr,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fwc_em
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=perr,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # Nmin
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=perr, Nmax=self.Nmax, tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead)
            pass

        # Nmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=perr, tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead)
            pass

        # Nem
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead,
                                  Nem=perr)
            pass
        pass

    def test_invalid_real_positive_scalar(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.rpslist

        # alpha0
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=perr, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # alpha1
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=perr, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # tmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=perr, gmax=self.gmax,
                                  overhead=self.overhead)
            pass

        # gmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=perr, overhead=self.overhead)
            pass

    def test_invalid_nonnegative_scalar_integer(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.nsilist

        # opt_choice
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  opt_choice=perr)
            pass

    def test_invalid_fluxe_bright(self):
        """Invalid inputs fail as expected"""
        perr = self.fluxe_bright*1.1
        with self.assertRaises(ValueError):
            calc_gain_exptime(target_snr=self.target_snr, fluxe=perr,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead)

    def test_invalid_alpha_gt_1(self):
        """Invalid inputs fail as expected"""
        perr = 1.1
        with self.assertRaises(ValueError):
            calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=perr, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead)

        perr = 1.1
        with self.assertRaises(ValueError):
            calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=perr, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead)

    def test_invalid_nmax_lt_nmin(self):
        """Invalid inputs caught"""
        nmax = 10
        for nmin in [20, 10]:

            with self.assertRaises(ValueError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=nmin, Nmax=nmax, tmin=self.tmin,
                            tmax=self.tmax, gmax=self.gmax,
                            overhead=self.overhead)
        pass

    def test_invalid_tmax_lt_tmin(self):
        """Invalid inputs caught"""
        tmax = 10
        for tmin in [10, 20]:

            with self.assertRaises(ValueError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                            tmax=tmax, gmax=self.gmax, overhead=self.overhead)
        pass

    def test_invalid_gain(self):
        """Invalid inputs caught"""

        for perr in [0.9, 1]:
            with self.assertRaises(ValueError):
                calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                tmin=self.tmin, tmax=self.tmax, gmax=perr,
                                overhead=self.overhead)
                pass
        pass

    def test_invalid_opt_choice(self):
        """Invalid inputs caught"""
        perr = 2

        with self.assertRaises(ValueError):
            calc_gain_exptime(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              opt_choice=perr)
            pass

    def test_invalid_full_wells(self):
        """Check the two full well cases raise expected exceptions"""
        fluxe = 5
        fluxe_bright = 10
        darke = 10
        cic = 100
        tmin = 10
        alpha0 = 1
        alpha1 = 1
        fwc = 10
        fwc_em = 10

        # per-pixel
        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_exptime(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=fluxe_bright, darke=darke,
                              cic=cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=alpha0, fwc=fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead)

        # gain register
        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_exptime(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=fluxe_bright, darke=darke,
                              cic=cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=alpha1, fwc_em=fwc_em, Nmin=self.Nmin,
                              Nmax=self.Nmax, tmin=tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead)

        pass

class TestCalcGainFixedTime(unittest.TestCase):
    """
    Tests for calc_gain_fixed_time function.
    """

    def setUp(self):
        self.target_snr = 3 # unitless
        self.fluxe = 10 # e-/sec
        self.fluxe_bright = 100 # e-sec
        self.darke = 8.33e-4 # e-/sec
        self.cic = 0.02 # e-
        self.rn = 160 # e-
        self.X = 5e4 # hits/m^2/sec
        self.a = 1.69e-10 # m^2/pixel
        self.Lij = 512 # pixels
        self.alpha0 = 0.75 # unitless
        self.fwc = 50000 # e-
        self.alpha1 = 0.75 # unitless
        self.fwc_em = 90000 # e-
        self.Nmin = 1 # frames
        self.Nmax = 49 # frames
        self.t = 10 # s
        self.gmax = 5000 # unitless
        self.overhead = 0 # seconds
        self.opt_choice = 0 # 0 or 1
        self.n = 4 # number of standard deviations below fwc
        self.Nem = 604 # number of gain register cells
        self.tol = 1e-30 # tolerance level used by optimizations
        self.delta_constr = 1e-4 # constraints satisfied up to this fraction

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """good inputs for opt_choice = 0 complete without an exception"""
        calc_gain_fixed_time(target_snr=self.target_snr, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, t=self.t,
                          gmax=self.gmax, overhead=self.overhead)
        pass

    def test_opt_choice_success(self):
        """good inputs for opt_choice=1 complete without exception"""
        calc_gain_fixed_time(target_snr=self.target_snr, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, t=self.t,
                          gmax=self.gmax, overhead=self.overhead, opt_choice=1)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_first_both(self, mock_min):
        """good inputs for opt_choice = 0 complete without an exception.
        res1_cond and res2_cond both true.  Ends after first optimization
         with res2 results since it has the shorter total exposure time."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools, and they definitely
        # satisfy rail, em_rail, and target_snr constraints in the optimization
        _g1 = 1
        _t1 = 0.264 # the usual "minimum" time, to be safe
        _N1 = self.Nmin+1
        _N2 = self.Nmin  # so N2*tfr2 will be better

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g1, _N2])
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_fixed_time(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, t=_t1,
                                gmax=self.gmax, overhead=self.overhead)
        self.assertTrue(out[-1] == 0)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N2)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_first_one(self, mock_min):
        """good inputs for opt_choice = 0 complete without an exception.
        res1_cond true while res2_cond false.  Ends after first optimization
         with res1 results."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools, and they definitely
        # satisfy rail, em_rail, and target_snr constraints in the optimization
        _g1 = 1
        _t1 = 0.264
        _N1 = self.Nmin+1
        _g2 = 1e30 # ensures that res2_cond is false
        _N2 = self.Nmin  # would give a shorter N2*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _N2])
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_fixed_time(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, t=_t1,
                                gmax=self.gmax, overhead=self.overhead)
        self.assertTrue(out[-1] == 0)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N1)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_second_both(self, mock_min):
        """bad inputs for opt_choice = 0.
        Ends after second optimization due to constraint disagreement in first
        optimization scheme.  res3_cond and res4_cond both true.  Gives res4
        outputs due to bigger SNR."""

        _g1 = 1
        _t1 = 0.264
        _N1 = self.Nmin

        # constraints not met (can't reach target SNR of 1e15)
        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g1, _N1])

        #now for 2nd optimzation scheme, let the constraints be met
        _N4 = self.Nmin+1  # gives bigger snr since snr proportional to sqrt(N)
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _N4])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_fixed_time(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, t=_t1,
                                gmax=self.gmax, overhead=self.overhead)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N4)
        pass


    @patch('scipy.optimize.minimize')
    def test_end_after_second_one(self, mock_min):
        """bad inputs for opt_choice = 0.
        Ends after 2nd optimization due to constraint disagreement in first
        optimization scheme.  res3_cond true while res4_cond false.  Gives res3
        results."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools
        _g1 = 1e30 # res1_cond false
        _t1 = 0.264
        _N1 = self.Nmin+1
        _g2 = 1e30 # ensures that res2_cond is false
        _N2 = self.Nmin  # would give a shorter N2*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _N2])

        _g3 = 1
        _g4 = 1e30 # breaks em_rail constraint, even though it would give
        #bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_fixed_time(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, t=_t1,
                                gmax=self.gmax, overhead=self.overhead)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N1)
        pass


    @patch('scipy.optimize.minimize')
    def test_both_fail(self, mock_min):
        """For opt_choice = 0, the 2nd optimization scheme fails due to
        constraint disagreement, and an exception is raised."""

         # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools
        _g1 = 1e30 # res1_cond false
        _t1 = 0.264
        _N1 = self.Nmin+1
        _g2 = 1e30 # ensures that res2_cond is false
        _N2 = self.Nmin  # would give a shorter N2*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _N2])

        #now for 2nd optimzation scheme, let the constraints disagree
        _g3 = 1e30  # breaks em_rail constraint
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_time(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, t=_t1,
                              gmax=self.gmax, overhead=self.overhead)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_both(self, mock_min):
        """For opt_choice = 1, both res3_cond and res4_cond are true.  Gives
        res4 outputs due to bigger SNR."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = 1
        _t3 = 0.264
        _N3 = self.Nmin
        _N4 = _N3+1  # gives the bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _N3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _N4])

        mock_min.side_effect = [_res3, _res4, _res3]  # should stop after
                                                      # second mock

        # target_snr should have no bearing when opt_choice=1
        out = calc_gain_fixed_time(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, t=_t3,
                                gmax=self.gmax, overhead=self.overhead,
                                opt_choice=1)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t3)
        self.assertTrue(out[2] == _N4)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_one(self, mock_min):
        """For opt_choice = 1, res3_cond is true and res4_cond is false.  Gives
        res3 outputs."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = 1
        _t3 = 0.264
        _N3 = self.Nmin
        _g4 = 1e30 # breaks em_rail condition, even though gives bigger snr
        _N4 = _N3+1  # would give the bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _N3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _N4])

        mock_min.side_effect = [_res3, _res4, _res4]  # should stop after
                                                      # second mock

        # target_snr should have no bearing when opt_choice=1
        out = calc_gain_fixed_time(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, t=_t3,
                                gmax=self.gmax, overhead=self.overhead,
                                opt_choice=1)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t3)
        self.assertTrue(out[2] == _N3)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_both_fail(self, mock_min):
        """For opt_choice = 1, the 2nd optimization scheme fails due to
        constraint disagreement, and an exception is raised."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = 1e30  # break em_rail condition
        _t3 = 0.264
        _N3 = self.Nmin
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _N3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _N3])

        mock_min.side_effect = [_res3, _res4, _res4]  # should stop after
                                                      # second mock

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_time(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, t=_t3,
                              gmax=self.gmax, overhead=self.overhead,
                              opt_choice=1)
        pass

    def test_successful_run_multiple_pixels(self):
        '''Successful run for values of num_pixels other than the default.
        Other checks comparing multi-pixel SNR with single-pixel SNR done in
        ut_cgi_eetc.py.'''
        values = [15, 0.5, 3.3] # non-integer values should work, too
        for val in values:
            calc_gain_fixed_time(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, t=self.t,
                              gmax=self.gmax, overhead=self.overhead,
                              num_pixels=val)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""

        check_list = ut_check.rnslist

        # target_snr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=perr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  t=self.t,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fluxe
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr, fluxe=perr,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  t=self.t,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fluxe_bright
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=perr,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead)
            pass

        # darke
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright, darke=perr,
                                cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                                Lij=self.Lij, alpha0=self.alpha0,
                                fwc=self.fwc, alpha1=self.alpha1,
                                fwc_em=self.fwc_em, Nmin=self.Nmin,
                                Nmax=self.Nmax, t=self.t, gmax=self.gmax,
                                overhead=self.overhead)
            pass

        # cic
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=perr, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead)
            pass

        # rn
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=perr,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead)
            pass

        # X
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=perr, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead)
            pass

        # a
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=perr, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead)
            pass

        # overhead
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=perr)
            pass


        # n
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead, n=perr)
            pass

        # tol
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead,
                                tol=perr)
            pass

        # delta_constr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead,
                                delta_constr=perr)
            pass
        pass

    # num_pixels
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead,
                                num_pixels=perr)
            pass
        pass

    def test_invalid_postive_scalar_integer(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.psilist

        # Lij
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=perr,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead)
            pass

        # fwc
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=perr,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead)
            pass

        # fwc_em
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=perr,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead)
            pass

        # Nmin
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=perr, Nmax=self.Nmax, t=self.t,
                                gmax=self.gmax, overhead=self.overhead)
            pass

        # Nmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=perr, t=self.t,
                                gmax=self.gmax, overhead=self.overhead)
            pass

        # Nem
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t, gmax=self.gmax,
                                overhead=self.overhead,
                                Nem=perr)
            pass
        pass

    def test_invalid_real_positive_scalar(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.rpslist

        # alpha0
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=perr, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead)
            pass

        # alpha1
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=perr, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=self.gmax, overhead=self.overhead)
            pass

        # t
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                  fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  t=perr, gmax=self.gmax,
                                  overhead=self.overhead)
            pass

        # gmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t,
                                gmax=perr, overhead=self.overhead)
            pass

    def test_invalid_nonnegative_scalar_integer(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.nsilist

        # opt_choice
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                  fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  t=self.t,
                                  gmax=self.gmax, overhead=self.overhead,
                                  opt_choice=perr)
            pass

    def test_invalid_fluxe_bright(self):
        """Invalid inputs fail as expected"""
        perr = self.fluxe_bright*1.1
        with self.assertRaises(ValueError):
            calc_gain_fixed_time(target_snr=self.target_snr, fluxe=perr,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              t=self.t,
                              gmax=self.gmax, overhead=self.overhead)

    def test_invalid_alpha_gt_1(self):
        """Invalid inputs fail as expected"""
        perr = 1.1
        with self.assertRaises(ValueError):
            calc_gain_fixed_time(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=perr, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              t=self.t,
                              gmax=self.gmax, overhead=self.overhead)

        perr = 1.1
        with self.assertRaises(ValueError):
            calc_gain_fixed_time(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=perr, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              t=self.t,
                              gmax=self.gmax, overhead=self.overhead)

    def test_invalid_nmax_lt_nmin(self):
        """Invalid inputs caught"""
        nmax = 10
        for nmin in [20, 10]:

            with self.assertRaises(ValueError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                            fluxe=self.fluxe, fluxe_bright=self.fluxe_bright,
                            darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=nmin, Nmax=nmax, t=self.t, gmax=self.gmax,
                            overhead=self.overhead)
            pass

    def test_invalid_gain(self):
        """Invalid inputs caught"""

        for perr in [0.9, 1]:
            with self.assertRaises(ValueError):
                calc_gain_fixed_time(target_snr=self.target_snr,
                                fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                t=self.t, gmax=perr, overhead=self.overhead)
                pass
        pass

    def test_invalid_opt_choice(self):
        """Invalid inputs caught"""
        perr = 2

        with self.assertRaises(ValueError):
            calc_gain_fixed_time(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              t=self.t,
                              gmax=self.gmax, overhead=self.overhead,
                              opt_choice=perr)
            pass

    def test_invalid_full_wells(self):
        """Check the two full well cases raise expected exceptions"""
        fluxe = 5
        fluxe_bright = 10
        darke = 10
        cic = 100
        t = 10
        alpha0 = 1
        alpha1 = 1
        fwc = 10
        fwc_em = 10

        # per-pixel
        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_time(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=fluxe_bright, darke=darke,
                              cic=cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=alpha0, fwc=fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, t=t,
                              gmax=self.gmax, overhead=self.overhead)

        # gain register
        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_time(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=fluxe_bright, darke=darke,
                              cic=cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=alpha1, fwc_em=fwc_em, Nmin=self.Nmin,
                              Nmax=self.Nmax, t=t,
                              gmax=self.gmax, overhead=self.overhead)

        pass

class TestCalcGainFixedN(unittest.TestCase):
    """
    Tests for calc_gain_fixed_N function.
    """

    def setUp(self):
        self.target_snr = 7 # unitless
        self.fluxe = 10 # e-/sec
        self.fluxe_bright = 100 # e-sec
        self.darke = 8.33e-4 # e-/sec
        self.cic = 0.02 # e-
        self.rn = 160 # e-
        self.X = 5e4 # hits/m^2/sec
        self.a = 1.69e-10 # m^2/pixel
        self.Lij = 512 # pixels
        self.alpha0 = 0.75 # unitless
        self.fwc = 50000 # e-
        self.alpha1 = 0.75 # unitless
        self.fwc_em = 90000 # e-
        self.N = 20 # frames
        self.tmin = 0.264 # seconds/frame
        self.tmax = 120. # seconds/frame
        self.gmax = 5000 # unitless
        self.overhead = 0 # seconds
        self.opt_choice = 0 # 0 or 1
        self.n = 4 # number of standard deviations below fwc
        self.Nem = 604 # number of gain register cells
        self.tol = 1e-30 # tolerance level used by optimizations
        self.delta_constr = 1e-4 # constraints satisfied up to this fraction

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """good inputs for opt_choice = 0 complete without an exception"""
        calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          N=self.N, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead)
        pass

    def test_opt_choice_success(self):
        """good inputs for opt_choice=1 complete without exception"""
        calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          N=self.N, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead, opt_choice=1)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_first_both(self, mock_min):
        """good inputs for opt_choice = 0 complete without an exception.
        res1_cond and res2_cond both true.  Ends after first optimization
         with res2 results since it has the shorter total exposure time."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools, and they definitely
        # satisfy rail, em_rail, and target_snr constraints in the optimization
        _g1 = 1
        _t1 = self.tmin+1
        _t2 = self.tmin # so N*tfr2 will be better

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g1, _t2])
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_fixed_N(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead)
        self.assertTrue(out[-1] == 0)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t2)
        self.assertTrue(out[2] == self.N)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_first_one(self, mock_min):
        """good inputs for opt_choice = 0 complete without an exception.
        res1_cond true while res2_cond false.  Ends after first optimization
         with res1 results."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools, and they definitely
        # satisfy rail, em_rail, and target_snr constraints in the optimization
        _g1 = 1
        _t1 = self.tmin+1
        _g2 = 1e30 # ensures that res2_cond is false
        _t2 = self.tmin # would give a shorter N*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _t2])
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_fixed_N(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead)
        self.assertTrue(out[-1] == 0)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == self.N)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_second_both(self, mock_min):
        """bad inputs for opt_choice = 0.
        Ends after second optimization due to constraint disagreement in first
        optimization scheme.  res3_cond and res4_cond both true.  Gives res4
        outputs due to bigger SNR."""

        _g1 = 1
        _t1 = self.tmin

        # constraints not met (can't reach target SNR of 1e15)
        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g1, _t1])

        #now for 2nd optimzation scheme, let the constraints be met
        _t4 = _t1+1  # gives bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t4])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_fixed_N(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t4)
        self.assertTrue(out[2] == self.N)
        pass


    @patch('scipy.optimize.minimize')
    def test_end_after_second_one(self, mock_min):
        """bad inputs for opt_choice = 0.
        Ends after 2nd optimization due to constraint disagreement in first
        optimization scheme.  res3_cond true while res4_cond false.  Gives res3
        results."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools
        _g1 = 1e30 # res1_cond false
        _t1 = self.tmin+1
        _g2 = 1e30 # ensures that res2_cond is false
        _t2 = self.tmin  # would give a shorter N2*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _t2])

        _g3 = 1
        _g4 = 1e30 # breaks em_rail constraint, even though it would give
        #bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _t2])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_fixed_N(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == self.N)
        pass


    @patch('scipy.optimize.minimize')
    def test_both_fail(self, mock_min):
        """For opt_choice = 0, the 2nd optimization scheme fails due to
        constraint disagreement, and an exception is raised."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools
        _g1 = 1e30 # res1_cond false
        _t1 = self.tmin
        _g2 = 1e30 # ensures that res2_cond is false
        _t2 = self.tmin

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _t2])

        #now for 2nd optimzation scheme, let the constraints disagree
        _g3 = 1e30  # breaks em_rail constraint
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _t1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N, tmin=self.tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_both(self, mock_min):
        """For opt_choice = 1, both res3_cond and res4_cond are true.  Gives
        res4 outputs due to bigger SNR."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = 1
        _t3 = self.tmin
        _t4 = self.tmin+1 #gives bigger SNR
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _t4])

        mock_min.side_effect = [_res3, _res4, _res3]  # should stop after
                                                      # second mock

        # target_snr should have no bearing when opt_choice=1
        out = calc_gain_fixed_N(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead, opt_choice=1)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t4)
        self.assertTrue(out[2] == self.N)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_one(self, mock_min):
        """For opt_choice = 1, res3_cond is true and res4_cond is false.  Gives
        res3 outputs."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = 1
        _t3 = self.tmin
        _g4 = 1e30 # breaks em_rail condition, even though gives bigger snr
        _t4 = self.tmin+1  # would give the bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _t4])

        mock_min.side_effect = [_res3, _res4, _res4]  # should stop after
                                                      # second mock

        # target_snr should have no bearing when opt_choice=1
        out = calc_gain_fixed_N(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead, opt_choice=1)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t3)
        self.assertTrue(out[2] == self.N)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_both_fail(self, mock_min):
        """For opt_choice = 1, the 2nd optimization scheme fails due to
        constraint disagreement, and an exception is raised."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = 1e30  # break em_rail condition
        _t3 = self.tmin
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _t3])

        mock_min.side_effect = [_res3, _res4, _res4]  # should stop after
                                                      # second mock

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N, tmin=self.tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead, opt_choice=1)
        pass

    def test_successful_run_multiple_pixels(self):
        '''Successful run for values of num_pixels other than the default.
        Other checks comparing multi-pixel SNR with single-pixel SNR done in
        ut_cgi_eetc.py.'''
        values = [15, 0.5, 3.3] # non-integer values should work, too
        for val in values:
            calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N, tmin=self.tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead, num_pixels=val)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""

        check_list = ut_check.rnslist

        # target_snr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=perr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fluxe
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=perr,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fluxe_bright
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=perr,
                                  fluxe_bright=perr,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # darke
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright, darke=perr,
                                  cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                                  Lij=self.Lij, alpha0=self.alpha0,
                                  fwc=self.fwc, alpha1=self.alpha1,
                                  fwc_em=self.fwc_em, N=self.N,
                                  tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead)
            pass

        # cic
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=perr, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # rn
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=perr,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # X
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=perr, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # a
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=perr, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # tmin
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N, tmin=perr,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead)
            pass

        # overhead
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N, tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=perr)
            pass

        # n
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  n=perr)
            pass

        # tol
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  tol=perr)
            pass

        # delta_constr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  delta_constr=perr)
            pass
        pass

        # num_pixels
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  num_pixels=perr)
            pass
        pass

    def test_invalid_positive_scalar_integer(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.psilist

        # Lij
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=perr,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fwc
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=perr,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fwc_em
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=perr,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # Nem
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead,
                                  Nem=perr)
            pass

        # N
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=perr,
                                  tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead)
            pass
        pass

    def test_invalid_real_positive_scalar(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.rpslist

        # alpha0
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=perr, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # alpha1
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=perr, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # tmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=perr, gmax=self.gmax,
                                  overhead=self.overhead)
            pass

        # gmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=perr, overhead=self.overhead)
            pass

    def test_invalid_nonnegative_scalar_integer(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.nsilist

        # opt_choice
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  opt_choice=perr)
            pass

    def test_invalid_fluxe_bright(self):
        """Invalid inputs fail as expected"""
        perr = self.fluxe_bright*1.1
        with self.assertRaises(ValueError):
            calc_gain_fixed_N(target_snr=self.target_snr, fluxe=perr,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead)

    def test_invalid_alpha_gt_1(self):
        """Invalid inputs fail as expected"""
        perr = 1.1
        with self.assertRaises(ValueError):
            calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=perr, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead)

        perr = 1.1
        with self.assertRaises(ValueError):
            calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=perr, fwc_em=self.fwc_em,
                              N=self.N,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead)

    def test_invalid_tmax_lt_tmin(self):
        """Invalid inputs caught"""
        tmax = 10
        for tmin in [10, 20]:

            with self.assertRaises(ValueError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            N=self.N, tmin=tmin,
                            tmax=tmax, gmax=self.gmax, overhead=self.overhead)
        pass

    def test_invalid_gain(self):
        """Invalid inputs caught"""

        for perr in [0.9, 1]:
            with self.assertRaises(ValueError):
                calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N,
                                tmin=self.tmin, tmax=self.tmax, gmax=perr,
                                overhead=self.overhead)
                pass
        pass

    def test_invalid_opt_choice(self):
        """Invalid inputs caught"""
        perr = 2

        with self.assertRaises(ValueError):
            calc_gain_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              opt_choice=perr)
            pass

    def test_invalid_full_wells(self):
        """Check the two full well cases raise expected exceptions"""
        fluxe = 5
        fluxe_bright = 10
        darke = 10
        cic = 100
        tmin = 10
        alpha0 = 1
        alpha1 = 1
        fwc = 10
        fwc_em = 10

        # per-pixel
        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_N(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=fluxe_bright, darke=darke,
                              cic=cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=alpha0, fwc=fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N, tmin=tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead)

        # gain register
        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_N(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=fluxe_bright, darke=darke,
                              cic=cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=alpha1, fwc_em=fwc_em, N=self.N,
                              tmin=tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead)

        pass

class TestCalcGainFixedG(unittest.TestCase):
    """
    Tests for calc_gain_fixed_g function.
    """

    def setUp(self):
        self.target_snr = 7 # unitless
        self.fluxe = 10 # e-/sec
        self.fluxe_bright = 100 # e-sec
        self.darke = 8.33e-4 # e-/sec
        self.cic = 0.02 # e-
        self.rn = 160 # e-
        self.X = 5e4 # hits/m^2/sec
        self.a = 1.69e-10 # m^2/pixel
        self.Lij = 512 # pixels
        self.alpha0 = 0.75 # unitless
        self.fwc = 50000 # e-
        self.alpha1 = 0.75 # unitless
        self.fwc_em = 90000 # e-
        self.Nmin = 1 # frames
        self.Nmax = 49 # frames
        self.tmin = 0.264 # seconds/frame
        self.tmax = 120. # seconds/frame
        self.g = 5 # unitless
        self.overhead = 0 # seconds
        self.opt_choice = 0 # 0 or 1
        self.n = 4 # number of standard deviations below fwc
        self.Nem = 604 # number of gain register cells
        self.tol = 1e-30 # tolerance level used by optimizations
        self.delta_constr = 1e-4 # constraints satisfied up to this fraction

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """good inputs for opt_choice = 0 complete without an exception"""
        calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, g=self.g, overhead=self.overhead)
        pass

    def test_opt_choice_success(self):
        """good inputs for opt_choice=1 complete without exception"""
        calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, g=self.g, overhead=self.overhead,
                          opt_choice=1)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_first_both(self, mock_min):
        """good inputs for opt_choice = 0 complete without an exception.
        res1_cond and res2_cond both true.  Ends after first optimization
         with res2 results since it has the shorter total exposure time."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools, and they definitely
        # satisfy rail, em_rail, and target_snr constraints in the optimization
        _t1 = self.tmin
        _N1 = self.Nmin+1
        _N2 = self.Nmin  # so N2*tfr2 will be better

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_t1, _N2])
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_t1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_t1, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_fixed_g(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, g=self.g,
                                overhead=self.overhead)
        self.assertTrue(out[-1] == 0)
        self.assertTrue(out[0] == self.g)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N2)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_first_one(self, mock_min):
        """good inputs for opt_choice = 0 complete without an exception.
        res1_cond true while res2_cond false.  Ends after first optimization
         with res1 results."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools, and they definitely
        # satisfy rail, em_rail, and target_snr constraints in the optimization
        _t1 = self.tmin
        _N1 = self.Nmin+1
        _t2 = 1e30 # ensures that res2_cond is false
        _N2 = self.Nmin

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_t2, _N2])
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_t1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_t1, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_fixed_g(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, g=self.g,
                                overhead=self.overhead)
        self.assertTrue(out[-1] == 0)
        self.assertTrue(out[0] == self.g)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N1)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_second_both(self, mock_min):
        """bad inputs for opt_choice = 0.
        Ends after second optimization due to constraint disagreement in first
        optimization scheme.  res3_cond and res4_cond both true.  Gives res4
        outputs due to bigger SNR."""

        _t1 = self.tmin
        _N1 = self.Nmin

        # constraints not met (can't reach target SNR of 1e15)
        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_t1, _N1])

        #now for 2nd optimzation scheme, let the constraints be met
        _N4 = self.Nmin+1  # gives bigger snr since snr proportional to sqrt(N)
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_t1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_t1, _N4])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_fixed_g(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, g=self.g,
                                overhead=self.overhead)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == self.g)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N4)
        pass


    @patch('scipy.optimize.minimize')
    def test_end_after_second_one(self, mock_min):
        """bad inputs for opt_choice = 0.
        Ends after 2nd optimization due to constraint disagreement in first
        optimization scheme.  res3_cond true while res4_cond false.  Gives res3
        results."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools
        _t1 = 1e30 # ensures that res1_cond false
        _N1 = self.Nmin+1
        _t2 = 1e30 # ensures that res2_cond is false
        _N2 = self.Nmin  # would give a shorter N2*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_t2, _N2])

        _t3 = self.tmin
        _t4 = 1e30 # breaks em_rail constraint, even though it would give
        #bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_t3, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_t4, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_gain_fixed_g(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, g=self.g,
                                overhead=self.overhead)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == self.g)
        self.assertTrue(out[1] == _t3)
        self.assertTrue(out[2] == _N1)
        pass


    @patch('scipy.optimize.minimize')
    def test_both_fail(self, mock_min):
        """For opt_choice = 0, the 2nd optimization scheme fails due to
        constraint disagreement, and an exception is raised."""

         # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools
        _t1 = 1e30 # res1_cond false
        _N1 = self.Nmin+1
        _t2 = 1e30 # ensures that res2_cond is false
        _N2 = self.Nmin  # would give a shorter N2*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_t2, _N2])

        #now for 2nd optimzation scheme, let the constraints disagree
        _t3 = 1e30  # breaks constraint
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_t3, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_t3, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                              tmax=self.tmax, g=self.g, overhead=self.overhead)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_both(self, mock_min):
        """For opt_choice = 1, both res3_cond and res4_cond are true.  Gives
        res4 outputs due to bigger SNR."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _t3 = self.tmin
        _N3 = self.Nmin
        _N4 = _N3+1  # gives the bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_t3, _N3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_t3, _N4])

        mock_min.side_effect = [_res3, _res4, _res3]  # should stop after
                                                      # second mock

        # target_snr should have no bearing when opt_choice=1
        out = calc_gain_fixed_g(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, g=self.g,
                                overhead=self.overhead, opt_choice=1)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == self.g)
        self.assertTrue(out[1] == _t3)
        self.assertTrue(out[2] == _N4)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_one(self, mock_min):
        """For opt_choice = 1, res3_cond is true and res4_cond is false.  Gives
        res3 outputs."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _t3 = self.tmin
        _N3 = self.Nmin
        _t4 = 1e30 # breaks em_rail condition, even though gives bigger snr
        _N4 = _N3+1
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_t3, _N3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_t4, _N4])

        mock_min.side_effect = [_res3, _res4, _res4]  # should stop after
                                                      # second mock

        # target_snr should have no bearing when opt_choice=1
        out = calc_gain_fixed_g(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, g=self.g,
                                overhead=self.overhead, opt_choice=1)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == self.g)
        self.assertTrue(out[1] == _t3)
        self.assertTrue(out[2] == _N3)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_both_fail(self, mock_min):
        """For opt_choice = 1, the 2nd optimization scheme fails due to
        constraint disagreement, and an exception is raised."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _t3 = 1e30  # break constraint
        _N3 = self.Nmin
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_t3, _N3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_t3, _N3])

        mock_min.side_effect = [_res3, _res4, _res4]  # should stop after
                                                      # second mock

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                              tmax=self.tmax, g=self.g,
                              overhead=self.overhead, opt_choice=1)
        pass

    def test_successful_run_multiple_pixels(self):
        '''Successful run for values of num_pixels other than the default.
        Other checks comparing multi-pixel SNR with single-pixel SNR done in
        ut_cgi_eetc.py.'''
        values = [15, 0.5, 3.3] # non-integer values should work, too
        for val in values:
            calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                              tmax=self.tmax, g=self.g,
                              overhead=self.overhead, num_pixels=val)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""

        check_list = ut_check.rnslist

        # target_snr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=perr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead)
            pass

        # fluxe
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=perr,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead)
            pass

        # fluxe_bright
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=perr,
                                  fluxe_bright=perr,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead)
            pass

        # darke
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright, darke=perr,
                                  cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                                  Lij=self.Lij, alpha0=self.alpha0,
                                  fwc=self.fwc, alpha1=self.alpha1,
                                  fwc_em=self.fwc_em, Nmin=self.Nmin,
                                  Nmax=self.Nmax, tmin=self.tmin,
                                  tmax=self.tmax, g=self.g,
                                  overhead=self.overhead)
            pass

        # cic
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=perr, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead)
            pass

        # rn
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=perr,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead)
            pass

        # X
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=perr, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead)
            pass

        # a
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=perr, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead)
            pass

        # tmin
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax, tmin=perr,
                                  tmax=self.tmax, g=self.g,
                                  overhead=self.overhead)
            pass

        # overhead
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax, g=self.g,
                                  overhead=perr)
            pass


        # n
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead, n=perr)
            pass

        # tol
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead, tol=perr)
            pass

        # delta_constr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead,
                                  delta_constr=perr)
            pass
        pass

        # num_pixels
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead,
                                  num_pixels=perr)
            pass
        pass

    def test_invalid_postive_scalar_integer(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.psilist

        # Lij
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=perr,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead)
            pass

        # fwc
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=perr,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead)
            pass

        # fwc_em
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=perr,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead)
            pass

        # Nmin
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=perr, Nmax=self.Nmax, tmin=self.tmin,
                                  tmax=self.tmax, g=self.g,
                                  overhead=self.overhead)
            pass

        # Nmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=perr, tmin=self.tmin,
                                  tmax=self.tmax, g=self.g,
                                  overhead=self.overhead)
            pass

        # Nem
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax, g=self.g,
                                  overhead=self.overhead,
                                  Nem=perr)
            pass
        pass

    def test_invalid_real_positive_scalar(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.rpslist

        # alpha0
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=perr, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead)
            pass

        # alpha1
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=perr, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead)
            pass

        # tmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=perr, g=self.g,
                                  overhead=self.overhead)
            pass

        # g
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=perr, overhead=self.overhead)
            pass

    def test_invalid_nonnegative_scalar_integer(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.nsilist

        # opt_choice
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  g=self.g, overhead=self.overhead,
                                  opt_choice=perr)
            pass

    def test_invalid_fluxe_bright(self):
        """Invalid inputs fail as expected"""
        perr = self.fluxe_bright*1.1
        with self.assertRaises(ValueError):
            calc_gain_fixed_g(target_snr=self.target_snr, fluxe=perr,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              g=self.g, overhead=self.overhead)

    def test_invalid_alpha_gt_1(self):
        """Invalid inputs fail as expected"""
        perr = 1.1
        with self.assertRaises(ValueError):
            calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=perr, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              g=self.g, overhead=self.overhead)

        perr = 1.1
        with self.assertRaises(ValueError):
            calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=perr, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              g=self.g, overhead=self.overhead)

    def test_invalid_nmax_lt_nmin(self):
        """Invalid inputs caught"""
        nmax = 10
        for nmin in [20, 10]:

            with self.assertRaises(ValueError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=nmin, Nmax=nmax, tmin=self.tmin,
                            tmax=self.tmax, g=self.g, overhead=self.overhead)
        pass

    def test_invalid_tmax_lt_tmin(self):
        """Invalid inputs caught"""
        tmax = 10
        for tmin in [10, 20]:

            with self.assertRaises(ValueError):
                calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                            tmax=tmax, g=self.g, overhead=self.overhead)
        pass

    def test_invalid_opt_choice(self):
        """Invalid inputs caught"""
        perr = 2

        with self.assertRaises(ValueError):
            calc_gain_fixed_g(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              g=self.g, overhead=self.overhead,
                              opt_choice=perr)
            pass

    def test_invalid_full_wells(self):
        """Check the two full well cases raise expected exceptions"""
        fluxe = 5
        fluxe_bright = 10
        darke = 10
        cic = 100
        tmin = 10
        alpha0 = 1
        alpha1 = 1
        fwc = 10
        fwc_em = 10

        # per-pixel
        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_g(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=fluxe_bright, darke=darke,
                              cic=cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=alpha0, fwc=fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                              tmax=self.tmax, g=self.g, overhead=self.overhead)

        # gain register
        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_g(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=fluxe_bright, darke=darke,
                              cic=cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=alpha1, fwc_em=fwc_em, Nmin=self.Nmin,
                              Nmax=self.Nmax, tmin=tmin, tmax=self.tmax,
                              g=self.g, overhead=self.overhead)

        pass

class TestCalcGainFixedNTime(unittest.TestCase):
    """
    Tests for calc_gain_fixed_Ntime function.
    """

    def setUp(self):
        self.t_tot = 200 # s
        self.fluxe = 10 # e-/sec
        self.fluxe_bright = 100 # e-sec
        self.darke = 8.33e-4 # e-/sec
        self.cic = 0.02 # e-
        self.rn = 160 # e-
        self.X = 5e4 # hits/m^2/sec
        self.a = 1.69e-10 # m^2/pixel
        self.Lij = 512 # pixels
        self.alpha0 = 0.75 # unitless
        self.fwc = 50000 # e-
        self.alpha1 = 0.75 # unitless
        self.fwc_em = 90000 # e-
        self.Nmin = 1 # frames
        self.Nmax = 49 # frames
        self.tmin = 0.264 # seconds/frame
        self.tmax = 120. # seconds/frame
        self.gmax = 5000 # unitless
        self.overhead = 0 # seconds
        self.n = 4 # number of standard deviations below fwc
        self.Nem = 604 # number of gain register cells
        self.tol = 1e-30 # tolerance level used by optimizations
        self.delta_constr = 1e-4 # constraints satisfied up to this fraction
        self.hard_limit = True # impose N*t = t_tot exactly

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """good inputs complete without an exception. And this is a case
        where t_lb != t_ub."""
        out = calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead)
        # t_lb can only equal t_ub if they are tmax or tmin
        self.assertTrue(out[1] > self.tmin)
        self.assertTrue(out[1] < self.tmax)
        pass

    def test_invalid_wells(self):
        """Inputs saturate wells before getting to scipy solving functions."""
        fluxe = 5
        fluxe_bright = 10
        darke = 10
        cic = 100
        tmin = 10
        alpha0 = 1
        alpha1 = 1
        fwc = 10
        fwc_em = 10

        # per-pixel well
        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=darke,
                          cic=cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=alpha0, fwc=fwc,
                          alpha1=alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead)

        # EM gain well
        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=darke,
                          cic=cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=alpha0, fwc=self.fwc,
                          alpha1=alpha1, fwc_em=fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead)

    def test_invalid_t_tot(self):
        """t_tot input incompatible with Nmin, tmin, Nmax, and tmax."""
        # t_tot too big
        t_tot_big = self.Nmax*self.tmax + 1
        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_Ntime(t_tot=t_tot_big, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead)

        # t_tot too small
        Nmin = 5
        t_tot_small = Nmin*self.tmin - 1
        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_Ntime(t_tot=t_tot_small, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead)


    @patch('scipy.optimize.minimize')
    def test_hard_limit_one(self, mock_min):
        """One passes:  res3 passes b/c of bigger snr, while res4 saturates."""
        _g3 = 10 # bigger than _g4, so gives bigger snr
        _t1 = self.t_tot/(self.Nmax - 2) # so N will be integer
        _g4 = 1e30 # saturates
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _t1])

        mock_min.side_effect = [_res3, _res4]

        out = calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                    fluxe_bright=self.fluxe_bright, darke=self.darke,
                    cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                    Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                    alpha1=self.alpha1, fwc_em=self.fwc_em,
                    Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                    tmax=self.tmax, gmax=self.gmax, overhead=self.overhead)

        # since N integer, neither t  nor g will be adjusted
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == self.Nmax - 2)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_both(self, mock_min):
        """Both pass:  constraints met for both res3 and res4, but res4 wins
        b/c of bigger snr."""
        _g4 = 10 # bigger than _g3, so gives bigger snr
        _t1 = self.t_tot/(self.Nmax - 2) # so N will be integer
        # bigger time but smaller N, and N drives SNR more than t here
        _t2 = self.t_tot/(self.Nmax - 3)
        _g3 = 5
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _t2])

        mock_min.side_effect = [_res3, _res4]

        out = calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                        fluxe_bright=self.fluxe_bright, darke=self.darke,
                        cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                        Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                        alpha1=self.alpha1, fwc_em=self.fwc_em,
                        Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                        tmax=self.tmax, gmax=self.gmax, overhead=self.overhead)
        # since N integer, neither t nor g will not be adjusted
        self.assertTrue(out[0] == _g4)
        self.assertTrue(out[1] == _t2)
        self.assertTrue(out[2] == self.Nmax - 3)

    @patch('scipy.optimize.minimize')
    def test_both_fail_init(self, mock_min):
        """inputs pass initial well checks and give an exception for a case
        where t_lb != t_ub (since we use the same numbers as previous test).
        res*_cond_init are False.  This is the case regardless of hard_limit's
        value since this check happens before hard_limit is used."""
        _g1 = 1e30 # saturates
        _t1 = self.t_tot/2
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res4]

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead)
        pass

    def test_hard_limit(self):
        """Test that hard_limit=True gives lower or equal SNR compared to the
        False case, typically."""

        out_true = calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead)

        out_false = calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead, hard_limit=False)

        #SNR comparison
        self.assertTrue(out_true[3] >= out_false[3])
        # since t is solved based on the rounded-up N value, t for True should
        #be smaller (or equal, if N from optimization happens to be integer)
        self.assertTrue(out_true[1] <= out_false[1])
        #should be same number of frames in both cases, though
        self.assertTrue(out_true[2] == out_false[2])
        # the False case relaxes the t_tot constraint, and N*t could be a bit
        # bigger than t_tot
        self.assertTrue(out_false[4] >= self.t_tot)


    def test_overhead(self):
        """Test that nonzero overhead is correctly incorporated"""
        t_tot = 80
        overhead = 3
        tol = 1e-13

        out_true = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=overhead, hard_limit=True)
        out_false = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=overhead, hard_limit=False)
        # Hard limit enforced for True
        self.assertTrue(np.max(np.abs(t_tot - out_true[4])) < tol)
        # ...not necessarily for False
        self.assertTrue(out_false[4] >= t_tot)
        # Either way, overhead is included
        self.assertTrue(out_true[1]*out_true[2] < t_tot)
        self.assertTrue(out_false[1]*out_false[2] < t_tot)


    @patch('scipy.optimize.minimize')
    def test_hard_limit_local_snr_max(self, mock_min):
        """Test hard_limit=True case where there's a local max in SNR with
        respect to g."""
        _g1 = 30
        _t1 = self.t_tot/4.5 #so N is 4.5
        _g2 = 10
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g2, _t1])
        mock_min.side_effect = [_res3, _res4, _res3, _res4]
        fwc_em = 50000
        fluxe = 10
        fluxe_bright1 = 10
        #this causes the bound on g due to saturation to dip below _g1
        fluxe_bright2 = 50

        out1 = calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright1, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead)
        # g is some value b/w 0 and gmax for the local SNR max
        self.assertTrue(out1[0] < self.gmax)

        out2 = calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright2, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead)

        # the g that gives a local SNR max shouldn't change since it's
        # independent of fluxe_bright, so the lower g output must be due to
        # the lower value of g that saturates due to the higher fluxe_bright2
        self.assertTrue(out2[0] < out1[0])

    @patch('scipy.optimize.minimize')
    def test_hard_limit_Nmin(self, mock_min):
        """The values of N found for res3 and/or res4 are below Nmin, so
        Nmin is the output value for N."""
        _g1 = 1
        t_tot = 80
        delta_constr = 1 # to ensure that res_*_init are passed
        #(otherwise, I could increase Nmin)
        _t1 = t_tot/(self.Nmin - 0.5) #gives N < Nmin
        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res3]

        out = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          delta_constr=delta_constr)

        self.assertTrue(out[2] == self.Nmin)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_Nmin_fail(self, mock_min):
        """The values of N found for res3 and/or res4 are below Nmin, and
        increasing N to Nmin makes t dip below tmin.  Otherwise, other
        constraints are satisfied, as illustrated in unit test just above this
        one."""
        tmin = self.t_tot/self.Nmin - 1
        tmax = tmin + 5 #won't mess up the satisfaction of other constraints
        _g1 = 1
        t_tot = 80
        _t1 = self.t_tot/(self.Nmin - 0.5) #gives N < Nmin
        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res3]

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                            tmax=tmax, gmax=self.gmax, overhead=self.overhead)

    def test_hard_limit_both_fail_final(self):
        """Both res3_cond_final and res4_cond_final fail.
        With hard_limit=False, these inputs do not fail.  The decrease in N
        in the True case increases t enough to cause saturation.  Also, these
        inputs give gsol < 1 for floor of N (before gsol is restricted to be
        between _gmin and gmax).  res*_cond_init is fine, but the final checks
        fail due to the change in t when hard_limit=True."""
        rn = 100
        fluxe = 1000
        fwc_em = 50000
        fwc = 60000
        fluxe_bright = 2000
        t_tot = 14760
        Nmax = 25200
        tmax = 6300
        # runs fine
        calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=fwc,
                          alpha1=self.alpha1, fwc_em=fwc_em,
                          Nmin=self.Nmin, Nmax=Nmax, tmin=self.tmin,
                          tmax=tmax, gmax=self.gmax,
                          overhead=self.overhead, hard_limit=False)

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=fwc,
                          alpha1=self.alpha1, fwc_em=fwc_em,
                          Nmin=self.Nmin, Nmax=Nmax, tmin=self.tmin,
                          tmax=tmax, gmax=self.gmax,
                          overhead=self.overhead, hard_limit=True)

    @patch('scipy.optimize.minimize')
    def test_hard_limit_middle_fail(self, mock_min):
        """The 'middle' elif case (floor and ceiling of N both b/w Nmin
        and Nmax) where the adjusted t with either ceiling or floor is outside
        of t bounds."""
        _g1 = 1
        t_tot = 80
        _t1 = 1.98 # gives N as t_tot/_t1 = 40.4
        tmin = 1.952 #just above 80/41
        tmax = 1.99 # just below 80/40

        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res3]

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax, overhead=self.overhead)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_middle_tmax(self, mock_min):
        """The 'middle' elif case (floor and ceiling of N both b/w Nmin
        and Nmax) where the adjusted t with floor is bigger than tmax."""
        _g1 = 1
        t_tot = 80
        _t1 = 1.98 # gives N as t_tot/_t1 = 40.4
        tmin = 1
        tmax = 1.99 # just below 80/40

        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res3]

        out = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                        fluxe_bright=self.fluxe_bright, darke=self.darke,
                        cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                        Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                        alpha1=self.alpha1, fwc_em=self.fwc_em,
                        Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                        tmax=tmax, gmax=self.gmax, overhead=self.overhead)

        self.assertTrue(out[2] == 41)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_middle_tmin(self, mock_min):
        """The 'middle' elif case (floor and ceiling of N both b/w Nmin
        and Nmax) where the adjusted t with ceil is smaller than tmin."""
        _g1 = 1
        t_tot = 80
        _t1 = 1.98 # gives N as t_tot/_t1 = 40.4
        tmin = 1.952 #just above 80/41
        tmax = 3

        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res3]

        out = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                        fluxe_bright=self.fluxe_bright, darke=self.darke,
                        cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                        Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                        alpha1=self.alpha1, fwc_em=self.fwc_em,
                        Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                        tmax=tmax, gmax=self.gmax, overhead=self.overhead)

        self.assertTrue(out[2] == 40)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_Nmax_2(self, mock_min):
        """The values of N found for res3 and/or res4 are above Nmax, so
        Nmax is the output value for N."""
        _g1 = 1
        t_tot = 80
        delta_constr = 1 # to ensure that res_*_init are passed
        _t1 = t_tot/(self.Nmax + 0.5) #gives N > Nmax
        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res3]

        out = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          delta_constr=delta_constr)

        self.assertTrue(out[2] == self.Nmax)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_Nmax_fail(self, mock_min):
        """The values of N found for res3 and/or res4 are above Nmax, and
        decreasing N to Nmax makes t rise above tmax.  Otherwise, other
        constraints are satisfied, as illustrated in unit test just above this
        one."""
        tmax = self.t_tot/self.Nmin + 1
        tmin = tmax - 5 #won't mess up the satisfaction of other constraints
        _g1 = 1
        t_tot = 80
        _t1 = self.t_tot/(self.Nmax + 0.5) #gives N > Nmax
        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res3]

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                            tmax=tmax, gmax=self.gmax, overhead=self.overhead)

    def test_hard_limit_t_snr(self):
        """Illustrate case where time drives the SNR more than N, and floor
        instead of ceiling of N is optimal. This also tests case where
        N ceiling and floor are contained within Nmin and Nmax."""
        fluxe = 1000
        fluxe_bright = 1000
        rn = 100
        t_tot = 14760
        Nmax = 25200
        tmax = 6300
        tmin = 1
        delta_constr = 1e-2

        out_true = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax,
                          overhead=self.overhead, hard_limit=True,
                          delta_constr=delta_constr)

        out_false = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax,
                          overhead=self.overhead, hard_limit=False,
                          delta_constr=delta_constr)
        # in this case, the hard_limit rounded N down to get a bigger snr
        self.assertTrue(out_true[2] == out_false[2] - 1)
        # and the time for the True case is bigger since N was rounded down
        self.assertTrue(out_true[1] >= out_false[1])

    @patch('scipy.optimize.minimize')
    def test_not_hard_limit_Nmin(self, mock_min):
        """The values of N found for res3 and/or res4 are below Nmin, so
        the ceiling of N, Nmin, is the output value. For hard_limit=False."""
        _g1 = 1
        t_tot = 80
        delta_constr = 1 # to ensure that res_*_init are passed
        #(otherwise, I could increase Nmin)
        _t1 = t_tot/(self.Nmin - 0.5) #gives N < Nmin
        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res3]

        out = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead, hard_limit=False,
                          delta_constr=delta_constr)

        self.assertTrue(out[2] == self.Nmin)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_Nmax(self, mock_min):
        """The values of N found for res3 and/or res4 are above Nmax, so
        Nmax is the output value for N."""
        _g1 = 1
        t_tot = 80
        delta_constr = 1 # to ensure that res_*_init are passed
        #(otherwise, I could increase Nmin)
        _t1 = t_tot/(self.Nmax + 0.5) #gives N > Nmax
        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res3]

        out = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead, hard_limit=False,
                          delta_constr=delta_constr)

        self.assertTrue(out[2] == self.Nmax)
        pass

    def test_t_lb_equals_t_ub_no_gsol(self):
        """Exercise the case where emrail has no
        solution for g that saturates.  In this case, gmax is what maximizes
        SNR."""
        cic = 0
        darke = 0
        fluxe_bright = 0
        fluxe = 0
        t_tot = 3600
        tmin = 1
        tmax = 120
        Nmin = 1
        Nmax = 30

        out = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=darke,
                          cic=cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax, overhead=self.overhead)

        self.assertTrue(out[0] == self.gmax)
        # SNR is expected be 0 in this case
        self.assertTrue(out[3] == 0)

    def test_t_lb_equals_t_ub(self):
        """inputs such that t_lb=t_ub with gsol b/w _gmin and gmax."""
        # set up a case where t_ub = tmax < t_tot/Nmin and t_lb = tmax = t_ub
        t_tot = 3600
        tmin = 1
        tmax = 120
        Nmin = 1
        Nmax = 30

        out = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax, overhead=self.overhead)

        self.assertTrue(out[0] > 1)
        self.assertTrue(out[0] < self.gmax)
        self.assertTrue(out[1] == tmax)
        self.assertTrue(out[2] == t_tot/tmax)
        self.assertTrue(out[-1] == t_tot)
        pass

    def test_t_lb_equals_t_ub_gmin(self):
        """inputs such that t_lb=t_ub with gsol = _gmin."""
        # set up a case where t_ub = tmax < t_tot/Nmin and t_lb = tmax = t_ub
        t_tot = 3600
        tmin = 1
        tmax = 120
        Nmin = 1
        Nmax = 30
        fwc = self.fwc_em
        n = 0
        # set fluxe_bright big enough so that gsol = 1 (manipulate emrail)
        fluxe_bright = ((self.alpha1*self.fwc_em - self.cic - self.darke*tmax)
                        /tmax)

        out = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax,
                          overhead=self.overhead, n=n)

        self.assertTrue(out[0] == 1)
        self.assertTrue(out[1] == tmax)
        self.assertTrue(out[2] == t_tot/tmax)
        self.assertTrue(out[-1] == t_tot)
        pass

    def test_t_lb_equals_t_ub_gmax(self):
        """inputs such that t_lb=t_ub with gsol > gmax."""
        # set up a case where t_ub = tmax < t_tot/Nmin and t_lb = tmax = t_ub
        t_tot = 3600
        tmin = 1
        tmax = 120
        Nmin = 1
        Nmax = 30
        fluxe = 0.0001
        n = 0
        # set fluxe_bright small enough so that gsol > gmax
        # (manipulate emrail with g = self.gmax+10)
        fluxe_bright = ((self.alpha1*self.fwc_em/(self.gmax+10) - self.cic -
                        self.darke*tmax)/tmax)

        out = calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax,
                          overhead=self.overhead, n=n)

        self.assertTrue(out[0] == self.gmax) #g should be truncated to gmax
        self.assertTrue(out[1] == tmax)
        self.assertTrue(out[2] == t_tot/tmax)
        self.assertTrue(out[-1] == t_tot)
        pass

    @patch('scipy.optimize.fsolve')
    def test_t_lb_equals_t_ub_exception(self, mock_fsolve):
        """inputs such that t_lb=t_ub with gsol b/w _gmin and gmax, but
        simulate a wonky fsolve with a gsol that saturates and causes an
        exception."""
        # set up a case where t_ub = tmax < t_tot/Nmin and t_lb = tmax = t_ub
        t_tot = 3600
        tmin = 1
        tmax = 120
        Nmin = 1
        Nmax = 30
        # output expected in this form
        _fsolve = (np.array([4000]), {}, 1, '')
        # one fsolve for max g from emrail constraint, the next from the g
        #that gives local max in SNR
        mock_fsolve.side_effect = [_fsolve, _fsolve]

        with self.assertRaises(EXCAMOptimizeException):
            calc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax, overhead=self.overhead)
        pass

    def test_successful_run_multiple_pixels(self):
        '''Successful run for values of num_pixels other than the default.
        Other checks comparing multi-pixel SNR with single-pixel SNR done in
        ut_cgi_eetc.py.'''
        values = [15, 0.5, 3.3] # non-integer values should work, too
        for val in values:
            calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead, num_pixels=val)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""

        check_list = ut_check.rnslist

        # t_tot
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=perr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fluxe
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=perr,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fluxe_bright
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=perr,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # darke
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=perr, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # cic
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=perr, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # rn
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=perr,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # X
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=perr, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # a
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=perr, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # tmin
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=perr, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # overhead
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=perr)
            pass

        # n
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax,
                                  overhead=self.overhead, n=perr)
            pass

        # tol
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax,
                                  overhead=self.overhead, tol=perr)
            pass

        # delta_constr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax,
                                  overhead=self.overhead, delta_constr=perr)
            pass
        pass

        # num_pixels
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax,
                                  overhead=self.overhead, num_pixels=perr)
            pass
        pass

    def test_invalid_postive_scalar_integer(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.psilist

        # Lij
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=perr,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fwc
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=perr,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # fwc_em
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=perr,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # Nmin
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=perr, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # Nmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=perr,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # Nem
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax,
                                  overhead=self.overhead, Nem=perr)
            pass
        pass

    def test_invalid_real_positive_scalar(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.rpslist

        # alpha0
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=perr, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # alpha1
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=perr, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # tmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=perr,
                                  gmax=self.gmax, overhead=self.overhead)
            pass

        # gmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=perr, overhead=self.overhead)
            pass

    def test_invalid_hard_limit(self):
        """invalid inputs caught as expected"""

        check_list = [2, 'foo', -3.4]

        # hard_limit
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  hard_limit=perr)
            pass

    def test_invalid_fluxe_bright(self):
        """Invalid inputs fail as expected"""
        perr = self.fluxe_bright*1.1
        with self.assertRaises(ValueError):
            calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=perr,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead)


    def test_invalid_t_tot_overhead(self):
        """Invalid inputs fail as expected"""
        overhead = self.t_tot + 0.1
        with self.assertRaises(ValueError):
            calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=overhead)



    def test_invalid_alpha_gt_1(self):
        """Invalid inputs fail as expected"""
        perr = 1.1
        with self.assertRaises(ValueError):
            calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=perr, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead)

        perr = 1.1
        with self.assertRaises(ValueError):
            calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=perr, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead)

    def test_invalid_nmax_lt_nmin(self):
        """Invalid inputs caught"""
        nmax = 10
        for nmin in [20, 10]:

            with self.assertRaises(ValueError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=nmin, Nmax=nmax, tmin=self.tmin,
                            tmax=self.tmax, gmax=self.gmax,
                            overhead=self.overhead)
        pass

    def test_invalid_tmax_lt_tmin(self):
        """Invalid inputs caught"""
        tmax = 10
        for tmin in [10, 20]:

            with self.assertRaises(ValueError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                            tmax=tmax, gmax=self.gmax, overhead=self.overhead)
        pass

    def test_invalid_gain(self):
        """Invalid inputs caught"""

        for perr in [0.9, 1]:
            with self.assertRaises(ValueError):
                calc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                tmin=self.tmin, tmax=self.tmax,
                                gmax=perr, overhead=self.overhead)
                pass
        pass


class TestCalcPC(unittest.TestCase):
    """
    Tests for calc_pc function.
    """

    def setUp(self):
        self.target_snr = 1 # unitless
        self.fluxe = 0.01 # e-/sec
        self.fluxe_bright = 1 # e-sec
        self.darke = 8.33e-4 # e-/sec
        self.cic = 0.02 # e-
        self.rn = 160 # e-
        self.X = 5e4 # hits/m^2/sec
        self.a = 1.69e-10 # m^2/pixel
        self.Lij = 512 # pixels
        self.alpha0 = 0.75 # unitless
        self.fwc = 50000 # e-
        self.alpha1 = 0.75 # unitless
        self.fwc_em = 90000 # e-
        self.Nmin = 1 # frames
        self.Nmax = 3150*8 # frames
        self.tmin = 0.264 # seconds/frame
        self.tmax = 6300. # seconds/frame
        self.gmax = 5000 # unitless
        self.overhead = 0 # seconds
        self.pc_ecount_max = 0.1 # e-
        self.T_factor = 5 # unitless; # of read noise std devs for threshold
        self.opt_choice = 0 # 0 or 1
        self.n = 4 # number of standard deviations below fwc
        self.Nem = 604 # number of gain register cells
        self.tol = 1e-30 # tolerance level used by optimizations
        self.delta_constr = 1e-4 # constraints satisfied up to this fraction

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """good inputs for opt_choice = 0 complete without an exception"""
        calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=self.pc_ecount_max)
        pass

    def test_opt_choice_success(self):
        """good inputs for opt_choice=1 complete without exception"""
        calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=self.pc_ecount_max, opt_choice=1)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_first_both(self, mock_min):
        """good inputs for opt_choice = 0 complete without an exception.
        res1_cond and res2_cond both true.  Ends after first optimization
         with res2 results since it has the shorter total exposure time."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools, and they definitely
        # satisfy rail, em_rail, and target_snr constraints in the optimization
        _g1 = 5*self.rn
        _t1 = self.tmin
        _N1 = self.Nmin+1
        _N2 = self.Nmin  # so N2*tfr2 will be better

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g1, _t1, _N2])
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t1, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_pc(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max)
        self.assertTrue(out[-1] == 0)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N2)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_first_one(self, mock_min):
        """good inputs for opt_choice = 0 complete without an exception.
        res1_cond true while res2_cond false.  Ends after first optimization
         with res1 results."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools, and they definitely
        # satisfy rail, em_rail, and target_snr constraints in the optimization
        _g1 = 5*self.rn
        _t1 = self.tmin
        _N1 = self.Nmin+1
        _g2 = self.fwc_em/(self.fluxe*self.tmin) # ensures res2_cond is false
        _t2 = self.tmin
        _N2 = self.Nmin  # would give a shorter N2*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _t2, _N2])
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t1, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_pc(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max)
        self.assertTrue(out[-1] == 0)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N1)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_second_both(self, mock_min):
        """bad inputs for opt_choice = 0.
        Ends after second optimization due to constraint disagreement in first
        optimization scheme.  res3_cond and res4_cond both true.  Gives res4
        outputs due to bigger SNR."""

        _g1 = 5*self.rn
        _t1 = self.tmin
        _N1 = 40

        # constraints not met (can't reach target SNR of 1e15)
        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g1, _t1, _N1])

        #now for 2nd optimzation scheme, let the constraints be met
        _N4 = 41 # gives bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t1, _N4])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_pc(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N4)
        pass


    @patch('scipy.optimize.minimize')
    def test_end_after_second_one(self, mock_min):
        """bad inputs for opt_choice = 0.
        Ends after 2nd optimization due to constraint disagreement in first
        optimization scheme.  res3_cond true while res4_cond false.  Gives res3
        results."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools
        _g1 = self.fwc_em/(self.fluxe*self.tmin) # res1_cond false
        _t1 = self.tmin
        _N1 = self.Nmin+1
        _g2 = self.fwc_em/(self.fluxe*self.tmin) # ensures res2_cond is false
        _t2 = self.tmin
        _N2 = self.Nmin  # would give a shorter N2*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _t2, _N2])

        _g3 = 5*self.rn
        _g4 = self.fwc_em/(self.fluxe*self.tmin) # breaks em_rail constraint,
        #even though it would give bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _t1, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_pc(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == _N1)
        pass


    @patch('scipy.optimize.minimize')
    def test_both_fail(self, mock_min):
        """For opt_choice = 0, the 2nd optimization scheme fails due to
        constraint disagreement, and an exception is raised."""

         # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools
        _g1 = self.fwc_em/(self.fluxe*self.tmin) # res1_cond false
        _t1 = self.tmin
        _N1 = self.Nmin+1
        _g2 = self.fwc_em/(self.fluxe*self.tmin) # ensures res2_cond is false
        _t2 = self.tmin
        _N2 = self.Nmin  # would give a shorter N2*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1, _N1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _t2, _N2])

        #now for 2nd optimzation scheme, let the constraints disagree
        _g3 = self.fwc_em/(self.fluxe*self.tmin) # breaks em_rail constraint
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t1, _N1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _t1, _N1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        with self.assertRaises(EXCAMOptimizeException):
            calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_both(self, mock_min):
        """For opt_choice = 1, both res3_cond and res4_cond are true.  Gives
        res4 outputs due to bigger SNR."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = 5*self.rn
        _t3 = self.tmin
        _N3 = 40
        _N4 = 41  # gives the bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t3, _N3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _t3, _N4])

        mock_min.side_effect = [_res3, _res4, _res3]  # should stop after
                                                      # second mock

        # target_snr should have no bearing when opt_choice=1
        out = calc_pc(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max,
                                opt_choice=1)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t3)
        self.assertTrue(out[2] == _N4)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_one(self, mock_min):
        """For opt_choice = 1, res3_cond is true and res4_cond is false.  Gives
        res3 outputs."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = 5*self.rn
        _t3 = self.tmin
        _N3 = self.Nmin
        _g4 = self.fwc_em/(self.fluxe*self.tmin) # breaks em_rail condition,
        #even though gives bigger snr
        _N4 = _N3+1  # would give the bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t3, _N3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _t3, _N4])

        mock_min.side_effect = [_res3, _res4, _res4]  # should stop after
                                                      # second mock

        # target_snr should have no bearing when opt_choice=1
        out = calc_pc(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max,
                                opt_choice=1)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t3)
        self.assertTrue(out[2] == _N3)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_both_fail(self, mock_min):
        """For opt_choice = 1, the 2nd optimization scheme fails due to
        constraint disagreement, and an exception is raised."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = self.fwc_em/(self.fluxe*self.tmin)  # break em_rail condition
        _t3 = self.tmin
        _N3 = self.Nmin
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t3, _N3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _t3, _N3])

        mock_min.side_effect = [_res3, _res4, _res4]  # should stop after
                                                      # second mock

        with self.assertRaises(EXCAMOptimizeException):
            calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max,
                              opt_choice=1)
        pass

    def test_successful_run_multiple_pixels(self):
        '''Successful run for values of num_pixels other than the default.
        Other checks comparing multi-pixel SNR with single-pixel SNR done in
        ut_cgi_eetc.py.'''
        values = [15, 0.5, 3.3] # non-integer values should work, too
        for val in values:
            calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max,
                              num_pixels=val)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""

        check_list = ut_check.rnslist

        # target_snr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=perr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # fluxe
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=perr,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # fluxe_bright
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=perr,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # darke
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright, darke=perr,
                                  cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                                  Lij=self.Lij, alpha0=self.alpha0,
                                  fwc=self.fwc, alpha1=self.alpha1,
                                  fwc_em=self.fwc_em, Nmin=self.Nmin,
                                  Nmax=self.Nmax, tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # cic
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=perr, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # rn
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=perr,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # X
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=perr, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # a
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=perr, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # tmin
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax, tmin=perr,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # overhead
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=perr,
                                  pc_ecount_max=self.pc_ecount_max)
            pass


        # n
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max, n=perr)
            pass

        # tol
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max, tol=perr)
            pass

        # delta_constr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max,
                                  delta_constr=perr)
            pass
        pass

        # num_pixels
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max,
                                  num_pixels=perr)
            pass
        pass

    def test_invalid_postive_scalar_integer(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.psilist

        # Lij
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=perr,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # fwc
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=perr,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # fwc_em
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=perr,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # Nmin
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=perr, Nmax=self.Nmax,
                                  tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # Nmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=perr,
                                  tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # Nem
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max,
                                  Nem=perr)
            pass
        pass

    def test_invalid_real_positive_scalar(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.rpslist

        # alpha0
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=perr, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # alpha1
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=perr, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # tmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=perr,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)

        # gmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=perr, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)

        # pc_ecount_max
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=perr)
            pass

        #T_factor
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max,
                                  T_factor=perr)
        pass

    def test_invalid_fluxe_bright(self):
        """Invalid inputs fail as expected"""
        perr = self.fluxe_bright*1.1
        with self.assertRaises(ValueError):
            calc_pc(target_snr=self.target_snr, fluxe=perr,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

    def test_invalid_alpha_gt_1(self):
        """Invalid inputs fail as expected"""
        perr = 1.1
        with self.assertRaises(ValueError):
            calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=perr, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

        perr = 1.1
        with self.assertRaises(ValueError):
            calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=perr, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

    def test_invalid_nmax_lt_nmin(self):
        """Invalid inputs caught"""
        nmax = 10
        for nmin in [20, 10]:

            with self.assertRaises(ValueError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=nmin, Nmax=nmax, tmin=self.tmin,
                            tmax=self.tmax, gmax=self.gmax,
                            overhead=self.overhead,
                            pc_ecount_max=self.pc_ecount_max)
            pass

    def test_invalid_tmax_lt_tmin(self):
        """Invalid inputs caught"""
        tmax = 10
        for tmin in [20, 10]:

            with self.assertRaises(ValueError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                            tmax=tmax, gmax=self.gmax,
                            overhead=self.overhead,
                            pc_ecount_max=self.pc_ecount_max)
            pass

    def test_invalid_gain(self):
        """Invalid inputs caught"""

        for perr in [0.9, 1]:
            with self.assertRaises(ValueError):
                calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                tmin=self.tmin, tmax=self.tmax, gmax=perr,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max)
                pass
        pass

    def test_invalid_opt_choice(self):
        """Invalid inputs caught"""
        perr = 2

        with self.assertRaises(ValueError):
            calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max,
                              opt_choice=perr)
            pass

    def test_invalid_full_wells(self):
        """Check the two full well cases raise expected exceptions"""
        fluxe = 5
        fluxe_bright = 10
        darke = 10
        cic = 100
        tmin = 10
        alpha0 = 1
        alpha1 = 1
        fwc = 10
        fwc_em = 10

        # per-pixel
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=fluxe_bright, darke=darke,
                              cic=cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=alpha0, fwc=fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

        # gain register
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=fluxe_bright, darke=darke,
                              cic=cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=alpha1, fwc_em=fwc_em, Nmin=self.Nmin,
                              Nmax=self.Nmax, tmin=tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

        pass

    def invalid_g_lb(self):
        """Check that g_lb >= gmax raises expected exception"""
        rn = self.gmax*2/5  # So that T = 5*rn > gmax
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

    def invalid_t_ub(self):
        """Check that t_lb >= t_ub raises expected exception"""
        # So that t_pcmax=tmin and thus t_ub = tmin:
        fluxe = self.pc_ecount_max/self.tmin
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)


class TestCalcPCFixedN(unittest.TestCase):
    """
    Tests for calc_pc_fixed_N function.
    """

    def setUp(self):
        self.target_snr = 1 # unitless
        self.fluxe = 0.01 # e-/sec
        self.fluxe_bright = 1 # e-sec
        self.darke = 8.33e-4 # e-/sec
        self.cic = 0.02 # e-
        self.rn = 160 # e-
        self.X = 5e4 # hits/m^2/sec
        self.a = 1.69e-10 # m^2/pixel
        self.Lij = 512 # pixels
        self.alpha0 = 0.75 # unitless
        self.fwc = 50000 # e-
        self.alpha1 = 0.75 # unitless
        self.fwc_em = 90000 # e-
        self.N = 1 # frames
        self.tmin = 0.264 # seconds/frame
        self.tmax = 6300. # seconds/frame
        self.gmax = 5000 # unitless
        self.overhead = 0 # seconds
        self.pc_ecount_max = 0.1 # e-
        self.T_factor = 5 # unitless; # of read noise std devs for threshold
        self.opt_choice = 0 # 0 or 1
        self.n = 4 # number of standard deviations below fwc
        self.Nem = 604 # number of gain register cells
        self.tol = 1e-30 # tolerance level used by optimizations
        self.delta_constr = 1e-4 # constraints satisfied up to this fraction

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """good inputs for opt_choice = 0 complete without an exception"""
        calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          N=self.N, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=self.pc_ecount_max)
        pass

    def test_opt_choice_success(self):
        """good inputs for opt_choice=1 complete without exception"""
        calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          N=self.N, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=self.pc_ecount_max, opt_choice=1)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_first_both(self, mock_min):
        """good inputs for opt_choice = 0 complete without an exception.
        res1_cond and res2_cond both true.  Ends after first optimization
         with res2 results since it has the shorter total exposure time."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools, and they definitely
        # satisfy rail, em_rail, and target_snr constraints in the optimization
        _g1 = 5*self.rn
        _t1 = self.tmin+1 # so N2*tfr2 will be better
        _t2 = self.tmin

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g1, _t2])
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_pc_fixed_N(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max)
        self.assertTrue(out[-1] == 0)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t2)
        self.assertTrue(out[2] == self.N)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_first_one(self, mock_min):
        """good inputs for opt_choice = 0 complete without an exception.
        res1_cond true while res2_cond false.  Ends after first optimization
         with res1 results."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools, and they definitely
        # satisfy rail, em_rail, and target_snr constraints in the optimization
        _g1 = 5*self.rn
        _t1 = self.tmin+1
        _g2 = self.fwc_em/(self.fluxe*self.tmin) # ensures res2_cond is false
        _t2 = self.tmin # would give a shorter N*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _t2])
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_pc_fixed_N(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max)
        self.assertTrue(out[-1] == 0)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == self.N)
        pass

    @patch('scipy.optimize.minimize')
    def test_end_after_second_both(self, mock_min):
        """bad inputs for opt_choice = 0.
        Ends after second optimization due to constraint disagreement in first
        optimization scheme.  res3_cond and res4_cond both true.  Gives res4
        outputs due to bigger SNR."""

        _g1 = 5*self.rn
        _t1 = self.tmin

        # constraints not met (can't reach target SNR of 1e15)
        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g1, _t1])

        #now for 2nd optimzation scheme, let the constraints be met
        _t4 = _t1+1  # gives bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t4])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_pc_fixed_N(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g1)
        self.assertTrue(out[1] == _t4)
        self.assertTrue(out[2] == self.N)
        pass


    @patch('scipy.optimize.minimize')
    def test_end_after_second_one(self, mock_min):
        """bad inputs for opt_choice = 0.
        Ends after 2nd optimization due to constraint disagreement in first
        optimization scheme.  res3_cond true while res4_cond false.  Gives res3
        results."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools
        _g1 = self.fwc_em/(self.fluxe*self.tmin) # res1_cond false
        _t1 = self.tmin+1
        _g2 = self.fwc_em/(self.fluxe*self.tmin) # ensures res2_cond is false
        _t2 = self.tmin # would give a shorter N2*tfr2

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _t2])

        _g3 = 5*self.rn
        _g4 = self.fwc_em/(self.fluxe*self.tmin) # breaks em_rail constraint,
        #even though it would give bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _t2])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        out = calc_pc_fixed_N(target_snr=1e-30, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == self.N)
        pass


    @patch('scipy.optimize.minimize')
    def test_both_fail(self, mock_min):
        """For opt_choice = 0, the 2nd optimization scheme fails due to
        constraint disagreement, and an exception is raised."""

         # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools
        _g1 = self.fwc_em/(self.fluxe*self.tmin) # res1_cond false
        _t1 = self.tmin
        _g2 = self.fwc_em/(self.fluxe*self.tmin) # ensures res2_cond is false
        _t2 = self.tmin

        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([_g1, _t1])
        _res2 = scipy.optimize.OptimizeResult()
        _res2['x'] = np.array([_g2, _t2])

        #now for 2nd optimzation scheme, let the constraints disagree
        _g3 = self.fwc_em/(self.fluxe*self.tmin) # breaks em_rail constraint
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _t1])

        mock_min.side_effect = [_res1, _res2, _res3, _res4]

        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N, tmin=self.tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_both(self, mock_min):
        """For opt_choice = 1, both res3_cond and res4_cond are true.  Gives
        res4 outputs due to bigger SNR."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = 5*self.rn
        _t3 = self.tmin
        _t4 = self.tmin+1 #gives bigger SNR
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _t4])

        mock_min.side_effect = [_res3, _res4, _res3]  # should stop after
                                                      # second mock

        # target_snr should have no bearing when opt_choice=1
        out = calc_pc_fixed_N(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max,
                                opt_choice=1)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t4)
        self.assertTrue(out[2] == self.N)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_one(self, mock_min):
        """For opt_choice = 1, res3_cond is true and res4_cond is false.  Gives
        res3 outputs."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = 5*self.rn
        _t3 = self.tmin
        _g4 = self.fwc_em/(self.fluxe*self.tmin) # breaks em_rail condition,
        #even though gives bigger snr
        _t4 = self.tmin+1 # would give the bigger snr
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _t4])

        mock_min.side_effect = [_res3, _res4, _res4]  # should stop after
                                                      # second mock

        # target_snr should have no bearing when opt_choice=1
        out = calc_pc_fixed_N(target_snr=1e15, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N, tmin=self.tmin,
                                tmax=self.tmax, gmax=self.gmax,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max,
                                opt_choice=1)
        self.assertTrue(out[-1] == 1)
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t3)
        self.assertTrue(out[2] == self.N)
        pass

    @patch('scipy.optimize.minimize')
    def test_choice1_both_fail(self, mock_min):
        """For opt_choice = 1, the 2nd optimization scheme fails due to
        constraint disagreement, and an exception is raised."""

        # these are made-up inputs for the mock; they aren't the actual outputs
        # of the optimizer when running excam_tools.
        _g3 = self.fwc_em/(self.fluxe*self.tmin)  # break em_rail condition
        _t3 = self.tmin
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t3])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g3, _t3])

        mock_min.side_effect = [_res3, _res4, _res4]  # should stop after
                                                      # second mock

        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N, tmin=self.tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max,
                              opt_choice=1)
        pass

    def test_successful_run_multiple_pixels(self):
        '''Successful run for values of num_pixels other than the default.
        Other checks comparing multi-pixel SNR with single-pixel SNR done in
        ut_cgi_eetc.py.'''
        values = [15, 0.5, 3.3] # non-integer values should work, too
        for val in values:
            calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright, darke=self.darke,
                              cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N, tmin=self.tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max,
                              num_pixels=val)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""

        check_list = ut_check.rnslist

        # target_snr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=perr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # fluxe
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=perr,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # fluxe_bright
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=perr,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # darke
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright, darke=perr,
                                  cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                                  Lij=self.Lij, alpha0=self.alpha0,
                                  fwc=self.fwc, alpha1=self.alpha1,
                                  fwc_em=self.fwc_em,
                                  N=self.N, tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # cic
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=perr, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # rn
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=perr,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # X
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=perr, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # a
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=perr, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # tmin
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N, tmin=perr,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # overhead
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N, tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=perr,
                                  pc_ecount_max=self.pc_ecount_max)
            pass


        # n
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max, n=perr)
            pass

        # tol
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max, tol=perr)
            pass

        # delta_constr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max,
                                  delta_constr=perr)
            pass
        pass

        # num_pixels
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max,
                                  num_pixels=perr)
            pass
        pass

    def test_invalid_positive_scalar_integer(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.psilist

        # Lij
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=perr,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # fwc
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=perr,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # fwc_em
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=perr,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # N
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=perr,
                                  tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # Nem
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin,
                                  tmax=self.tmax, gmax=self.gmax,
                                  overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max,
                                  Nem=perr)
            pass
        pass

    def test_invalid_real_positive_scalar(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.rpslist

        # alpha0
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=perr, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # alpha1
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=perr, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # tmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=perr,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)

        # gmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=perr, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)

        # pc_ecount_max
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=perr)
            pass

        #T_factor
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  N=self.N,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max,
                                  T_factor=perr)
        pass

    def test_invalid_fluxe_bright(self):
        """Invalid inputs fail as expected"""
        perr = self.fluxe_bright*1.1
        with self.assertRaises(ValueError):
            calc_pc_fixed_N(target_snr=self.target_snr, fluxe=perr,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

    def test_invalid_alpha_gt_1(self):
        """Invalid inputs fail as expected"""
        perr = 1.1
        with self.assertRaises(ValueError):
            calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=perr, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

        perr = 1.1
        with self.assertRaises(ValueError):
            calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=perr, fwc_em=self.fwc_em,
                              N=self.N,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)


    def test_invalid_tmax_lt_tmin(self):
        """Invalid inputs caught"""
        tmax = 10
        for tmin in [20, 10]:

            with self.assertRaises(ValueError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            N=self.N, tmin=tmin,
                            tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                            pc_ecount_max=self.pc_ecount_max)
            pass

    def test_invalid_gain(self):
        """Invalid inputs caught"""

        for perr in [0.9, 1]:
            with self.assertRaises(ValueError):
                calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                N=self.N,
                                tmin=self.tmin, tmax=self.tmax, gmax=perr,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max)
                pass
        pass

    def test_invalid_opt_choice(self):
        """Invalid inputs caught"""
        perr = 2

        with self.assertRaises(ValueError):
            calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max,
                              opt_choice=perr)
            pass

    def test_invalid_full_wells(self):
        """Check the two full well cases raise expected exceptions"""
        fluxe = 5
        fluxe_bright = 10
        darke = 10
        cic = 100
        tmin = 10
        alpha0 = 1
        alpha1 = 1
        fwc = 10
        fwc_em = 10

        # per-pixel
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_fixed_N(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=fluxe_bright, darke=darke,
                              cic=cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=alpha0, fwc=fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N, tmin=tmin,
                              tmax=self.tmax, gmax=self.gmax,
                              overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

        # gain register
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_fixed_N(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=fluxe_bright, darke=darke,
                              cic=cic, rn=self.rn, X=self.X, a=self.a,
                              Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=alpha1, fwc_em=fwc_em,
                              N=self.N, tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

        pass

    def invalid_g_lb(self):
        """Check that g_lb >= gmax raises expected exception"""
        rn = self.gmax*2/5  # So that T = 5*rn > gmax
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_fixed_N(target_snr=self.target_snr, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

    def invalid_t_ub(self):
        """Check that t_lb >= t_ub raises expected exception"""
        # So that t_pcmax=tmin and thus t_ub = tmin:
        fluxe = self.pc_ecount_max/self.tmin
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_fixed_N(target_snr=self.target_snr, fluxe=fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              N=self.N,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

class TestCalcPCGainFixedNTime(unittest.TestCase):
    """
    Tests for calc_pc_gain_fixed_Ntime function.
    """

    def setUp(self):
        self.t_tot = 200 # s
        self.fluxe = 0.01 # e-/sec
        self.fluxe_bright = 0.1 # e-sec
        self.darke = 8.33e-4 # e-/sec
        self.cic = 0.02 # e-
        self.rn = 160 # e-
        self.X = 5e4 # hits/m^2/sec
        self.a = 1.69e-10 # m^2/pixel
        self.Lij = 512 # pixels
        self.alpha0 = 0.75 # unitless
        self.fwc = 50000 # e-
        self.alpha1 = 0.75 # unitless
        self.fwc_em = 90000 # e-
        self.Nmin = 1 # frames
        self.Nmax = 25200 # frames
        self.tmin = 0.264 # seconds/frame
        self.tmax = 6300 # seconds/frame
        self.gmax = 5000 # unitless
        self.overhead = 0 # seconds
        self.pc_ecount_max = 0.1 # e-
        self.T_factor = 5 # unitless; # of read noise std devs for threshold
        self.n = 4 # number of standard deviations below fwc
        self.Nem = 604 # number of gain register cells
        self.tol = 1e-30 # tolerance level used by optimizations
        self.delta_constr = 1e-4 # constraints satisfied up to this fraction
        self.hard_limit = True # impose N*t = t_tot exactly

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """good inputs complete without an exception. And this is a case
        where t_lb != t_ub."""
        out = calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=self.pc_ecount_max)
        # t_lb can only equal t_ub if they are tmax or tmin
        self.assertTrue(out[1] > self.tmin)
        self.assertTrue(out[1] < self.tmax)
        pass

    def test_overhead(self):
        """Test that nonzero overhead is correctly incorporated"""
        t_tot = 80
        overhead = 3
        tol = 1e-13

        out_true = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=overhead,
                          pc_ecount_max=self.pc_ecount_max, hard_limit=True)
        out_false = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=overhead,
                          pc_ecount_max=self.pc_ecount_max, hard_limit=False)

        # Hard limit enforced for True
        self.assertTrue(np.max(np.abs(t_tot - out_true[4])) < tol)
        # ...not necessarily for False
        self.assertTrue(out_false[4] >= t_tot)
        # Either way, overhead is included
        self.assertTrue(out_true[1]*out_true[2] < t_tot)
        self.assertTrue(out_false[1]*out_false[2] < t_tot)


    def test_invalid_wells(self):
        """Inputs saturate wells before getting to scipy solving functions."""
        fluxe = 5
        fluxe_bright = 10
        darke = 10
        cic = 100
        tmin = 10
        alpha0 = 1
        alpha1 = 1
        fwc = 10
        fwc_em = 10

        # per-pixel well
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=darke,
                          cic=cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=alpha0, fwc=fwc,
                          alpha1=alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=self.pc_ecount_max)

        # EM gain well
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=darke,
                          cic=cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=alpha0, fwc=self.fwc,
                          alpha1=alpha1, fwc_em=fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=self.pc_ecount_max)

    def test_invalid_t_tot(self):
        """t_tot input incompatible with Nmin, tmin, Nmax, and tmax."""
        # t_tot too big
        t_tot_big = self.Nmax*self.tmax + 1
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_gain_fixed_Ntime(t_tot=t_tot_big, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=self.pc_ecount_max)

        # t_tot too small
        Nmin = 5
        t_tot_small = Nmin*self.tmin - 1
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_gain_fixed_Ntime(t_tot=t_tot_small, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=self.pc_ecount_max)

    def invalid_g_lb(self):
        """Check that g_lb >= gmax raises expected exception"""
        rn = self.gmax*2/5  # So that T = 5*rn > gmax
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

    def invalid_t_ub(self):
        """Check that t_lb >= t_ub raises expected exception"""
        # So that t_pcmax=tmin and thus t_ub = tmin:
        fluxe = self.pc_ecount_max/self.tmin
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

    def invalid_t_tot(self):
        """Check that Nmax*t_pcmax < t_tot raises expected exception"""
        # So that Nmax*t_UB < t_tot:
        t_tot = self.Nmax*(self.pc_ecount_max/self.fluxe) + 1
        tmax = min(t_tot/self.Nmin -1, self.pc_ecount_max/self.fluxe - 1)
        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

    @patch('scipy.optimize.minimize')
    def test_hard_limit_one(self, mock_min):
        """One passes:  res3 passes b/c of bigger snr, while res4 saturates."""
        pc_ecount_max = self.fwc_em
        T_factor = 0.1
        Nmax = 49
        tmax = 120
        _g3 = 10 # bigger than _g4, so gives bigger snr
        _t1 = self.t_tot/(Nmax - 2) # so N will be integer
        _g4 = self.fwc_em #1e-30 # saturates
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _t1])

        mock_min.side_effect = [_res3, _res4]

        out = calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                    fluxe_bright=self.fluxe_bright, darke=self.darke,
                    cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                    Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                    alpha1=self.alpha1, fwc_em=self.fwc_em,
                    Nmin=self.Nmin, Nmax=Nmax, tmin=self.tmin,
                    tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                    pc_ecount_max=pc_ecount_max, T_factor=T_factor)

        # since N integer, neither t nor g will be adjusted
        self.assertTrue(out[0] == _g3)
        self.assertTrue(out[1] == _t1)
        self.assertTrue(out[2] == Nmax - 2)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_both(self, mock_min):
        """Both pass:  constraints met for both res3 and res4, but res4 wins
        b/c of bigger snr."""
        pc_ecount_max = self.fwc_em # eliminate this usual constraint
        T_factor = 0.01
        tmax = 120
        Nmax = 49
        _g4 = 10 # bigger than _g3, so gives bigger snr
        _t1 = self.t_tot/(Nmax - 2) # so N will be integer
        # bigger time but smaller N, and N drives SNR more than t here
        _t2 = self.t_tot/(Nmax - 3)
        _g3 = 5
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g3, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g4, _t2])

        mock_min.side_effect = [_res3, _res4]

        out = calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                        fluxe_bright=self.fluxe_bright, darke=self.darke,
                        cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                        Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                        alpha1=self.alpha1, fwc_em=self.fwc_em,
                        Nmin=self.Nmin, Nmax=Nmax, tmin=self.tmin,
                        tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                        pc_ecount_max=pc_ecount_max, T_factor=T_factor)
        # since N integer, neither t nor g will not be adjusted
        self.assertTrue(out[0] == _g4)
        self.assertTrue(out[1] == _t2)
        self.assertTrue(out[2] == Nmax - 3)

    @patch('scipy.optimize.minimize')
    def test_both_fail_init(self, mock_min):
        """inputs pass initial well checks and give an exception for a case
        where t_lb != t_ub (since we use the same numbers as previous test).
        res*_cond_init are False.  This is the case regardless of hard_limit's
        value since this check happens before hard_limit is used."""
        _g1 = 1e30 # saturates
        _t1 = self.t_tot/2
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        _res4 = scipy.optimize.OptimizeResult()
        _res4['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res4]

        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=self.pc_ecount_max)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_Nmin(self, mock_min):
        """The values of N found for res3 and/or res4 are below Nmin, so
        Nmin is the output value for N."""
        _g1 = 1
        t_tot = 80
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        delta_constr = 1 # to ensure that res_*_init are passed
        #(otherwise, I could increase Nmin)
        _t1 = t_tot/(self.Nmin - 0.5) #gives N < Nmin
        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        _resn = scipy.optimize.OptimizeResult()
        _resn['x'] = np.array([_g1])
        mock_min.side_effect = [_res3, _res3, _resn, _resn]

        out = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max,
                          delta_constr=delta_constr)

        self.assertTrue(out[2] == self.Nmin)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_Nmin_fail(self, mock_min):
        """The values of N found for res3 and/or res4 are below Nmin, and
        increasing N to Nmin makes t dip below tmin.  Otherwise, other
        constraints are satisfied, as illustrated in unit test just above this
        one."""
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        tmin = self.t_tot/self.Nmin - 1
        tmax = tmin + 5 #won't mess up the satisfaction of other constraints
        _g1 = 1
        t_tot = 80
        _t1 = self.t_tot/(self.Nmin - 0.5) #gives N < Nmin
        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        # _resn = scipy.optimize.OptimizeResult()
        # _resn['x'] = np.array([_g1])
        mock_min.side_effect = [_res3, _res3]

        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                            tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                            pc_ecount_max=pc_ecount_max)

    def test_hard_limit_both_fail_final(self):
        """Both res3_cond_final and res4_cond_final fail.
        With hard_limit=False, these inputs do not fail.  The decrease in N
        in the True case increases t enough to cause saturation.  Also, these
        inputs give gsol < 1 for floor of N (before gsol is restricted to be
        between _gmin and gmax).  res*_cond_init is fine, but the final checks
        fail due to the change in t when hard_limit=True."""
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        # to allow passage through infeasibility constraints which use g_lb
        T_factor = 0.001
        rn = 100
        fluxe = 1
        fwc_em = 25000 #50000
        fwc = 60000
        fluxe_bright = 2000
        t_tot = 200
        Nmax = 70
        tmax = 6300
        # runs fine
        calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=fwc,
                          alpha1=self.alpha1, fwc_em=fwc_em,
                          Nmin=self.Nmin, Nmax=Nmax, tmin=self.tmin,
                          tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max,
                          T_factor=T_factor, hard_limit=False)

        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=fwc,
                          alpha1=self.alpha1, fwc_em=fwc_em,
                          Nmin=self.Nmin, Nmax=Nmax, tmin=self.tmin,
                          tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max,
                          T_factor=T_factor, hard_limit=True)

    @patch('scipy.optimize.minimize')
    def test_hard_limit_middle_fail(self, mock_min):
        """The 'middle' elif case (floor and ceiling of N both b/w Nmin
        and Nmax) where the adjusted t with either ceiling or floor is outside
        of t bounds."""
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        _g1 = 1
        t_tot = 80
        _t1 = 1.98 # gives N as t_tot/_t1 = 40.4
        tmin = 1.952 #just above 80/41
        tmax = 1.99 # just below 80/40

        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res3, _res3, _res3]

        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_middle_tmax(self, mock_min):
        """The 'middle' elif case (floor and ceiling of N both b/w Nmin
        and Nmax) where the adjusted t with floor is bigger than tmax."""
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        T_factor = 0.01
        _g1 = 1
        t_tot = 80
        _t1 = 1.98 # gives N as t_tot/_t1 = 40.4
        tmin = 1
        tmax = 1.99 # just below 80/40

        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        _resn = scipy.optimize.OptimizeResult()
        _resn['x'] = np.array([self.gmax])
        mock_min.side_effect = [_res3, _res3, _resn, _resn]

        out = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                        fluxe_bright=self.fluxe_bright, darke=self.darke,
                        cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                        Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                        alpha1=self.alpha1, fwc_em=self.fwc_em,
                        Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                        tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                        pc_ecount_max=pc_ecount_max, T_factor=T_factor)

        self.assertTrue(out[2] == 41)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_middle_tmin(self, mock_min):
        """The 'middle' elif case (floor and ceiling of N both b/w Nmin
        and Nmax) where the adjusted t with ceil is smaller than tmin."""
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        T_factor = 0.01
        _g1 = 1
        t_tot = 80
        _t1 = 1.98 # gives N as t_tot/_t1 = 40.4
        tmin = 1.952 #just above 80/41
        tmax = 3

        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        _resn = scipy.optimize.OptimizeResult()
        _resn['x'] = np.array([_g1])

        mock_min.side_effect = [_res3, _res3, _resn, _resn]

        out = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                        fluxe_bright=self.fluxe_bright, darke=self.darke,
                        cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                        Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                        alpha1=self.alpha1, fwc_em=self.fwc_em,
                        Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                        tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                        pc_ecount_max=pc_ecount_max, T_factor=T_factor)

        self.assertTrue(out[2] == 40)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_Nmax(self, mock_min):
        """The values of N found for res3 and/or res4 are above Nmax, so
        Nmax is the output value for N."""
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        _g1 = 1
        t_tot = 80
        _t1 = t_tot/(self.Nmax + 0.5) #gives N > Nmax
        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])
        _resn = scipy.optimize.OptimizeResult()
        _resn['x'] = np.array([_g1])
        mock_min.side_effect = [_res3, _res3, _resn, _resn]

        out = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max)

        self.assertTrue(out[2] == self.Nmax)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_Nmax_fail(self, mock_min):
        """The values of N found for res3 and/or res4 are above Nmax, and
        decreasing N to Nmax makes t rise above tmax.  Otherwise, other
        constraints are satisfied, as illustrated in unit test just above this
        one."""
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        tmax = self.t_tot/self.Nmin + 1
        tmin = tmax - 5 #won't mess up the satisfaction of other constraints
        _g1 = 1
        t_tot = 80
        _t1 = self.t_tot/(self.Nmax + 0.5) #gives N > Nmax
        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res3, _res3, _res3]

        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                            tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                            pc_ecount_max=pc_ecount_max)

    def test_hard_limit_t_snr(self):
        """Illustrate case where time drives the SNR more than N, and floor
        instead of ceiling of N is optimal. This also tests case where
        N ceiling and floor are contained within Nmin and Nmax."""
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        # to allow passage through infeasibility constraints which use g_lb
        T_factor = 0.01
        fluxe = 0.12
        fluxe_bright = 1000
        rn = 100
        t_tot = 14760
        Nmax = 25200
        tmax = 6300
        tmin = 1
        delta_constr = 1e-2

        out_true = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max,
                          T_factor=T_factor, hard_limit=True,
                          delta_constr=delta_constr)

        out_false = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max,
                          T_factor=T_factor, hard_limit=False,
                          delta_constr=delta_constr)
        # in this case, the hard_limit rounded N down to get a bigger snr
        self.assertTrue(out_true[2] == out_false[2] - 1)
        # and the time for the True case is bigger since N was rounded down
        self.assertTrue(out_true[1] >= out_false[1])

    @patch('scipy.optimize.minimize')
    def test_not_hard_limit_Nmin(self, mock_min):
        """The values of N found for res3 and/or res4 are below Nmin, so
        the ceiling of N, Nmin, is the output value. For hard_limit=False."""
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        # to allow passage through infeasibility constraints which use g_lb
        T_factor = 0.01
        _g1 = 1
        t_tot = 80
        delta_constr = 1 # to ensure that res_*_init are passed
        #(otherwise, I could increase Nmin)
        _t1 = t_tot/(self.Nmin - 0.5) #gives N < Nmin
        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res3, _res3, _res3]

        out = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max,
                          T_factor=T_factor, hard_limit=False,
                          delta_constr=delta_constr)

        self.assertTrue(out[2] == self.Nmin)
        pass

    @patch('scipy.optimize.minimize')
    def test_hard_limit_Nmax_2(self, mock_min):
        """The values of N found for res3 and/or res4 are above Nmax, so
        Nmax is the output value for N."""
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        # to allow passage through infeasibility constraints which use g_lb
        T_factor = 0.01
        _g1 = 1
        t_tot = 80
        delta_constr = 1 # to ensure that res_*_init are passed
        #(otherwise, I could increase Nmin)
        _t1 = t_tot/(self.Nmax + 0.5) #gives N > Nmax
        # use this for both res3 and res4
        _res3 = scipy.optimize.OptimizeResult()
        _res3['x'] = np.array([_g1, _t1])

        mock_min.side_effect = [_res3, _res3, _res3, _res3]

        out = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max,
                          T_factor=T_factor, hard_limit=False,
                          delta_constr=delta_constr)

        self.assertTrue(out[2] == self.Nmax)
        pass

    def test_t_lb_equals_t_ub_no_gsol(self):
        """Exercise the case where emrail has no
        solution for g that saturates."""
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        cic = 0
        darke = 0
        fluxe_bright = 0.01
        fluxe = 0.01
        t_tot = 3600
        tmin = 1
        tmax = 120
        Nmin = 1
        Nmax = 30

        out = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=darke,
                          cic=cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max)
        # would be one of these, gmax or g_lb
        self.assertTrue(out[0] == self.gmax or out[0] == self.T_factor*self.rn)


    def test_t_lb_equals_t_ub(self):
        """inputs such that t_lb=t_ub with gsol b/w _gmin and gmax."""
        # set up a case where t_ub = tmax < t_tot/Nmin and t_lb = tmax = t_ub
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        t_tot = 3600
        tmin = 1
        tmax = 120
        Nmin = 1
        Nmax = 30

        out = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max)

        self.assertTrue(out[0] > 1)
        self.assertTrue(out[0] < self.gmax)
        self.assertTrue(out[1] == tmax)
        self.assertTrue(out[2] == t_tot/tmax)
        self.assertTrue(out[-1] == t_tot)
        pass

    def test_t_lb_equals_t_ub_gmin(self):
        """inputs such that t_lb=t_ub with gsol = _gmin."""
        # set up a case where t_ub = tmax < t_tot/Nmin and t_lb = tmax = t_ub
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        # to allow passage through infeasibility constraints which use g_lb
        T_factor = 0.001
        t_tot = 3600
        tmin = 1
        tmax = 120
        Nmin = 1
        Nmax = 30
        fwc = self.fwc_em
        n = 0
        # set fluxe_bright big enough so that gsol = 1 (manipulate emrail)
        fluxe_bright = ((self.alpha1*self.fwc_em - self.cic - self.darke*tmax)
                        /tmax)

        out = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max, T_factor=T_factor, n=n)

        self.assertTrue(out[0] == 1)
        self.assertTrue(out[1] == tmax)
        self.assertTrue(out[2] == t_tot/tmax)
        self.assertTrue(out[-1] == t_tot)
        pass

    def test_t_lb_equals_t_ub_gmax(self):
        """inputs such that t_lb=t_ub with gsol > gmax."""
        # set up a case where t_ub = tmax < t_tot/Nmin and t_lb = tmax = t_ub
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        # to allow passage through infeasibility constraints which use g_lb
        T_factor = 0.01
        t_tot = 3600
        tmin = 1
        tmax = 120
        Nmin = 1
        Nmax = 30
        fluxe = 0.01
        n = 0
        # set fluxe_bright small enough so that gsol > gmax
        # (manipulate emrail with g = self.gmax+10)
        fluxe_bright = ((self.alpha1*self.fwc_em/(self.gmax+10) - self.cic -
                        self.darke*tmax)/tmax)

        out = calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=fluxe,
                          fluxe_bright=fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max, T_factor=T_factor, n=n)

        self.assertTrue(out[0] == self.gmax) #g should be truncated to gmax
        self.assertTrue(out[1] == tmax)
        self.assertTrue(out[2] == t_tot/tmax)
        self.assertTrue(out[-1] == t_tot)
        pass

    @patch('scipy.optimize.minimize')
    @patch('scipy.optimize.fsolve')
    def test_t_lb_equals_t_ub_exception(self, mock_fsolve, mock_minimize):
        """inputs such that t_lb=t_ub with gsol b/w _gmin and gmax, but
        simulate a wonky fsolve with a gsol that saturates and causes an
        exception."""
        # set up a case where t_ub = tmax < t_tot/Nmin and t_lb = tmax = t_ub
        pc_ecount_max = self.fwc_em # to eliminate this usual constraint
        t_tot = 3600
        tmin = 1
        tmax = 120
        Nmin = 1
        Nmax = 30
        fwc_em = 20000
        fwc = 50000
        # output expected in this form
        _fsolve = (np.array([4000]), {}, 1, '')
        _res1 = scipy.optimize.OptimizeResult()
        _res1['x'] = np.array([4000])
        # one fsolve for max g from emrail constraint, the next from the g
        #that gives local max in SNR
        mock_fsolve.side_effect = [_fsolve]
        mock_minimize.side_effect = [_res1, _res1, _res1, _res1]

        with self.assertRaises(EXCAMOptimizeException):
            calc_pc_gain_fixed_Ntime(t_tot=t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=fwc,
                          alpha1=self.alpha1, fwc_em=fwc_em,
                          Nmin=Nmin, Nmax=Nmax, tmin=tmin,
                          tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                          pc_ecount_max=pc_ecount_max)
        pass

    def test_successful_run_multiple_pixels(self):
        '''Successful run for values of num_pixels other than the default.
        Other checks comparing multi-pixel SNR with single-pixel SNR done in
        ut_cgi_eetc.py.'''
        values = [15, 0.5, 3.3] # non-integer values should work, too
        for val in values:
            calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                          fluxe_bright=self.fluxe_bright, darke=self.darke,
                          cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                          Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                          alpha1=self.alpha1, fwc_em=self.fwc_em,
                          Nmin=self.Nmin, Nmax=self.Nmax, tmin=self.tmin,
                          tmax=self.tmax, gmax=self.gmax,
                          overhead=self.overhead,
                          pc_ecount_max=self.pc_ecount_max, num_pixels=val)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""

        check_list = ut_check.rnslist

        # t_tot
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=perr, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # fluxe
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=perr,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # fluxe_bright
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=perr,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # darke
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=perr, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # cic
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=perr, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # rn
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=perr,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # X
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=perr, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # a
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=perr, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # tmin
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=perr, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # overhead
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=perr,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # n
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max, n=perr)
            pass

        # tol
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max, tol=perr)
            pass

        # delta_constr
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max,
                                  delta_constr=perr)
            pass
        pass

        # num_pixels
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max,
                                  num_pixels=perr)
            pass
        pass

    def test_invalid_postive_scalar_integer(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.psilist

        # Lij
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=perr,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # fwc
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=perr,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # fwc_em
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=perr,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # Nmin
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=perr, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # Nmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=perr,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # Nem
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max, Nem=perr)
            pass
        pass

    def test_invalid_real_positive_scalar(self):
        """invalid inputs caught as expected"""

        check_list = ut_check.rpslist

        # alpha0
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=perr, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # alpha1
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=perr, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # tmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=perr,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # gmax
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=perr, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max)
            pass

        # pc_ecount_max
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=perr)
            pass

        # T_factor
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max,
                                  T_factor=perr)
            pass

    def test_invalid_hard_limit(self):
        """invalid inputs caught as expected"""

        check_list = [2, 'foo', -3.4]

        # hard_limit
        for perr in check_list:
            with self.assertRaises(TypeError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                  fluxe_bright=self.fluxe_bright,
                                  darke=self.darke, cic=self.cic, rn=self.rn,
                                  X=self.X, a=self.a, Lij=self.Lij,
                                  alpha0=self.alpha0, fwc=self.fwc,
                                  alpha1=self.alpha1, fwc_em=self.fwc_em,
                                  Nmin=self.Nmin, Nmax=self.Nmax,
                                  tmin=self.tmin, tmax=self.tmax,
                                  gmax=self.gmax, overhead=self.overhead,
                                  pc_ecount_max=self.pc_ecount_max,
                                  hard_limit=perr)
            pass

    def test_invalid_fluxe_bright(self):
        """Invalid inputs fail as expected"""
        perr = self.fluxe_bright*1.1
        with self.assertRaises(ValueError):
            calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=perr,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)


    def test_invalid_t_tot_overhead(self):
        """Invalid inputs fail as expected"""
        overhead = self.t_tot + 0.1
        with self.assertRaises(ValueError):
            calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=overhead,
                              pc_ecount_max=self.pc_ecount_max)


    def test_invalid_alpha_gt_1(self):
        """Invalid inputs fail as expected"""
        perr = 1.1
        with self.assertRaises(ValueError):
            calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=perr, fwc=self.fwc,
                              alpha1=self.alpha1, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

        perr = 1.1
        with self.assertRaises(ValueError):
            calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                              fluxe_bright=self.fluxe_bright,
                              darke=self.darke, cic=self.cic, rn=self.rn,
                              X=self.X, a=self.a, Lij=self.Lij,
                              alpha0=self.alpha0, fwc=self.fwc,
                              alpha1=perr, fwc_em=self.fwc_em,
                              Nmin=self.Nmin, Nmax=self.Nmax,
                              tmin=self.tmin, tmax=self.tmax,
                              gmax=self.gmax, overhead=self.overhead,
                              pc_ecount_max=self.pc_ecount_max)

    def test_invalid_nmax_lt_nmin(self):
        """Invalid inputs caught"""
        nmax = 10
        for nmin in [20, 10]:

            with self.assertRaises(ValueError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=nmin, Nmax=nmax, tmin=self.tmin,
                            tmax=self.tmax, gmax=self.gmax,
                            overhead=self.overhead,
                            pc_ecount_max=self.pc_ecount_max)
        pass

    def test_invalid_tmax_lt_tmin(self):
        """Invalid inputs caught"""
        tmax = 10
        for tmin in [10, 20]:

            with self.assertRaises(ValueError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                            fluxe_bright=self.fluxe_bright, darke=self.darke,
                            cic=self.cic, rn=self.rn, X=self.X, a=self.a,
                            Lij=self.Lij, alpha0=self.alpha0, fwc=self.fwc,
                            alpha1=self.alpha1, fwc_em=self.fwc_em,
                            Nmin=self.Nmin, Nmax=self.Nmax, tmin=tmin,
                            tmax=tmax, gmax=self.gmax, overhead=self.overhead,
                            pc_ecount_max=self.pc_ecount_max)
        pass

    def test_invalid_gain(self):
        """Invalid inputs caught"""

        for perr in [0.9, 1]:
            with self.assertRaises(ValueError):
                calc_pc_gain_fixed_Ntime(t_tot=self.t_tot, fluxe=self.fluxe,
                                fluxe_bright=self.fluxe_bright,
                                darke=self.darke, cic=self.cic, rn=self.rn,
                                X=self.X, a=self.a, Lij=self.Lij,
                                alpha0=self.alpha0, fwc=self.fwc,
                                alpha1=self.alpha1, fwc_em=self.fwc_em,
                                Nmin=self.Nmin, Nmax=self.Nmax,
                                tmin=self.tmin, tmax=self.tmax, gmax=perr,
                                overhead=self.overhead,
                                pc_ecount_max=self.pc_ecount_max)
                pass
        pass


if __name__ == '__main__':
    unittest.main()
