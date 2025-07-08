# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit test suite for sequence_tools module.
"""
import unittest

import os
import numpy as np
from scipy.special import erf
import scipy.integrate as integrate
from astropy.io import fits

import eetc
from eetc.sequence_tools import (get_peak_flux_ratio_pix,
                                 get_num_pixels_and_fraction, _rot_gauss_spot)
import eetc.util.ut_check as ut_check

class TestGetPeakFluxRatioPix(unittest.TestCase):
    """
    Unit tests for get_peak_flux_ratio_pix function.
    """

    def test_success(self):
        """Given valid inputs, completes without error"""
        get_peak_flux_ratio_pix(np.eye(3))
        pass


    def test_invalid_array(self):
        """Fails as expected on invalid inputs"""
        perrlist = [0, 1, (1,), None, 'txt', -1.5j, # not array
                    np.ones((2,)), np.ones((2, 2, 2)), # not 2D
                    1j*np.ones((3, 3)), # not real-valued
                    -1*np.ones((3, 3)), # no positives
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                get_peak_flux_ratio_pix(perr)
                pass
            pass
        pass


    def test_all_zeros(self):
        """Fails as expected on invalid inputs"""
        with self.assertRaises(ValueError):
            get_peak_flux_ratio_pix(np.zeros((3, 3)))
            pass
        pass


    def test_all_nans(self):
        """Fails as expected on invalid inputs"""
        with self.assertRaises(ValueError):
            get_peak_flux_ratio_pix(np.nan*np.ones((3, 3)))
            pass
        pass

    def test_infinite(self):
            """Fails as expected on invalid inputs"""
            with self.assertRaises(TypeError):
                ar = np.ones((3, 3))
                ar[1, 1] = np.inf
                get_peak_flux_ratio_pix(ar)
                pass
            pass


    def test_all_zeros_and_nans(self):
        """Fails as expected on invalid inputs"""
        test = np.zeros((3, 3))
        test[0, 0] = np.nan
        test[-1, -1] = np.nan

        with self.assertRaises(ValueError):
            get_peak_flux_ratio_pix(test)
            pass
        pass


    def test_uniform_succeed(self):
        """Uniform array that isn't 0 or NaN works fine"""
        tol = 1e-13

        nrow = 5
        ncol = 13

        out = get_peak_flux_ratio_pix(np.ones((nrow, ncol)))

        self.assertTrue(np.abs(out - 1/(nrow*ncol)) < tol)

        pass


    def test_uniform_succeed_with_nan(self):
        """Uniform array that has NaNs but not all NaNs works fine"""
        tol = 1e-13

        nrow = 5
        ncol = 13

        test = np.ones((nrow, ncol))
        test[0, 0] = np.nan
        test[-1, -1] = np.nan

        out = get_peak_flux_ratio_pix(test)

        self.assertTrue(np.abs(out - 1/(nrow*ncol - 2)) < tol)

        pass


    def test_nonuniform_exact(self):
        """Nonuniform array works as expected"""
        tol = 1e-13

        tarray = np.asarray([[0, 0, 0, 0, 0],
                             [0, 1, 4, 1, 0],
                             [0, 4, 9, 4, 0],
                             [0, 1, 4, 1, 0],
                             [0, 0, 0, 0, 0]])
        goal = 9/(9 + 4*4 + 1*4)

        out = get_peak_flux_ratio_pix(tarray)
        self.assertTrue(np.abs(out - goal) < tol)

        pass


    def test_nonuniform_exact_with_nans(self):
        """Nonuniform array works as expected"""
        tol = 1e-13

        tarray = np.asarray([[0, 0, 0, np.nan, 0],
                             [0, 1, 4, -1, 0],
                             [0, 4, 9, np.nan, 0],
                             [0, np.nan, 4, 1, 0],
                             [0, 0, 0, 0, 0]])
        goal = 9/(9 + 4*3 + 1*2 - 1)

        out = get_peak_flux_ratio_pix(tarray)
        self.assertTrue(np.abs(out - goal) < tol)

        pass


    def test_zero_to_one(self):
        """Check some edge cases to be sure result is in (0, 1]."""
        testlist = [
            np.asarray([[0, 0, 0],
                        [0, 5, 0],
                        [0, 0, 0]]),
            7*np.ones((1, 1)),
            np.asarray([[1, 5],
                        [2, 1000000000]]),
            ]

        for t in testlist:
            out = get_peak_flux_ratio_pix(t)
            self.assertTrue(out > 0)
            self.assertTrue(out <= 1)
            pass
        pass

class TestGetNumPixelsAndFraction(unittest.TestCase):
    """
    Unit tests for get_num_pixels_and_fraction function.
    """
    def setUp(self):
        self.ar = np.zeros((50, 50))

        x = np.arange(50)
        y = np.arange(50)
        X, Y = np.meshgrid(x, y)
        # adding a elliptic, rotated Gaussian
        offset = 100
        A =  2000
        x0 = 20
        y0 = 29
        sx = 4
        sy = 2
        theta = np.pi/2
        self.ar += _rot_gauss_spot((X,Y), offset, A, x0, y0, sx, sy, theta)
        # theoretically expected area, in pixels
        FWHMx = 2*np.sqrt(2*np.log(2))*sx
        FWHMy = 2*np.sqrt(2*np.log(2))*sy
        self.num_pixels = np.pi*0.5*FWHMx*0.5*FWHMy

        # theoretically expected fraction
        # integral over ellipsoidal FWHM:  integral in x is analytic,
        # the next one in y isn't, so do it numerically here
        def integrand(y):
            return (A*np.e**(-y**2/(2*sy**2))*np.sqrt(2*np.pi)*sx*
                       erf((-y**2/2+sy**2*np.log(2))**0.5/sy)+
                       2*offset*sx*
                       (-y**2+sy**2*np.log(4))**0.5/sy)

        resel_sig, _ = integrate.quad(integrand, -0.5*FWHMy, 0.5*FWHMy)

        self.fraction = resel_sig/np.sum(self.ar)


    def test_invalid_array(self):
        """Fails as expected on invalid inputs"""
        perrlist = [0, 1, (1,), None, 'txt', -1.5j, # not array
                    np.ones((2,)), np.ones((2, 2, 2)), # not 2D
                    1j*np.ones((3, 3)), # not real-valued
                    -1*np.ones((3, 3)), # no positives
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                get_num_pixels_and_fraction(perr)
                pass
            pass
        pass


    def test_all_zeros(self):
        """Fails as expected on invalid inputs"""
        with self.assertRaises(ValueError):
            get_num_pixels_and_fraction(np.zeros((3, 3)))
            pass
        pass


    def test_all_nans(self):
        """Fails as expected on invalid inputs"""
        with self.assertRaises(ValueError):
            get_num_pixels_and_fraction(np.nan*np.ones((3, 3)))
            pass
        pass

    def test_infinite(self):
            """Fails as expected on invalid inputs"""
            with self.assertRaises(TypeError):
                ar = np.ones((3, 3))
                ar[1, 1] = np.inf
                get_num_pixels_and_fraction(ar)
                pass
            pass


    def test_all_zeros_and_nans(self):
        """Fails as expected on invalid inputs"""
        test = np.zeros((3, 3))
        test[0, 0] = np.nan
        test[-1, -1] = np.nan

        with self.assertRaises(ValueError):
            get_num_pixels_and_fraction(test)
            pass
        pass


    def test_thresh(self):
        """Fails as expected on invalid inputs"""
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                get_num_pixels_and_fraction(self.ar, perr)

        # fails when thresh > 1
        with self.assertRaises(ValueError):
            get_num_pixels_and_fraction(self.ar, 1.1)


    def test_success(self):
        '''Successful run for PSF with no noise added.'''
        num_pixels, fraction, Rsq_adj = get_num_pixels_and_fraction(self.ar)

        # assert outputs as expected, accounting for slight disagreement
        # due to pixelation of num_pixels and fraction
        self.assertTrue(np.isclose(num_pixels, self.num_pixels, rtol=0.1))
        self.assertTrue(np.isclose(fraction, self.fraction, rtol=0.1))
        # assert Rsq_adj basically 1 since there's no noise
        self.assertTrue(np.isclose(Rsq_adj, 1))


    def test_success_noisy_PSF(self):
        """Successful run for PSF with noise added and a dark subtracted.
        It also has some nan pixels in the PSF region.
        The array was generated using emccd_detect in the
        if __name__ == '__main__' section of sequence_tools.py."""
        ar = fits.getdata(os.path.join(eetc.lib_dir, 'noised_PSF.fits'))
        num_pixels, fraction, Rsq_adj = get_num_pixels_and_fraction(ar, 0.7)

        # assert outputs as expected, accounting for slight disagreement
        # due to pixelation of num_pixels; fraction will just be off due to
        # nan-ing some pixels in PSF core

        self.assertTrue(np.isclose(num_pixels, self.num_pixels, rtol=0.1))
        self.assertTrue(np.isclose(fraction, self.fraction, rtol=0.1))
        self.assertTrue(Rsq_adj > 0.7)


    def test_Rsq_thresh_fail(self):
        """Array that doesn't get an adjusted R^2 higher than thresh."""
        ar = fits.getdata(os.path.join(eetc.lib_dir, 'noised_PSF.fits'))
        with self.assertRaises(ValueError):
            get_num_pixels_and_fraction(ar, 1)

    def test_no_PSF(self):
        '''If array has no discernable PSF, the fitting should fail.'''
        ar = np.ones((5, 5))
        with self.assertRaises(Exception):
            get_num_pixels_and_fraction(ar)



if __name__ == '__main__':
    unittest.main()
