# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit test suite for cgi_eetc module.
"""
import sys
import os
import unittest
import warnings
from pathlib import Path
import copy
import tempfile
import logging

import numpy as np
import yaml

import eetc
import eetc.util.ut_check as ut_check
from eetc.cgi_eetc import CGIEETC, _ENF
from eetc.excam_tools import (EXCAMOptimizeException, _SNR_CR_resel,
                              _SNR_CR_pc_resel)
import eetc.constants

LOCAL_PATH = eetc.lib_dir

# Lists for checking valid inputs
strlist = ut_check.strlist
rslist = ut_check.rslist
rnslist = ut_check.rnslist
rpslist = ut_check.rpslist

# Keep logger spam out of unit test results
if sys.platform.startswith('win'):
    logging.basicConfig(filename='NUL')
else:
    logging.basicConfig(filename='/dev/null')

class TestCGIEETCInit(unittest.TestCase):
    """
    Unit tests for __init__ method of CGIEETC class.
    """

    def setUp(self):
        self.local_path = LOCAL_PATH

        self.mag = 11.8
        self.phot = 'v'
        self.spt = 'G2V'
        self.ut_pointer_path = str(Path(self.local_path, 'ut_pointer.yaml'))

        self.cgieetc = CGIEETC(self.mag, self.phot, self.spt,
                               self.ut_pointer_path)
        # self.ut_cgieetc = CGIEETC(self.mag, self.phot, self.spt,
        #                           self.ut_pointer_path)

        # self.flux_grid = load_flux_grid(self.ut_cgieetc.flux_grid_path)
        # self.flux_grid_vals, self.flux_grid_spts, self.flux_grid_cfams, \
        #     self.ref_mag = self.flux_grid[self.phot]

    def test_invalid_real_scalar(self):
        """invalid inputs fail as expected"""

        # mag
        for perr in rslist:
            with self.assertRaises(TypeError):
                CGIEETC(perr, self.phot, self.spt, self.ut_pointer_path)

    def test_invalid_string(self):
        """invalid inputs fail as expected"""

        # phot
        for perr in strlist:
            with self.assertRaises(TypeError):
                CGIEETC(self.mag, perr, self.spt, self.ut_pointer_path)

        # spt
        for perr in strlist:
            with self.assertRaises(TypeError):
                CGIEETC(self.mag, self.phot, perr, self.ut_pointer_path)

        # pointer_path
        for perr in strlist:
            with self.assertRaises(TypeError):
                CGIEETC(self.mag, self.phot, self.spt, perr)

    def test_invalid_file_location(self):
        """invalid inputs fail as expected."""
        pointer_path_err = self.ut_pointer_path + '_invalid'

        with self.assertRaises(IOError):
            CGIEETC(self.mag, self.phot, self.spt, pointer_path_err)

    def test_invalid_pointer_keys(self):
        """Verify invalid pointer file keys fail as expected."""
        pointer_path_invalid = str(Path(self.local_path,
                                        'ut_pointer_invalid_key.yaml'))

        with self.assertRaises(KeyError):
            CGIEETC(self.mag, self.phot, self.spt, pointer_path_invalid)

    def test_invalid_pointer_vals(self):
        """Verify invalid pointer file values fail as expected."""
        pointer_path_invalid = str(Path(self.local_path,
                                        'ut_pointer_invalid_val.yaml'))

        with self.assertRaises(TypeError):
            CGIEETC(self.mag, self.phot, self.spt, pointer_path_invalid)

    def test_invalid_phot(self):
        """Verify invalid phot fails as expected."""
        phot_invalid = 'a'

        with self.assertRaises(KeyError):
            CGIEETC(self.mag, phot_invalid, self.spt, self.ut_pointer_path)

    def test_invalid_spt(self):
        """Verify invalid spt fails as expected."""
        spt_invalid = 'a'

        with self.assertRaises(KeyError):
            CGIEETC(self.mag, self.phot, spt_invalid, self.ut_pointer_path)


    def test_load_abs_rel(self):
        """Verify we can load from both absolute and relative paths"""

        substitute = copy.deepcopy(self.cgieetc.pointer)
        basepath = os.path.dirname(os.path.abspath(self.ut_pointer_path))
        substitute['sequences'] = os.path.join(basepath,
                                               substitute['sequences'])

        try:
            (fd, name) = tempfile.mkstemp()
            with open(name, 'w') as FILE:
                yaml.dump(substitute, FILE)
                pass
            # success if this completes without errors
            CGIEETC(self.mag, self.phot, self.spt, name)
        finally:
            os.close(fd)
            os.unlink(name)
            pass
        pass


    def test_success_allnone(self):
        """
        A CGIEETC object can be created from a pointer file of all
        none values
        """
        nonepath = str(Path(self.local_path, 'ut_pointer_allnone.yaml'))
        CGIEETC(self.mag, self.phot, self.spt, nonepath)
        pass



class TestCGIEETCCalcFluxRate(unittest.TestCase):
    """
    Unit tests for calc_flux_rate method of CGIEETC class.
    """

    def setUp(self):
        self.local_path = LOCAL_PATH

        self.mag = 11.8
        self.phot = 'v'
        self.spt = 'G2V'
        self.ut_pointer_path = str(Path(self.local_path, 'ut_pointer.yaml'))

        self.cgieetc = CGIEETC(self.mag, self.phot, self.spt,
                               self.ut_pointer_path)
        # self.ut_cgieetc = CGIEETC(self.mag, self.phot, self.spt,
        #                           self.ut_pointer_path)

        self.sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'
        self.cfam = '1F' #cfam associated with the self.sequence_name
        self.snr = 500.
        self.manual = 1

        self.tol = 0.05

    def test_calc_flux_rate(self):
        """Verify correct flux rates are being calculated."""
        # Use mag associated with ut_sequence

        mag = 14.07

        # Calculate flux rate based on npix stored in ut_sequence
        flux_rate_expected = self.snr**2. / 10.  # counts

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        flux_rate, _ = cgieetc.calc_flux_rate(self.sequence_name, self.manual)

        self.assertTrue((np.abs(np.amax(flux_rate) - flux_rate_expected))
                        / flux_rate_expected <= self.tol)

    def test_calc_flux_rate_case(self):
        """Verify correct flux rates with inputs that are not
        case-sensitive."""
        # Use mag associated with ut_sequence

        mag = 14.07

        # Calculate flux rate based on npix stored in ut_sequence
        flux_rate_expected = self.snr**2. / 10.  # counts

        # change the case of some characters
        sequence_name = 'cGI_SEQ_NFOV_ALIgN_LSAm_0_uT'
        phot = 'V'
        spt = 'g2V'

        cgieetc = CGIEETC(mag, phot, spt, self.ut_pointer_path)
        flux_rate, _ = cgieetc.calc_flux_rate(sequence_name, self.manual)

        self.assertTrue((np.abs(np.amax(flux_rate) - flux_rate_expected))
                        / flux_rate_expected <= self.tol)

    def test_invalid_string(self):
        """invalid inputs fail as expected"""
        # sequence_name
        for perr in strlist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_flux_rate(perr, self.manual)

        perr = 'invalid'
        with self.assertRaises(KeyError):
            self.cgieetc.calc_flux_rate(perr, self.manual)


    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""
        # manual
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_flux_rate(self.sequence_name, perr)

class TestCGIEETCExcamSNR(unittest.TestCase):
    """
    Unit tests for excam_SNR method of CGIEETC class.
    """

    def setUp(self):
        self.local_path = LOCAL_PATH

        self.mag = 15.8
        self.phot = 'v'
        self.spt = 'G2V'
        self.ut_pointer_path = str(Path(self.local_path, 'ut_pointer.yaml'))

        self.cgieetc = CGIEETC(self.mag, self.phot, self.spt,
                               self.ut_pointer_path)
        self.sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'
        self.sequence_name2 = 'CGI_SEQ_NFOV_ALIGN_LSAM_1_UT'
        self.g = 10
        self.exptime = 1 #s
        self.nframes = 5
        _, self.flux_rate_peak_pix = self.cgieetc.calc_flux_rate(
                                                        self.sequence_name,
                                                        manual=1)
        self.pc_manual = 1e-6
        # for pc, do a low manual value
        _, self.pc_flux_rate_peak_pix = self.cgieetc.calc_flux_rate(
                                                        self.sequence_name,
                                                        manual=self.pc_manual)

        self.fluxe = self.flux_rate_peak_pix * 1 # scale of 1
        self.pc_fluxe = self.pc_flux_rate_peak_pix * 1 # scale of 1

        # now a sequence that has accurate num_pixels and fraction info
        self.seq = 'CGI_SEQ_DATA_SECONDARY_PR_INFOCUS'
        total_flux, _ = self.cgieetc.calc_flux_rate(self.seq)
        self.fraction_seq = self.cgieetc.sequences[self.seq]['fraction']
        self.num_pixels_seq = self.cgieetc.sequences[self.seq]['num_pixels']
        flux_resel = total_flux * self.fraction_seq
        # average flux per resel pixel
        self.fluxe_seq_an = flux_resel/self.num_pixels_seq

        # pc case for seq:
        total_flux_pc, _ = self.cgieetc.calc_flux_rate(self.seq,
                                                       manual=self.pc_manual)
        flux_resel_pc = total_flux_pc * self.fraction_seq
        self.fluxe_seq_pc = flux_resel_pc/self.num_pixels_seq

        self.cic = self.cgieetc.excam_config['cic']
        self.darke = self.cgieetc.excam_config['darke']
        self.rn = self.cgieetc.excam_config['rn']
        self.X = self.cgieetc.excam_config['X']
        self.Lij = self.cgieetc.excam_config['Lij']
        self.a = self.cgieetc.excam_config['a']
        self.gmax = self.cgieetc.excam_config['gmax']
        self.Nem = self.cgieetc.excam_config['Nem']
        self.Nmin = self.cgieetc.excam_config['Nmin']
        self.Nmax = self.cgieetc.excam_config['Nmax']
        self.tmin = self.cgieetc.excam_config['tmin']
        self.tmax = self.cgieetc.excam_config['tmax']
        self.Nmin = self.cgieetc.excam_config['Nmin']
        self.Nmax = self.cgieetc.excam_config['Nmax']
        self.pc_ecount_max = self.cgieetc.excam_config['pc_ecount_max']
        self.T_factor = self.cgieetc.excam_config['T_factor']

        self.T = self.T_factor*self.rn

        # for some of the pc cases tested, the SNR is not meaningful, and
        # runtime warnings will go off.  Suppress these:
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                 module='eetc.excam_tools')

    def test_success_analog_resel(self):
        '''Successful run for analog mode of 'resel' type.'''
        SNR, etc_max_time, etc_max_time_status = self.cgieetc.excam_SNR(
                            self.seq, self.g, self.exptime,
                            self.nframes, scale=1.,
                            scale_bright=1., manual=1.0, mode='analog')
        expected_SNR = _SNR_CR_resel(self.g, self.exptime, self.nframes,
                                     self.fluxe_seq_an, self.darke, self.cic,
                                     self.rn, self.X, self.a, self.Lij, 1,
                                     self.Nem,
                                     num_pixels=self.num_pixels_seq)
        _, _, exp_etc_max_time, exp_etc_max_time_status = \
            self.cgieetc.excam_saturation_time(self.seq, self.g, f=1,
                                     scale_bright=1., manual=1.0)
        self.assertEqual(SNR, expected_SNR)
        self.assertEqual(etc_max_time, exp_etc_max_time)
        self.assertEqual(etc_max_time_status, exp_etc_max_time_status)
        pass

    def test_success_analog_pixel(self):
        '''Successful run for analog mode of 'pixel' type.'''
        # these two inputs irrelevant when 'pixel' selected:
        num_pixels = 'foo'
        fraction = 'foo'
        SNR, etc_max_time, etc_max_time_status = self.cgieetc.excam_SNR(
                            self.sequence_name, self.g, self.exptime,
                            self.nframes, fraction=fraction,
                            num_pixels=num_pixels, scale=1.,
                            scale_bright=1., manual=1.0, mode='analog',
                            type='pixel')
        expected_SNR = _SNR_CR_resel(self.g, self.exptime, self.nframes,
                                     self.fluxe, self.darke, self.cic,
                                     self.rn, self.X, self.a, self.Lij, 1,
                                     self.Nem, num_pixels=1)
        _, _, exp_etc_max_time, exp_etc_max_time_status = \
            self.cgieetc.excam_saturation_time(self.sequence_name, self.g, f=1,
                                     scale_bright=1., manual=1.0)
        self.assertEqual(SNR, expected_SNR)
        self.assertEqual(etc_max_time, exp_etc_max_time)
        self.assertEqual(etc_max_time_status, exp_etc_max_time_status)
        pass

    def test_success_pc_resel(self):
        '''Successful run for pc mode of 'resel' type.'''
        g = 4000
        SNR, etc_max_time, etc_max_time_status = self.cgieetc.excam_SNR(
                            self.seq, g, self.exptime,
                            self.nframes, scale=1.,
                            scale_bright=1., manual=self.pc_manual, mode='pc')
        expected_SNR = _SNR_CR_pc_resel(g, self.exptime, self.nframes,
                                     self.fluxe_seq_pc, self.darke, self.cic,
                                     self.T, self.X, self.a, self.Lij, 1,
                                     num_pixels=self.num_pixels_seq)
        _, _, exp_etc_max_time, exp_etc_max_time_status = \
            self.cgieetc.excam_saturation_time(self.seq, g, f=1,
                                     scale_bright=1., manual=self.pc_manual)
        self.assertEqual(SNR, expected_SNR)
        self.assertEqual(etc_max_time, exp_etc_max_time)
        self.assertEqual(etc_max_time_status, exp_etc_max_time_status)
        pass

    def test_success_pc_pixel(self):
        '''Successful run for pc mode of 'resel' type.'''
        # these inputs irrelevant when 'pixel' selected:
        num_pixels = 'foo'
        fraction = 'foo'
        SNR, etc_max_time, etc_max_time_status = self.cgieetc.excam_SNR(
                            self.sequence_name, self.g, self.exptime,
                            self.nframes, fraction=fraction,
                            num_pixels=num_pixels, scale=1.,
                            scale_bright=1., manual=self.pc_manual, mode='pc',
                            type='pixel')
        expected_SNR = _SNR_CR_pc_resel(self.g, self.exptime, self.nframes,
                                     self.pc_fluxe, self.darke, self.cic,
                                     self.T, self.X, self.a, self.Lij, 1,
                                     num_pixels=1)
        _, _, exp_etc_max_time, exp_etc_max_time_status = \
            self.cgieetc.excam_saturation_time(self.sequence_name, self.g, f=1,
                                     scale_bright=1., manual=self.pc_manual)
        self.assertEqual(SNR, expected_SNR)
        self.assertEqual(etc_max_time, exp_etc_max_time)
        self.assertEqual(etc_max_time_status, exp_etc_max_time_status)
        pass


    def test_expected_analog_resel(self):
        '''Run an optimization function and then check the SNR with
        excam_SNR().'''
        # maximize SNR
        N, t, g, SNR, _ = self.cgieetc.calc_exp_time_resel(self.seq, snr=None)
        SNR2, _, _ = self.cgieetc.excam_SNR(self.seq, g, t, N)
        self.assertEqual(SNR, SNR2)


    def test_expected_analog_pixel(self):
        '''Run an optimization function and then check the SNR with
        excam_SNR().'''
        # maximize SNR
        N, t, g, SNR, _ = self.cgieetc.calc_exp_time(self.seq, snr=None)
        SNR2, _, _ = self.cgieetc.excam_SNR(self.seq, g, t, N, type='pixel')
        self.assertEqual(SNR, SNR2)


    def test_expected_pc_resel(self):
        '''Run an optimization function and then check the SNR with
        excam_SNR().'''
        # maximize SNR
        N, t, g, SNR, _ = self.cgieetc.calc_pc_exp_time_resel(self.seq,
                                                              snr=None,
                                                        manual=self.pc_manual)
        SNR2, _, _ = self.cgieetc.excam_SNR(self.seq, g, t, N,
                                            manual=self.pc_manual, mode='pc')
        self.assertEqual(SNR, SNR2)


    def test_expected_pc_pixel(self):
        '''Run an optimization function and then check the SNR with
        excam_SNR().'''
        # maximize SNR
        N, t, g, SNR, _ = self.cgieetc.calc_pc_exp_time(self.seq,
                                                              snr=None,
                                                        manual=self.pc_manual)
        SNR2, _, _ = self.cgieetc.excam_SNR(self.seq, g, t, N,
                                            manual=self.pc_manual, mode='pc',
                                            type='pixel')
        self.assertEqual(SNR, SNR2)


    def test_fluxe_0_analog(self):
        '''Gives SNR at tmax if fluxe=0 in analog mode.'''
        mag = 1e30 # to give fluxe of 0 to machine precision

        test_cgieetc = CGIEETC(mag, self.phot, self.spt,
                               self.ut_pointer_path)

        SNR, etc_max_time, etc_max_time_status = test_cgieetc.excam_SNR(
                                    self.sequence_name, self.g, self.exptime,
                                    self.nframes, scale=1, manual=1,
                                    num_pixels=1, mode='analog', type='pixel')
        _, flux_rate_peak_pix = test_cgieetc.calc_flux_rate(
                                                    self.sequence_name,
                                                    manual=1)
        fluxe = flux_rate_peak_pix * 1 #scale of 1
        expected_SNR = _SNR_CR_resel(self.g, self.exptime, self.nframes,
                                     fluxe, self.darke, self.cic,
                                     self.rn, self.X, self.a, self.Lij, 1,
                                     self.Nem, num_pixels=1)
        self.assertEqual(SNR, expected_SNR)
        self.assertEqual(etc_max_time, self.tmax)
        self.assertEqual(etc_max_time_status, 'tmax')


    def test_fluxe_0_pc(self):
        '''Gives SNR at tmax if fluxe=0 in pc mode.'''
        mag = 1e30 # to give fluxe of 0 to machine precision

        test_cgieetc = CGIEETC(mag, self.phot, self.spt,
                               self.ut_pointer_path)

        SNR, etc_max_time, etc_max_time_status = test_cgieetc.excam_SNR(
                                    self.sequence_name, self.g, self.exptime,
                                    self.nframes, scale=1, manual=1,
                                    mode='pc', type='pixel')
        _, flux_rate_peak_pix = test_cgieetc.calc_flux_rate(
                                                    self.sequence_name,
                                                    manual=1)
        fluxe = flux_rate_peak_pix * 1 #scale of 1
        expected_SNR = _SNR_CR_pc_resel(self.g, self.exptime, self.nframes,
                                     fluxe, self.darke, self.cic,
                                     self.T, self.X, self.a, self.Lij, 1,
                                     num_pixels=1)
        self.assertEqual(SNR, expected_SNR)
        self.assertEqual(etc_max_time, self.tmax)
        self.assertEqual(etc_max_time_status, 'tmax')

    def test_pc_g_lb(self):
        '''Tests the case where g_lb >= gmax, which happens when
        read noise and/or T_factor from excam_config.yaml
        is too high or when gmax is too low.'''
        # gmax is 2 below its minimum for pc
        self.cgieetc.excam_config['gmax'] = self.T - 2
        # We want just one of the 3 possible warnings to go off.  So we ensure
        # that tmin is not >= t_ub by making t_pcmax = tmax:
        self.cgieetc.excam_config['pc_ecount_max'] = self.tmax * self.pc_fluxe
        # and we ensure that exptime is not >= etc_max_time by setting it to
        # tmin:
        exptime = self.tmin
        with self.assertWarns(UserWarning):
            _, etc_max_time, _ = self.cgieetc.excam_SNR(
                                    self.sequence_name, self.g, exptime,
                                    self.nframes, scale=1,
                                    manual=self.pc_manual,
                                    mode='pc', type='pixel')
        # additional check of avoiding a warning using output:
        self.assertTrue(exptime < etc_max_time)

    def test_pc_tmin_t_ub(self):
        '''Tests the case where tmin >= t_ub, which happens when
        tmin is >= max exposure time allowed for pc, which is
        pc_ecount_max/fluxe.'''
        # make tmin >= t_ub by setting t_ub to be 0.9*tmin:
        self.cgieetc.excam_config['pc_ecount_max'] = self.tmin*.9*self.pc_fluxe
        # We want just one of the 3 possible warnings to go off.
        # Ensure g_lb < gmax:
        self.cgieetc.excam_config['gmax'] = self.T + 200
        # and we ensure that exptime is not >= etc_max_time by setting it to
        # tmin:
        exptime = self.tmin
        with self.assertWarns(UserWarning):
            _, etc_max_time, _ = self.cgieetc.excam_SNR(
                                    self.sequence_name, self.g, exptime,
                                    self.nframes, scale=1,
                                    manual=self.pc_manual,
                                    mode='pc', type='pixel')
        # additional check of avoiding a warning using output:
        self.assertTrue(exptime < etc_max_time)


    def test_pc_g_less_g_lb(self):
        '''g <= g_lb gives warning.'''
        with self.assertWarns(UserWarning):
            # g_lb is max(1, T_factor*rn). T_factor*rn is about 500 or so.
            self.cgieetc.excam_SNR(self.sequence_name, 10, self.exptime,
                                   self.nframes, manual=self.pc_manual,
                                   mode='pc', type='pixel')

    def test_pc_t_greater_t_ub(self):
        '''t >= t_ub gives warning.'''
        with self.assertWarns(UserWarning):
            # t_ub is min(tmax, pc_ecount_max/fluxe).
            self.cgieetc.excam_SNR(self.seq, self.g, 100,
                                   self.nframes, manual=self.pc_manual,
                                   mode='pc')


    def test_overflow_exception(self):
        '''Overflow error can happen when g too low for PC SNR; g is normally
        high.'''
        with self.assertRaises(OverflowError):
            self.cgieetc.excam_SNR(self.seq, 2, self.exptime,
                                   self.nframes, manual=self.pc_manual,
                                   mode='pc')


    def test_fluxe_bright_less_fluxe(self):
        '''A ValueError is raised if fluxe_bright < fluxe.'''
        # The ratio of fraction/num_pixels must be <= the ratio of the
        # peak pixel's flux to the total flux (which is < 1).  So if
        # fraction/num_pixels is bigger than 1:
        fraction = 0.01
        num_pixels = fraction/1.1
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.seq, self.g, self.exptime,
                                   self.nframes, fraction=fraction,
                                   num_pixels=num_pixels)

    def test_scale_bright_less_scale(self):
        '''A ValueError is raised if scale_bright < scale when type='pixel'.'''

        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.seq, self.g, self.exptime,
                                   self.nframes, scale=1, scale_bright=0.5,
                                   type='pixel')

    def test_pc_exptime(self):
        '''Tests the pc case where exptime > etc_max_time, which means
        detector saturation.'''
        # We want just one of the 3 possible warnings to go off.
        # ensure tmin < t_ub by setting t_ub to be 2*tmin:
        self.cgieetc.excam_config['pc_ecount_max'] = self.tmin*2*self.pc_fluxe
        # Ensure g_lb < gmax:
        self.cgieetc.excam_config['gmax'] = self.T + 200
        # to avoid a ValueError for exptime > tmax, set tmax really high:
        self.cgieetc.excam_config['tmax'] = 1e30
        # Now make exptime too big:
        _, _, maxt, _ = self.cgieetc.excam_saturation_time(self.sequence_name,
                                                        self.g,
                                                        manual=self.pc_manual)
        exptime = maxt + .1
        with self.assertWarns(UserWarning):
            self.cgieetc.excam_SNR(self.seq, self.g, exptime,
                                    self.nframes, scale=1,
                                    manual=self.pc_manual,
                                    mode='pc')

    def test_analog_exptime(self):
        '''Tests the analog case where exptime > etc_max_time, which means
        detector saturation.'''
        g = 1000
        _, _, maxt, _ = self.cgieetc.excam_saturation_time(self.seq,
                                                           g)
        exptime = maxt + .1
        with self.assertWarns(UserWarning):
            self.cgieetc.excam_SNR(self.seq, g, exptime,
                                    self.nframes, scale=1,
                                    mode='analog')

    # input checks for output of _unpack_excam_config()
    def test_darke(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['darke'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.seq, self.g,
                                           self.exptime, self.nframes)

    def test_cic(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['cic'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.seq, self.g,
                                           self.exptime, self.nframes)

    def test_rn(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['rn'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_X(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['X'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_a(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['a'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_Lij(self):
        '''Input check'''
        for perr in ut_check.psilist:
            self.cgieetc.excam_config['Lij'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_alpha0(self):
        '''Input check'''
        for perr in ut_check.rpslist:
            self.cgieetc.excam_config['alpha0'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)
        # alpha0>1:
        self.cgieetc.excam_config['alpha0'] = 2
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                        self.exptime, self.nframes)

    def test_fwc(self):
        '''Input check'''
        for perr in ut_check.psilist:
            self.cgieetc.excam_config['fwc'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_alpha1(self):
        '''Input check'''
        for perr in ut_check.rpslist:
            self.cgieetc.excam_config['alpha1'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)
        # alpha1>1:
        self.cgieetc.excam_config['alpha1'] = 2
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                        self.exptime, self.nframes)

    def test_fwc_em(self):
        '''Input check'''
        for perr in ut_check.psilist:
            self.cgieetc.excam_config['fwc_em'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_Nmin(self):
        '''Input check'''
        for perr in ut_check.psilist:
            self.cgieetc.excam_config['Nmin'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_Nmax(self):
        '''Input check'''
        for perr in ut_check.psilist:
            self.cgieetc.excam_config['Nmax'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)
        # Nmax<Nmin:
        with self.assertRaises(ValueError):
            self.cgieetc.excam_config['Nmax'] = self.Nmax
            self.cgieetc.excam_config['Nmin'] = self.Nmax+1
            self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_tmin(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['tmin'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_tmax(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['tmax'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)
        # tmax<tmin:
        with self.assertRaises(ValueError):
            self.cgieetc.excam_config['tmax'] = self.tmax
            self.cgieetc.excam_config['tmin'] = self.tmax+1
            self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_gmax(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['gmax'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)
        # gmax<1:
        self.cgieetc.excam_config['gmax'] = 0.9
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                        self.exptime, self.nframes)

    def test_n(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['n'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_Nem(self):
        '''Input check'''
        for perr in ut_check.psilist:
            self.cgieetc.excam_config['Nem'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_tol(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['tol'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_delta_constr(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['delta_constr'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_pc_ecount_max(self):
        '''Input check'''
        for perr in ut_check.rpslist:
            self.cgieetc.excam_config['pc_ecount_max'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)

    def test_T_factor(self):
        '''Input check'''
        for perr in ut_check.rpslist:
            self.cgieetc.excam_config['T_factor'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                           self.exptime, self.nframes)


    # input checks for excam_SNR()
    def test_sequence_name(self):
        '''Input check'''
        for perr in ut_check.strlist:
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(perr, self.g, self.exptime,
                                        self.nframes)

    def test_g(self):
        '''Input check'''
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, perr, self.exptime,
                                        self.nframes)
        # g<1:
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.sequence_name, 0.6, self.exptime,
                                        self.nframes)
        # g>gmax:
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.sequence_name, self.gmax+1,
                                       self.exptime, self.nframes)

    def test_exptime(self):
        '''Input check'''
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g, perr,
                                        self.nframes)
        # exptime<tmin:
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.sequence_name, self.g, self.tmin*0.9,
                                        self.nframes)
        # exptime>tmax:
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.sequence_name, self.g, self.tmax*1.1,
                                        self.nframes)

    def test_nframes(self):
        """Input check"""
        for perr in ut_check.psilist:
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                       self.exptime, perr)
        # nframes<Nmin:
        with self.assertRaises(ValueError):
            self.cgieetc.excam_config['Nmin'] = 2
            self.cgieetc.excam_SNR(self.sequence_name, self.g, self.exptime,
                                        1)
        # nframes>Nmax:
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.sequence_name, self.g, self.exptime,
                                        self.Nmax+1)

    def test_fraction(self):
        '''Input check'''
        for perr in ut_check.rpslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                       self.exptime, self.nframes,
                                       fraction=perr)

    def test_fraction_greater_1(self):
        '''Fraction can't be bigger than 1.'''
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                       self.exptime, self.nframes,
                                       fraction=1.1)

    def test_fraction_null(self):
        '''Input check. sequence_name has fraction = null, so this raises
        error.'''
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                    self.exptime, self.nframes,
                                    fraction=None)

    def test_num_pixels(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.seq, self.g,
                                       self.exptime, self.nframes,
                                       num_pixels=perr)

    def test_num_pixels_null(self):
        '''Input check.  sequence_name2 has num_pixels = null, so this raises
        error.'''
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.sequence_name2, self.g,
                                    self.exptime, self.nframes,
                                    num_pixels=None)

    def test_scale(self):
        '''Input check'''
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                       self.exptime, self.nframes,
                                       scale=perr)

    def test_scale_bright(self):
        '''Input check'''
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                       self.exptime, self.nframes,
                                       scale_bright=perr)

    def test_manual(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                       self.exptime, self.nframes,
                                       manual=perr)

    def test_mode(self):
        '''Input check'''
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                       self.exptime, self.nframes,
                                       mode='foo')

    def test_type(self):
        '''Input check'''
        with self.assertRaises(ValueError):
            self.cgieetc.excam_SNR(self.sequence_name, self.g,
                                       self.exptime, self.nframes,
                                       type='foo')

class TestCGIEETCExcamSaturationTime(unittest.TestCase):
    """
    Unit tests for excam_saturation_time method of CGIEETC class.
    """

    def setUp(self):
        self.local_path = LOCAL_PATH

        self.mag = 15.8
        self.phot = 'v'
        self.spt = 'G2V'
        self.ut_pointer_path = str(Path(self.local_path, 'ut_pointer.yaml'))

        self.cgieetc = CGIEETC(self.mag, self.phot, self.spt,
                               self.ut_pointer_path)
        self.sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'

        self.fwc_em = self.cgieetc.excam_config['fwc_em']
        self.fwc = self.cgieetc.excam_config['fwc']
        self.cic = self.cgieetc.excam_config['cic']
        self.darke = self.cgieetc.excam_config['darke']
        self.alpha1 = self.cgieetc.excam_config['alpha1']
        self.alpha0 = self.cgieetc.excam_config['alpha0']
        self.gmax = self.cgieetc.excam_config['gmax']
        self.Nem = self.cgieetc.excam_config['Nem']
        self.n = self.cgieetc.excam_config['n']

    def test_EMwell_expected(self):
        """Verify function gives time as expected (fwcem_t and emrail_t)."""
        g = 10
        f = 0.9 # just to exercise the use of something other than the default
        max_time, max_time_status, etc_max_time, etc_max_time_status = \
            self.cgieetc.excam_saturation_time(self.sequence_name, g=g, f=f)
        # fwc_em is not more than 10 times the per-pixel well, and gain is 10
        # So the fwc_em would be saturated first
        _, fluxe_bright = self.cgieetc.calc_flux_rate(self.sequence_name)

        emrail_t = (2*self.alpha1*self.fwc_em - 2*self.cic*g +
           _ENF(g, self.Nem)**2*g*self.n**2 -
           g*np.sqrt(4*self.alpha1*_ENF(g, self.Nem)**2*self.fwc_em*self.n**2/g
           + _ENF(g, self.Nem)**4*self.n**4))/(2*g*(fluxe_bright + self.darke))

        fwcem_t = (f*self.fwc_em - self.cic*g)/(g*(fluxe_bright + self.darke))

        self.assertEqual(etc_max_time, emrail_t)
        self.assertEqual(etc_max_time_status, 'EM gain well')
        self.assertEqual(max_time, fwcem_t)
        self.assertEqual(max_time_status, 'EM gain well')

    def test_ppwell_expected(self):
        """Verify function gives time as expected (fwc_t and rail_t)."""
        # needs to be something less than the ratio fwc_em/fwc in order for fwc
        # to saturate first
        g = 1
        f = 0.9 # just to exercise the use of something other than the default
        max_time, max_time_status, etc_max_time, etc_max_time_status = \
            self.cgieetc.excam_saturation_time(self.sequence_name, g=g, f=f)
        # fwc_em is not more than 10 times the per-pixel well, and gain is 10
        # So the fwc_em would be saturated first
        _, fluxe_bright = self.cgieetc.calc_flux_rate(self.sequence_name)

        rail_t = (2*self.alpha0*self.fwc + self.n*(self.n -
            np.sqrt(4*self.alpha0*self.fwc +
            self.n**2)))/(2*(fluxe_bright+self.darke))

        fwc_t = f*self.fwc/(fluxe_bright + self.darke)

        self.assertEqual(etc_max_time, rail_t)
        self.assertEqual(etc_max_time_status, 'per-pixel well')
        self.assertEqual(max_time, fwc_t)
        self.assertEqual(max_time_status, 'per-pixel well')

    def test_EMwell_tmax_expected(self):
        """Verify function gives time as expected (tmax and fwcem_t)."""
        # select very small peak flux so that time is tmax
        scale_bright = 0.0001
        f = 0.9 # just to exercise the use of something other than the default
        g = 10
        max_time, max_time_status, etc_max_time, etc_max_time_status = \
            self.cgieetc.excam_saturation_time(self.sequence_name, g=g,
                f=f, scale_bright=scale_bright, manual=0.9)

        _, flux_rate_peak_pix = self.cgieetc.calc_flux_rate(self.sequence_name,
                            manual=0.9)
        fluxe_bright = flux_rate_peak_pix * scale_bright

        # expected max time until fraction of fwc_em
        fwcem_t = (f*self.fwc_em - self.cic*g)/(g*(fluxe_bright +
                self.darke))
        self.assertEqual(etc_max_time, self.cgieetc.excam_config['tmax'])
        self.assertEqual(etc_max_time_status, 'tmax')
        self.assertEqual(max_time, fwcem_t)
        # since EM gain is not more than 10 times the per-pixel well, and gain
        # is 10
        self.assertEqual(max_time_status, 'EM gain well')

    def test_ppwell_tmax_expected(self):
        """Verify function gives time as expected (tmax and fwc_t)."""
        # needs to be something less than the ratio fwc_em/fwc in order for fwc
        # to saturate first
        g = 1.01
        f = 0.9 # just to exercise the use of something other than the default
        scale_bright = 0.0001
        max_time, max_time_status, etc_max_time, etc_max_time_status = \
            self.cgieetc.excam_saturation_time(self.sequence_name, g=g, f=f,
            scale_bright=scale_bright)
        # fwc_em is not more than 10 times the per-pixel well, and gain is 10
        # So the fwc_em would be saturated first
        _, flux_rate_peak = self.cgieetc.calc_flux_rate(self.sequence_name)
        fluxe_bright = flux_rate_peak * scale_bright

        fwc_t = f*self.fwc/(fluxe_bright + self.darke)

        self.assertEqual(etc_max_time, self.cgieetc.excam_config['tmax'])
        self.assertEqual(etc_max_time_status, 'tmax')
        self.assertEqual(max_time, fwc_t)
        self.assertEqual(max_time_status, 'per-pixel well')

    # input checks for output of _unpack_excam_config()
    def test_darke(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['darke'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_cic(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['cic'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_rn(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['rn'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_X(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['X'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_a(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['a'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_Lij(self):
        '''Input check'''
        for perr in ut_check.psilist:
            self.cgieetc.excam_config['Lij'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_alpha0(self):
        '''Input check'''
        for perr in ut_check.rpslist:
            self.cgieetc.excam_config['alpha0'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)
        # alpha0>1:
        self.cgieetc.excam_config['alpha0'] = 2
        with self.assertRaises(ValueError):
            self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_fwc(self):
        '''Input check'''
        for perr in ut_check.psilist:
            self.cgieetc.excam_config['fwc'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_alpha1(self):
        '''Input check'''
        for perr in ut_check.rpslist:
            self.cgieetc.excam_config['alpha1'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)
        # alpha1>1:
        self.cgieetc.excam_config['alpha1'] = 2
        with self.assertRaises(ValueError):
            self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_fwc_em(self):
        '''Input check'''
        for perr in ut_check.psilist:
            self.cgieetc.excam_config['fwc_em'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_Nmin(self):
        '''Input check'''
        for perr in ut_check.psilist:
            self.cgieetc.excam_config['Nmin'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_Nmax(self):
        '''Input check'''
        for perr in ut_check.psilist:
            self.cgieetc.excam_config['Nmax'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)
        # Nmax<Nmin:
        with self.assertRaises(ValueError):
            self.cgieetc.excam_config['Nmin'] = 2
            self.cgieetc.excam_config['Nmax'] = 1
            self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_tmin(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['tmin'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_tmax(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['tmax'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)
        # tmax<tmin:
        with self.assertRaises(ValueError):
            self.cgieetc.excam_config['tmax'] = \
                self.cgieetc.excam_config['tmin']*0.9
            self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_gmax(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['gmax'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)
        # gmax<1:
        self.cgieetc.excam_config['gmax'] = 0.9
        with self.assertRaises(ValueError):
            self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_n(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['n'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_Nem(self):
        '''Input check'''
        for perr in ut_check.psilist:
            self.cgieetc.excam_config['Nem'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_tol(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['tol'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    def test_delta_constr(self):
        '''Input check'''
        for perr in ut_check.rnslist:
            self.cgieetc.excam_config['delta_constr'] = perr
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, 10)

    # input checks for excam_saturation_time()
    def test_sequence_name(self):
        """Bad input for sequence_name."""
        for perr in ut_check.strlist:
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(sequence_name=perr, g=10)

    def test_g(self):
        '''Bad input for gain.'''
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, g=perr)

    def test_g_value(self):
        '''Bad values for g even though it's a real positive scalar.'''
        for perr in [0.5, self.gmax+1]:
            with self.assertRaises(ValueError):
                self.cgieetc.excam_saturation_time(self.sequence_name, g=perr)

    def test_f(self):
        '''Bad input for fraction.'''
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, g=10,
                f=perr)

    def test_f_value(self):
        '''Bad values for fraction even though it's a real positive scalar.'''
        with self.assertRaises(ValueError):
            self.cgieetc.excam_saturation_time(self.sequence_name, g=10,
                f=2.1)

    def test_scale_bright(self):
        '''Bad input for scale_bright.'''
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, g=10,
                scale_bright=perr)

    def test_manual(self):
        '''Bad input for manual.'''
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.excam_saturation_time(self.sequence_name, g=10,
                manual=perr)


class TestCGIEETCCalcConstIntTime(unittest.TestCase):
    """
    Unit tests for calc_const_int_time method of CGIEETC class.
    """

    def setUp(self):
        self.local_path = LOCAL_PATH

        self.mag = 15.8
        self.phot = 'v'
        self.spt = 'G2V'
        self.ut_pointer_path = str(Path(self.local_path, 'ut_pointer.yaml'))

        self.cgieetc = CGIEETC(self.mag, self.phot, self.spt,
                               self.ut_pointer_path)

        self.sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'
        self.cfam = '1F'  # cfam associated with this sequence
        self.t_tot = 200.
        self.scale = 1.
        self.scale_bright = 1.
        self.manual = 1.0

        self.tol = 0.05

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """Verify function completes without errors given valid inputs."""
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.calc_const_int_time(self.sequence_name,
                              self.t_tot,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        cgieetc.calc_const_int_time(self.sequence_name,
                              self.t_tot,
                              self.scale,
                              self.scale_bright,
                              self.manual,
                              hard_limit=False)
        pass


    def test_failure_one_or_more_fixed(self):
        """Verify unsupported constraint configurations caught"""
        # fixed g
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_const_int_time(self.sequence_name,
                                  self.t_tot,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)
        # fixed N
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_const_int_time(self.sequence_name,
                                  self.t_tot,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)
        # fixed t
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_const_int_time(self.sequence_name,
                                  self.t_tot,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)


    def test_reasonable_outputs(self):
        """Verify that outputs are reasonable."""
        mag = 14.07

        int_time = 17.

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        n_frames, exp_time, gain, snr, t_tot_out = cgieetc.calc_const_int_time(
                                                            self.sequence_name,
                                                            int_time,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # assuming saturation is avoided, the highest number of frames is
        # usually what maximizes SNR, which minimizes exposure time:
        self.assertTrue(np.isclose(exp_time, cgieetc.excam_config['tmin'],
                                    atol=2))
        # as expected for hard_limit=True:
        self.assertTrue(t_tot_out == int_time)

        # now compare to output of calc_exp_time() using max snr achieved for
        # a total integration time self.t_tot as the target SNR:
        n, t, g, snr2, _ = cgieetc.calc_exp_time(self.sequence_name, snr,
                                                self.scale, self.scale_bright,
                                                self.manual)
        self.assertTrue(n == n_frames)
        self.assertTrue(np.isclose(t, exp_time, rtol=0.05))
        self.assertTrue(np.isclose(g, gain, rtol=0.05))
        self.assertTrue(np.isclose(snr, snr2, rtol=0.05))


    def test_nd_filter_exp_time(self):
        """Verify ND filter is being applied correctly."""
        mag = 14.07
        manual = 0.1  # OD1

        int_time = 17  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        n_frames, exp_time, _, _, _ = cgieetc.calc_const_int_time(
                                                            self.sequence_name,
                                                            int_time,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # using manual:
        n_frames2, exp_time2, _, _, _ = cgieetc.calc_const_int_time(
                                                            self.sequence_name,
                                                            int_time,
                                                            self.scale,
                                                            self.scale_bright,
                                                            manual)
        # with 0.1 applied, the exposure time should be increased by roughly
        # a factor of 10, which would make N decreased by a factor of 10:
        self.assertTrue(np.isclose(n_frames2, n_frames/10))
        self.assertTrue(np.isclose(exp_time2, exp_time*10, rtol=0.05))


    def test_scale_factor(self):
        """Verify scale factor is being applied correctly."""
        mag = 14.07
        scale = 0.5
        scale_bright = 0.5

        int_time = 17  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        n_frames, exp_time, _, _, _ = cgieetc.calc_const_int_time(
                                                            self.sequence_name,
                                                            int_time,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # using scale and scale_bright:
        n_frames2, exp_time2, _, _, _ = cgieetc.calc_const_int_time(
                                                            self.sequence_name,
                                                            int_time,
                                                            scale,
                                                            scale_bright,
                                                            self.manual)
        # with 0.5 applied, the exposure time should be increased by roughly
        # a factor of 2, which would make N decreased by a factor of 2:
        self.assertTrue(np.isclose(n_frames2, n_frames/2))
        self.assertTrue(np.isclose(exp_time2, exp_time*2, rtol=0.05))

    def test_flux_ratio_psf(self):
        """Verify the flux ratio parameter is being used correctly."""
        mag = 14.07

        # Unit test sequence that has a peak flux ratio of 0.5, compared to 1
        # in self.sequence_name
        sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_1_UT'

        int_time = 17.  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        n_frames, exp_time, _, _, _ = cgieetc.calc_const_int_time(
                                                            self.sequence_name,
                                                            int_time,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # using sequence_name:
        n_frames2, exp_time2, _, _, _ = cgieetc.calc_const_int_time(
                                                            sequence_name,
                                                            int_time,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # with 0.5 applied, the exposure time should be increased by roughly
        # a factor of 2, which would make N decreased by a factor of 2:
        self.assertTrue(np.isclose(n_frames2, n_frames/2))
        self.assertTrue(np.isclose(exp_time2, exp_time*2, rtol=0.05))

    def test_invalid_string(self):
        """invalid inputs fail as expected"""
        # sequence_name
        for perr in strlist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time(perr, self.t_tot,
                                           self.scale, self.scale_bright,
                                           self.manual)

    def test_invalid_real_positive_scalar(self):
        """invalid inputs fail as expected"""
        # scale
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time(self.sequence_name,
                                           self.t_tot, perr, self.scale_bright,
                                           self.manual)

        # scale_bright
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time(self.sequence_name,
                                           self.t_tot, self.scale, perr,
                                           self.manual)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""
        # t_tot
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time(self.sequence_name, perr,
                                           self.scale, self.scale_bright,
                                           self.manual)

        # manual
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time(self.sequence_name,
                                           self.t_tot, self.scale,
                                           self.scale_bright, perr)

    def test_scale_bright_less_scale(self):
        """scale_bright must be >= scale."""
        scale_bright = self.scale/2
        with self.assertRaises(ValueError):
            self.cgieetc.calc_const_int_time(self.sequence_name,
                                           self.t_tot, self.scale,
                                           scale_bright, self.manual)

    def test_invalid_hard_limit(self):
        '''Invalid hard_limit caught.'''
        check_list = [2, 'foo', -3.4]
        for perr in check_list:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time(self.sequence_name,
                                           self.t_tot, self.scale,
                                           self.scale_bright, self.manual,
                                           hard_limit=perr)


class TestCGIEETCCalcConstIntTimeResel(unittest.TestCase):
    """
    Unit tests for calc_const_int_time_resel method of CGIEETC class.
    """

    def setUp(self):
        self.local_path = LOCAL_PATH

        self.mag = 15.8
        self.phot = 'v'
        self.spt = 'G2V'
        self.ut_pointer_path = str(Path(self.local_path, 'ut_pointer.yaml'))

        self.cgieetc = CGIEETC(self.mag, self.phot, self.spt,
                               self.ut_pointer_path)

        self.sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'
        self.sequence_name2 = 'CGI_SEQ_NFOV_ALIGN_LSAM_1_UT'
        self.cfam = '1F'  # cfam associated with this sequence
        self.t_tot = 200.
        self.scale = 1.
        self.scale_bright = 1.
        self.manual = 1.0
        self.fraction = 0.01
        self.num_pixels = 10

        self.tol = 0.05

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """Verify function completes without errors given valid inputs."""
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.calc_const_int_time_resel(self.sequence_name,
                              self.t_tot,
                              self.fraction,
                              self.num_pixels,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        # drawing from sequence data for num_pixels and fraction
        cgieetc.calc_const_int_time_resel(
                              'CGI_SEQ_DATA_SECONDARY_PR_INFOCUS',
                              self.t_tot,
                              None,
                              None,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        cgieetc.calc_const_int_time_resel(self.sequence_name,
                              self.t_tot,
                              self.fraction,
                              self.num_pixels,
                              self.scale,
                              self.scale_bright,
                              self.manual,
                              hard_limit=False)
        pass


    def test_failure_one_or_more_fixed(self):
        """Verify unsupported constraint configurations caught"""
        # fixed g
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_const_int_time_resel(self.sequence_name,
                                  self.t_tot,
                                  self.fraction,
                                  self.num_pixels,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)
        # fixed N
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_const_int_time_resel(self.sequence_name,
                                  self.t_tot,
                                  self.fraction,
                                  self.num_pixels,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)
        # fixed t
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_const_int_time_resel(self.sequence_name,
                                  self.t_tot,
                                  self.fraction,
                                  self.num_pixels,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)

    def test_biggest_scale(self):
        """Test that biggest possible value for fraction gives successful
        run.  And the result should agree with the result from
        calc_gain_fixed_Ntime(). """
        #ratio of peak-pixel flux to total flux:
        this_seq = self.cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        #fraction/(x*num_pixels) must be 1 or less, so test limiting case
        fraction = x*num_pixels

        n1, t1, g1, snr_resel, t_tot_out1 = \
                self.cgieetc.calc_const_int_time_resel(
                    self.sequence_name, self.t_tot, fraction, num_pixels)
        #calc_const_int_time() is for one pixel, so it should agree
        n2, t2, g2, snr_pp, t_tot_out2 = self.cgieetc.calc_const_int_time(
            self.sequence_name, self.t_tot)
        self.assertEqual(n1, n2)
        self.assertEqual(t1, t2)
        self.assertEqual(g1, g2)
        self.assertEqual(snr_resel, snr_pp)
        self.assertEqual(t_tot_out1, t_tot_out2)

    def test_equivalent_inputs(self):
        """If the fraction is decreased by some factor, the
        number of pixels would also decrease by the same factor, and the SNR
        per pixel should not change much, as long as the cosmic ray correction
        to the SNR is small, and it should be for a small number of pixels and
        a small exposure time.  The other outputs shouldn't change, as
        long as constraints aren't hit with either set of inputs."""
        num_pixels = 2
        n, t, g, snr, t_tot_out = \
            self.cgieetc.calc_const_int_time_resel(self.sequence_name,
            self.t_tot, self.fraction, num_pixels)

        snr_pp = snr/np.sqrt(num_pixels)

        f2 = self.fraction/2
        num2 = num_pixels/2

        #comparable target SNR per resel would be (SNR per pixel)*np.sqrt(num2)
        #expected_snr2 = snr_pp*np.sqrt(num2)

        n2, t2, g2, snr2, t_tot_out2 = \
            self.cgieetc.calc_const_int_time_resel(self.sequence_name,
            self.t_tot, f2, num2)

        snr_pp2 = snr2/np.sqrt(num2)

        self.assertTrue(n == n2)
        self.assertTrue(np.isclose(t, t2, rtol=1e-2))
        self.assertTrue(np.isclose(g, g2, rtol=1e-2))
        self.assertTrue(np.isclose(snr_pp, snr_pp2, rtol=1e-2))
        self.assertTrue(t_tot_out == t_tot_out2)

    def test_reasonable_outputs(self):
        """Verify that outputs are reasonable."""
        mag = 14.07
        int_time = 17.

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        this_seq = cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces the case of calc_const_int_time()
        n_frames, exp_time, gain, snr, t_tot_out = \
                                    cgieetc.calc_const_int_time_resel(
                                                            self.sequence_name,
                                                            int_time,
                                                            fraction,
                                                            num_pixels,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # assuming saturation is avoided, the highest number of frames is
        # usually what maximizes SNR, which minimizes exposure time:
        self.assertTrue(np.isclose(exp_time, cgieetc.excam_config['tmin'],
                                    atol=2))
        # as expected for hard_limit=True:
        self.assertTrue(t_tot_out == int_time)

        # now compare to output of calc_exp_time() using max snr achieved for
        # a total integration time self.t_tot as the target SNR:
        n, t, g, snr2, _ = cgieetc.calc_exp_time(self.sequence_name, snr,
                                                self.scale, self.scale_bright,
                                                self.manual)
        self.assertTrue(n == n_frames)
        self.assertTrue(np.isclose(t, exp_time, rtol=0.05))
        self.assertTrue(np.isclose(g, gain, rtol=0.05))
        self.assertTrue(np.isclose(snr, snr2, rtol=0.05))


    def test_nd_filter_exp_time(self):
        """Verify ND filter is being applied correctly."""
        mag = 14.07
        manual = 0.1  # OD1

        int_time = 17  # seconds
        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        this_seq = cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces the case of calc_const_int_time()
        n_frames, exp_time, _, _, _ = cgieetc.calc_const_int_time_resel(
                                                            self.sequence_name,
                                                            int_time,
                                                            fraction,
                                                            num_pixels,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # using manual:
        n_frames2, exp_time2, _, _, _ = cgieetc.calc_const_int_time_resel(
                                                            self.sequence_name,
                                                            int_time,
                                                            fraction,
                                                            num_pixels,
                                                            self.scale,
                                                            self.scale_bright,
                                                            manual)
        # with 0.1 applied, the exposure time should be increased by roughly
        # a factor of 10, which would make N decreased by a factor of 10:
        self.assertTrue(np.isclose(n_frames2, n_frames/10))
        self.assertTrue(np.isclose(exp_time2, exp_time*10, rtol=0.05))


    def test_scale_factor(self):
        """Verify scale factor is being applied correctly."""
        mag = 14.07
        scale = 0.5
        scale_bright = 0.5

        int_time = 17  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        this_seq = cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces the case of calc_const_int_time()
        n_frames, exp_time, _, _, _ = cgieetc.calc_const_int_time_resel(
                                                            self.sequence_name,
                                                            int_time,
                                                            fraction,
                                                            num_pixels,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # using scale and scale_bright:
        n_frames2, exp_time2, _, _, _ = cgieetc.calc_const_int_time_resel(
                                                            self.sequence_name,
                                                            int_time,
                                                            fraction,
                                                            num_pixels,
                                                            scale,
                                                            scale_bright,
                                                            self.manual)
        # with 0.5 applied, the exposure time should be increased by roughly
        # a factor of 2, which would make N decreased by a factor of 2:
        self.assertTrue(np.isclose(n_frames2, n_frames/2))
        self.assertTrue(np.isclose(exp_time2, exp_time*2, rtol=0.05))

    def test_flux_ratio_psf(self):
        """Verify the flux ratio parameter is being used correctly."""
        mag = 14.07

        # Unit test sequence that has a peak flux ratio of 0.5, compard to 1
        # in self.sequence_name
        sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_1_UT'

        int_time = 17.  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        this_seq = cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces the case of calc_const_int_time()
        n_frames, exp_time, _, _, _ = cgieetc.calc_const_int_time_resel(
                                                            self.sequence_name,
                                                            int_time,
                                                            fraction,
                                                            num_pixels,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # using sequence_name:
        this_seq2 = cgieetc.sequences[sequence_name]
        x2 = this_seq2['peak_flux_ratio_pix']
        num_pixels2 = 1
        # reproduces the case of calc_const_int_time()
        fraction2 = x2*num_pixels2
        n_frames2, exp_time2, _, _, _ = cgieetc.calc_const_int_time_resel(
                                                            sequence_name,
                                                            int_time,
                                                            fraction2,
                                                            num_pixels2,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # with 0.5 applied, the exposure time should be increased by roughly
        # a factor of 2, which would make N decreased by a factor of 2:
        self.assertTrue(np.isclose(n_frames2, n_frames/2))
        self.assertTrue(np.isclose(exp_time2, exp_time*2, rtol=0.05))

    def test_invalid_string(self):
        """invalid inputs fail as expected"""
        # sequence_name
        for perr in strlist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time_resel(perr, self.t_tot,
                                           self.fraction, self.num_pixels,
                                           self.scale, self.scale_bright,
                                           self.manual)

    def test_invalid_real_positive_scalar(self):
        """invalid inputs fail as expected"""
        # fraction
        for perr in rpslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, perr,
                                           self.num_pixels,
                                           self.scale, self.scale_bright,
                                           self.manual)

        # scale
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           self.num_pixels,
                                           perr, self.scale_bright,
                                           self.manual)

        # scale_bright
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           self.num_pixels, self.scale, perr,
                                           self.manual)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""
        # t_tot
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time_resel(self.sequence_name,
                                           perr, self.fraction,
                                           self.num_pixels,
                                           self.scale, self.scale_bright,
                                           self.manual)

        # num_pixels
        for perr in rnslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           perr,
                                           self.scale, self.scale_bright,
                                           self.manual)

        # manual
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           self.num_pixels, self.scale,
                                           self.scale_bright, perr)

    def test_fraction_less_than_1(self):
        '''Fraction must be 1 or less.'''
        fraction = 1.1
        with self.assertRaises(ValueError):
            self.cgieetc.calc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, fraction,
                                           self.num_pixels, self.scale,
                                           self.scale_bright, self.manual)

    def test_fraction_null(self):
        '''Fraction coming from sequence_name is null.'''
        with self.assertRaises(ValueError):
            self.cgieetc.calc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, None,
                                           self.num_pixels, self.scale,
                                           self.scale_bright, self.manual)

    def test_num_pixels_null(self):
        '''num_pixels coming from sequence_name2 is null.'''
        with self.assertRaises(ValueError):
            self.cgieetc.calc_const_int_time_resel(self.sequence_name2,
                                           self.t_tot, self.fraction,
                                           None, self.scale,
                                           self.scale_bright, self.manual)

    def test_scale_bright_less_scale(self):
        """scale_bright must be >= scale."""
        scale_bright = self.scale/2
        with self.assertRaises(ValueError):
            self.cgieetc.calc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           self.num_pixels, self.scale,
                                           scale_bright, self.manual)

    def test_invalid_hard_limit(self):
        '''Invalid hard_limit caught.'''
        check_list = [2, 'foo', -3.4]
        for perr in check_list:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           self.num_pixels, self.scale,
                                           self.scale_bright, self.manual,
                                           hard_limit=perr)

    def test_scale_bright_less_scale_2(self):
        '''A ValueError is raised if scale_bright < scale.'''
        # The ratio of fraction/num_pixels must be <= the ratio of the
        # peak pixel's flux to the total flux (which is < 1).  So if
        # fraction/num_pixels is bigger than 1:
        num_pixels = self.fraction/1.1
        with self.assertRaises(ValueError):
            self.cgieetc.calc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           num_pixels, self.scale,
                                           self.scale_bright, self.manual)


class TestCGIEETCCalcExpTime(unittest.TestCase):
    """
    Unit tests for calc_exp_time method of CGIEETC class.
    """

    def setUp(self):
        self.local_path = LOCAL_PATH

        self.mag = 15.8
        self.phot = 'v'
        self.spt = 'G2V'
        self.ut_pointer_path = str(Path(self.local_path, 'ut_pointer.yaml'))

        self.cgieetc = CGIEETC(self.mag, self.phot, self.spt,
                               self.ut_pointer_path)

        self.sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'
        self.cfam = '1F'  # cfam associated with this sequence
        self.snr = 500.
        self.scale = 1.
        self.scale_bright = 1.
        self.manual = 1.0

        self.tol = 0.05

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """Verify function completes without errors given valid inputs."""
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.calc_exp_time(self.sequence_name,
                              self.snr,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        pass


    def test_success_fixedt(self):
        """Verify function completes without errors given valid inputs."""
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        cgieetc.calc_exp_time(self.sequence_name,
                              self.snr,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        pass


    def test_success_fixedn(self):
        """Verify function completes without errors given valid inputs."""

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        cgieetc.calc_exp_time(self.sequence_name,
                              self.snr,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        pass


    def test_success_fixedg(self):
        """Verify function completes without errors given valid inputs."""

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.calc_exp_time(self.sequence_name,
                              self.snr,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        pass


    def test_failure_two_or_more_fixed(self):
        """Verify unsupported constraint configurations caught"""

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_exp_time(self.sequence_name,
                                  self.snr,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)


        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_exp_time(self.sequence_name,
                                  self.snr,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_exp_time(self.sequence_name,
                                  self.snr,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_exp_time(self.sequence_name,
                                  self.snr,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)

        pass


    def test_success_None(self):
        """
        Verify function completes without errors given valid inputs.

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.calc_exp_time(self.sequence_name,
                              None,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        pass


    def test_success_fixedt_None(self):
        """
        Verify function completes without errors given valid inputs.

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        cgieetc.calc_exp_time(self.sequence_name,
                              None,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        pass


    def test_success_fixedn_None(self):
        """
        Verify function completes without errors given valid inputs.

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        cgieetc.calc_exp_time(self.sequence_name,
                              None,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        pass


    def test_success_fixedg_None(self):
        """
        Verify function completes without errors given valid inputs.

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.calc_exp_time(self.sequence_name,
                              None,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        pass



    def test_calc_exp_time(self):
        """Verify that correct exposure times are being calculated."""
        mag = 14.07

        exp_time_expected = 17.

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        n_frames, exp_time, _, _, _ = cgieetc.calc_exp_time(self.sequence_name,
                                                            self.snr,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)

        self.assertTrue((np.abs((exp_time * n_frames) - exp_time_expected))
                        / exp_time_expected <= self.tol)

    def test_opt_flag(self):
        """
        Verify the optimization flag is set correctly when you ask for an SNR
        that is too high to be achieved within camera parameter boundaries.
        """
        mag = 14.07
        manual = 0.1  # OD1
        snr_high = 5e12

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        _, _, _, _, optflag = cgieetc.calc_exp_time(self.sequence_name,
                                                    self.snr,
                                                    self.scale,
                                                    self.scale_bright, manual)
        _, _, _, _, optflag_high = cgieetc.calc_exp_time(self.sequence_name,
                                                         snr_high, self.scale,
                                                         self.scale_bright,
                                                         manual)

        self.assertTrue(optflag == 0)
        self.assertTrue(optflag_high == 1)

    def test_high_snr_same_as_None(self):
        """
        Verify SNR-too-high gives the same result as no SNR constraint at all
        """
        snr_high = 5e30

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        n_frames1, exp_time1, gain1, _, _ = cgieetc.calc_exp_time(
                                                    self.sequence_name,
                                                    None,
                                                    self.scale,
                                                    self.scale_bright,
                                                    self.manual)
        n_frames2, exp_time2, gain2, _, _ = cgieetc.calc_exp_time(
                                                         self.sequence_name,
                                                         snr_high,
                                                         self.scale,
                                                         self.scale_bright,
                                                         self.manual)

        self.assertTrue(n_frames1 == n_frames2)
        self.assertTrue(np.max(np.abs(exp_time1 - exp_time2)) < self.tol)
        self.assertTrue(np.max(np.abs(gain1 - gain2)) < self.tol)
        pass



    def test_nd_filter_exp_time(self):
        """Verify ND filter is being applied correctly."""
        mag = 14.07
        manual = 0.1  # OD1

        exp_time_expected = 17./manual  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        n_frames, exp_time, _, _, _ = cgieetc.calc_exp_time(self.sequence_name,
                                                            self.snr,
                                                            self.scale,
                                                            self.scale_bright,
                                                            manual)

        # actual non-integer # of frames from excam_tools is 10.6, and
        # n_frames is 11.  So just make sure it's bounded b/w 10 and 11:
        self.assertTrue(
            (exp_time * (n_frames - 1) <= (1+self.tol) * exp_time_expected) and
            (exp_time * n_frames >= (1-self.tol) * exp_time_expected)
                        )

    def test_scale_factor(self):
        """Verify scale factor is being applied correctly."""
        mag = 3.49
        snr = 50.
        scale = 1e-6  # 10^-6 contrast
        scale_bright = 1e-6  # let's say it's also the brightest speckle


        #old: the answer (V of 3.5, 5700 K star, SNR of 50 yields t_exp=11.8 s)
        #old: the answer (V of 3.5, 5770 K star, SNR of 50 yields t_exp=12.4 s)
        exp_time_expected = 11.6    # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        n_frames, exp_time, _, _, _ = cgieetc.calc_exp_time(self.sequence_name,
                                                            snr,
                                                            scale,
                                                            scale_bright,
                                                            self.manual)

        self.assertTrue((np.abs((exp_time * n_frames) - exp_time_expected))
                        / exp_time_expected <= self.tol)

    def test_flux_ratio_psf(self):
        """Verify the flux ratio parameter is being used correctly."""
        mag = 14.07

        # Unit test sequence that has a peak flux ratio of 0.5, compared to 1
        # in self.sequence_name
        sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_1_UT'

        exp_time_expected = 17 * 2.  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        n_frames, exp_time, _, _, _ = cgieetc.calc_exp_time(sequence_name,
                                                            self.snr,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # actual non-integer # of frames from excam_tools is around 10.6, and
        # n_frames is 11.  So just make sure it's bounded b/w 10 and 11:
        self.assertTrue(
            (exp_time * (n_frames - 1) <= (1+self.tol) * exp_time_expected) and
            (exp_time * n_frames >= (1-self.tol) * exp_time_expected)
                        )

    def test_invalid_string(self):
        """invalid inputs fail as expected"""
        # sequence_name
        for perr in strlist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_exp_time(perr, self.snr,
                                           self.scale, self.scale_bright,
                                           self.manual)

    def test_invalid_real_positive_scalar(self):
        """invalid inputs fail as expected"""
        # snr (but allowed to be None)
        for perr in rpslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.calc_exp_time(self.sequence_name, perr,
                                           self.scale, self.scale_bright,
                                           self.manual)

        # scale
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_exp_time(self.sequence_name,
                                           self.snr, perr, self.scale_bright,
                                           self.manual)

        # scale_bright
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_exp_time(self.sequence_name,
                                           self.snr, self.scale, perr,
                                           self.manual)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""
        # manual
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_exp_time(self.sequence_name,
                                           self.snr, self.scale,
                                           self.scale_bright, perr)

    def test_scale_bright_less_scale(self):
        """scale_bright must be >= scale."""
        scale_bright = self.scale/2
        with self.assertRaises(ValueError):
            self.cgieetc.calc_exp_time(self.sequence_name,
                                           self.snr, self.scale,
                                           scale_bright, self.manual)


class TestCGIEETCCalcExpTimeResel(unittest.TestCase):
    """
    Unit tests for calc_exp_time_resel method of CGIEETC class.
    """

    def setUp(self):
        self.local_path = LOCAL_PATH

        self.mag = 15.8
        self.phot = 'v'
        self.spt = 'G2V'
        self.ut_pointer_path = str(Path(self.local_path, 'ut_pointer.yaml'))

        self.cgieetc = CGIEETC(self.mag, self.phot, self.spt,
                               self.ut_pointer_path)
        self.cgieetc.excam_config['overhead'] = 0

        # peak_flux_ratio_pix = 0.5 for this sequence
        self.sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_1_UT'
        self.sequence_name2 = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'
        self.cfam = '1F'  # cfam associated with this sequence
        self.snr = 5.
        self.fraction = 0.01
        self.num_pixels = 15
        self.manual = 1.0

        self.tol = 0.05

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        '''Successful run of the method. Includes the case of manual as
        something other than 1 and SNR as None.'''

        self.cgieetc.calc_exp_time_resel(self.sequence_name, self.snr,
            self.fraction, self.num_pixels, manual=1)

        self.cgieetc.calc_exp_time_resel(self.sequence_name, None,
            self.fraction, self.num_pixels, manual=2)

        # run with None defaults for fraction and num_pixels
        self.cgieetc.calc_exp_time_resel(
            'CGI_SEQ_DATA_SECONDARY_PR_INFOCUS', self.snr, None, None,
            manual=1)

    def test_biggest_scale(self):
        """Test that biggest possible value for fraction gives successful
        run.  And the result should agree with the result from
        calc_exp_time(). """
        #ratio of peak-pixel flux to total flux:
        this_seq = self.cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        #fraction/(x*num_pixels) must be 1 or less, so test limiting case
        fraction = x*num_pixels

        n1, t1, g1, snr_resel, opt1 = self.cgieetc.calc_exp_time_resel(
            self.sequence_name, self.snr, fraction, num_pixels)
        #calc_exp_time() is for one pixel, so it should agree
        n2, t2, g2, snr_pp, opt2 = self.cgieetc.calc_exp_time(
            self.sequence_name, self.snr)
        self.assertEqual(n1, n2)
        self.assertEqual(t1, t2)
        self.assertEqual(g1, g2)
        self.assertEqual(snr_resel, snr_pp)
        self.assertEqual(opt1, opt2)

    def test_equivalent_inputs(self):
        """If the fraction is decreased by some factor, the
        number of pixels would also decrease by the same factor, and the SNR
        per pixel should not change much, as long as the cosmic ray correction
        to the SNR is small, and it should be for a small number of pixels and
        a small exposure time.  The other outputs shouldn't change, as
        long as constraints aren't hit with either set of inputs."""
        num_pixels = 2
        n, t, g, snr, opt = \
            self.cgieetc.calc_exp_time_resel(self.sequence_name, self.snr,
            self.fraction, num_pixels)

        snr_pp = snr/np.sqrt(num_pixels)

        f2 = self.fraction/2
        num2 = num_pixels/2

        #comparable target SNR per resel would be (SNR per pixel)*np.sqrt(num2)
        tar_snr_pp = self.snr/np.sqrt(num_pixels)
        tar_snr2 = tar_snr_pp*np.sqrt(num2)

        n2, t2, g2, snr2, opt2 = \
            self.cgieetc.calc_exp_time_resel(self.sequence_name, tar_snr2,
            f2, num2)

        snr_pp2 = snr2/np.sqrt(num2)

        self.assertTrue(n == n2)
        self.assertTrue(np.isclose(t, t2, rtol=1e-2))
        self.assertTrue(np.isclose(g, g2, rtol=1e-2))
        self.assertTrue(np.isclose(snr_pp, snr_pp2, rtol=1e-2))
        self.assertTrue(opt == opt2)


    def test_success_fixedt(self):
        """Verify function completes without errors given valid inputs."""
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        cgieetc.calc_exp_time_resel(self.sequence_name,
                              self.snr,
                              self.fraction,
                              self.num_pixels)
        pass


    def test_success_fixedn(self):
        """Verify function completes without errors given valid inputs."""

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        cgieetc.calc_exp_time_resel(self.sequence_name,
                              self.snr,
                              self.fraction,
                              self.num_pixels)
        pass


    def test_success_fixedg(self):
        """Verify function completes without errors given valid inputs."""

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.calc_exp_time_resel(self.sequence_name,
                              self.snr,
                              self.fraction,
                              self.num_pixels)
        pass


    def test_failure_two_or_more_fixed(self):
        """Verify unsupported constraint configurations caught"""

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_exp_time_resel(self.sequence_name,
                              self.snr,
                              self.fraction,
                              self.num_pixels)


        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_exp_time_resel(self.sequence_name,
                              self.snr,
                              self.fraction,
                              self.num_pixels)

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_exp_time_resel(self.sequence_name,
                              self.snr,
                              self.fraction,
                              self.num_pixels)

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_exp_time_resel(self.sequence_name,
                              self.snr,
                              self.fraction,
                              self.num_pixels)

        pass


    def test_success_None(self):
        """
        Verify function completes without errors given valid inputs.

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.calc_exp_time_resel(self.sequence_name,
                              None,
                              self.fraction,
                              self.num_pixels)
        pass


    def test_success_fixedt_None(self):
        """
        Verify function completes without errors given valid inputs.

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        cgieetc.calc_exp_time_resel(self.sequence_name,
                              None,
                              self.fraction,
                              self.num_pixels)
        pass


    def test_success_fixedn_None(self):
        """
        Verify function completes without errors given valid inputs.

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        cgieetc.calc_exp_time_resel(self.sequence_name,
                              None,
                              self.fraction,
                              self.num_pixels)
        pass


    def test_success_fixedg_None(self):
        """
        Verify function completes without errors given valid inputs.

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.calc_exp_time_resel(self.sequence_name,
                              None,
                              self.fraction,
                              self.num_pixels)
        pass



    def test_calc_exp_time_resel(self):
        """Verify that correct exposure times are being calculated."""
        mag = 14.07
        snr = 500
        exp_time_expected = 17.
        sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        this_seq = cgieetc.sequences[sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces the case of calc_exp_time()
        n_frames, exp_time, _, _, _ = cgieetc.calc_exp_time_resel(
                                                            sequence_name,
                                                            snr,
                                                            fraction,
                                                            num_pixels)

        self.assertTrue((np.abs((exp_time * n_frames) - exp_time_expected))
                        / exp_time_expected <= self.tol)

    def test_opt_flag(self):
        """
        Verify the optimization flag is set correctly when you ask for an SNR
        that is too high to be achieved within camera parameter boundaries.
        """
        mag = 14.07
        manual = 0.1  # OD1
        snr = 500
        snr_high = 5e12
        sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        this_seq = cgieetc.sequences[sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces the case of calc_exp_time()
        _, _, _, _, optflag = cgieetc.calc_exp_time_resel(sequence_name,
                                                    snr,
                                                    fraction,
                                                    num_pixels, manual=manual)
        _, _, _, _, optflag_high = cgieetc.calc_exp_time_resel(
                                                         sequence_name,
                                                         snr_high, fraction,
                                                         num_pixels,
                                                         manual=manual)

        self.assertTrue(optflag == 0)
        self.assertTrue(optflag_high == 1)

    def test_high_snr_same_as_None(self):
        """
        Verify SNR-too-high gives the same result as no SNR constraint at all
        """
        snr_high = 5e30

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        this_seq = cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces the case of calc_exp_time()
        n_frames1, exp_time1, gain1, _, _ = cgieetc.calc_exp_time_resel(
                                                    self.sequence_name,
                                                    None,
                                                    fraction,
                                                    num_pixels)
        n_frames2, exp_time2, gain2, _, _ = cgieetc.calc_exp_time_resel(
                                                         self.sequence_name,
                                                         snr_high,
                                                         fraction,
                                                         num_pixels)

        self.assertTrue(n_frames1 == n_frames2)
        self.assertTrue(np.max(np.abs(exp_time1 - exp_time2)) < self.tol)
        self.assertTrue(np.max(np.abs(gain1 - gain2)) < self.tol)
        pass



    def test_nd_filter_exp_time(self):
        """Verify ND filter is being applied correctly."""
        mag = 14.07
        manual = 0.1  # OD1
        snr = 500
        sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'

        exp_time_expected = 17./manual  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        this_seq = cgieetc.sequences[sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces the case of calc_exp_time()
        n_frames, exp_time, _, _, _ = cgieetc.calc_exp_time_resel(
                                                            sequence_name,
                                                            snr,
                                                            fraction,
                                                            num_pixels,
                                                            manual=manual)

        # actual non-integer # of frames from excam_tools is 10.6, and
        # n_frames is 11.  So just make sure it's bounded b/w 10 and 11:
        self.assertTrue(
            (exp_time * (n_frames - 1) <= (1+self.tol) * exp_time_expected) and
            (exp_time * n_frames >= (1-self.tol) * exp_time_expected)
                        )

    def test_flux_ratio_psf(self):
        """Verify the flux ratio parameter is being used correctly."""
        mag = 14.07
        snr = 500
        # Unit test sequence that has a peak flux ratio of 0.5, compared to 1
        # in self.sequence_name
        sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_1_UT'

        exp_time_expected = 17 * 2.  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        this_seq = cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces the case of calc_exp_time()
        n_frames, exp_time, _, _, _ = cgieetc.calc_exp_time_resel(
                                                            sequence_name,
                                                            snr,
                                                            fraction,
                                                            num_pixels)
        # actual non-integer # of frames from excam_tools is around 10.6, and
        # n_frames is 11.  So just make sure it's bounded b/w 10 and 11:
        self.assertTrue(
            (exp_time * (n_frames - 1) <= (1+self.tol) * exp_time_expected) and
            (exp_time * n_frames >= (1-self.tol) * exp_time_expected)
                        )

    def test_scale_factor(self):
        """Verify scale factor is being applied correctly."""
        mag = 3.49

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        this_seq = cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces the case of calc_exp_time()

        snr = 50.
        # 10^-6 contrast (and 'converted' to a scale applied to resel)
        scale = 1e-6/x
        scale_bright = scale  # let's say it's also the brightest speckle


        #old: the answer (V of 3.5, 5700 K star, SNR of 50 yields t_exp=11.8 s)
        #old: the answer (V of 3.5, 5770 K star, SNR of 50 yields t_exp=12.4 s)
        exp_time_expected = 11.6    # seconds
        n_frames, exp_time, _, _, _ = cgieetc.calc_exp_time_resel(
                                                            self.sequence_name,
                                                            snr,
                                                            fraction,
                                                            num_pixels,
                                                            scale,
                                                            scale_bright)

        self.assertTrue((np.abs((exp_time * n_frames) - exp_time_expected))
                        / exp_time_expected <= self.tol)

    def test_invalid_string(self):
        '''Invalid inputs fail as expected.'''
        #sequence_name
        for perr in strlist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_exp_time_resel(perr, self.snr,
            self.fraction, self.num_pixels)

    def test_invalid_real_positive_scalar(self):
        """invalid inputs fail as expected"""
        # snr (but allowed to be None)
        for perr in rpslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.calc_exp_time_resel(self.sequence_name, perr,
                                           self.fraction, self.num_pixels)

        # fraction
        for perr in rpslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.calc_exp_time_resel(self.sequence_name,
                                           self.snr, perr, self.num_pixels)

        # scale
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                           self.snr, self.fraction,
                                           self.num_pixels, scale=perr)

        # scale_bright
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                           self.snr, self.fraction,
                                           self.num_pixels, scale_bright=perr)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""
        # num_pixels
        for perr in rnslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.calc_exp_time_resel(self.sequence_name,
                                           self.snr, self.fraction, perr)

        # manual
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_exp_time_resel(self.sequence_name,
                                           self.snr, self.fraction,
                                           self.num_pixels, manual=perr)

    def test_fraction_less_than_1(self):
        '''Fraction must be 1 or less.'''
        fraction = 1.1
        with self.assertRaises(ValueError):
            self.cgieetc.calc_exp_time_resel(self.sequence_name,
                                           self.snr, fraction,
                                           self.num_pixels)

    def test_fraction_null(self):
        '''Fraction is null.'''
        with self.assertRaises(ValueError):
            self.cgieetc.calc_exp_time_resel(self.sequence_name2,
                                           self.snr, None,
                                           self.num_pixels)

    def test_num_pixels_null(self):
        '''num_pixels is null.'''
        with self.assertRaises(ValueError):
            self.cgieetc.calc_exp_time_resel(self.sequence_name,
                                           self.snr, self.fraction,
                                           None)

    def test_scale_bright_less_scale(self):
        '''A ValueError is raised if scale_bright < scale.'''
        # The ratio of fraction/num_pixels must be <= the ratio of the
        # peak pixel's flux to the total flux (which is < 1).  So if
        # fraction/num_pixels is bigger than 1:
        num_pixels = self.fraction/1.1
        with self.assertRaises(ValueError):
            self.cgieetc.calc_exp_time_resel(self.sequence_name,
                                           self.snr, self.fraction,
                                           num_pixels)


class TestCGIEETCCalcLOCAMGain(unittest.TestCase):
    """
    Unit tests for calc_locam_gain method of CGIEETC class.
    """

    def setUp(self):
        self.local_path = LOCAL_PATH

        self.mag = 11.8
        self.phot = 'v'
        self.spt = 'G2V'
        self.ut_pointer_path = str(Path(self.local_path, 'ut_pointer.yaml'))

        self.cgieetc = CGIEETC(self.mag, self.phot, self.spt,
                               self.ut_pointer_path)

        self.sequence_name = 'LOCAM_NFOV_DM'
        self.cfam = '1F'  # cfam associated with this sequence
        self.manual = 1.0

        self.tol = 0.05

        pass

    def test_calc_locam_gain(self):
        """Verify that correct gains are being calculated."""
        mag = 2.0 # must be quite high; we can take a lot of light with a
                  # quasi-pupil-plane image and a 441us exposure time per frame
                  # and it will sit at the max allowed gain rail
        target = 31

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.locam_config['g_max_age'] = 200
        cgieetc.locam_config['e_max_age'] = 12800
        gain, _ = cgieetc.calc_locam_gain(self.sequence_name, self.manual)

        self.assertTrue(np.max(np.abs((gain - target)/target)) < self.tol)



    def test_invalid_string(self):
        """invalid inputs fail as expected"""
        # sequence_name
        for perr in strlist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_locam_gain(perr, self.manual)


    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""
        # manual
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_locam_gain(self.sequence_name, perr)


class TestCGIEETCCalcPCExpTime(unittest.TestCase):
    """
    Unit tests for calc_pc_exp_time method of CGIEETC class.
    """

    def setUp(self):
        self.local_path = LOCAL_PATH

        self.mag = 27.8
        self.phot = 'v'
        self.spt = 'G2V'
        self.ut_pointer_path = str(Path(self.local_path, 'ut_pointer.yaml'))

        self.cgieetc = CGIEETC(self.mag, self.phot, self.spt,
                               self.ut_pointer_path)

        self.sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'
        self.cfam = '1F'  # cfam associated with this sequence
        self.snr = 5.
        self.scale = 1.
        self.scale_bright = 1.
        self.manual = 1.0

        self.tol = 0.05

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """Verify function completes without errors given valid inputs."""
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.calc_pc_exp_time(self.sequence_name,
                              self.snr,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        pass


    def test_failure_fixedt(self):
        """Verify unsupported constraint configurations caught"""
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time(self.sequence_name,
                                  self.snr,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)
        pass


    def test_success_fixedn(self):
        """Verify function completes without errors given valid inputs."""

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        cgieetc.calc_pc_exp_time(self.sequence_name,
                              self.snr,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        pass


    def test_failure_fixedg(self):
        """Verify unsupported constraint configurations caught"""

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time(self.sequence_name,
                                  self.snr,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)
        pass


    def test_failure_two_or_more_fixed(self):
        """Verify unsupported constraint configurations caught"""

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time(self.sequence_name,
                                  self.snr,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)


        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time(self.sequence_name,
                                  self.snr,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time(self.sequence_name,
                                  self.snr,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time(self.sequence_name,
                                  self.snr,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)

        pass


    def test_success_None(self):
        """
        Verify function completes without errors given valid inputs.

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.calc_pc_exp_time(self.sequence_name,
                              None,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        pass


    def test_failure_fixedt_None(self):
        """
        Verify unsupported constraint configurations caught

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time(self.sequence_name,
                                  None,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)
        pass


    def test_success_fixedn_None(self):
        """
        Verify function completes without errors given valid inputs.

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        cgieetc.calc_pc_exp_time(self.sequence_name,
                              None,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        pass


    def test_failure_fixedg_None(self):
        """
        Verify unsupported constraint configurations caught

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time(self.sequence_name,
                                  None,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)
        pass

    def test_invalid_string(self):
        """invalid inputs fail as expected"""
        # sequence_name
        for perr in strlist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time(perr, self.snr,
                                           self.scale, self.scale_bright,
                                           self.manual)

    def test_invalid_real_positive_scalar(self):
        """invalid inputs fail as expected"""
        # snr (but allowed to be None)
        for perr in rpslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time(self.sequence_name, perr,
                                           self.scale, self.scale_bright,
                                           self.manual)

        # scale
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time(self.sequence_name,
                                           self.snr, perr, self.scale_bright,
                                           self.manual)

        # scale_bright
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time(self.sequence_name,
                                           self.snr, self.scale, perr,
                                           self.manual)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""
        # manual
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time(self.sequence_name,
                                           self.snr, self.scale,
                                           self.scale_bright, perr)

    def test_scale_bright_less_scale(self):
        """scale_bright must be >= scale."""
        scale_bright = self.scale/2
        with self.assertRaises(ValueError):
            self.cgieetc.calc_exp_time(self.sequence_name,
                                           self.snr, self.scale,
                                           scale_bright, self.manual)


class TestCGIEETCCalcPCExpTimeResel(unittest.TestCase):
    """
    Unit tests for calc_pc_exp_time_resel method of CGIEETC class.
    """

    def setUp(self):
        self.local_path = LOCAL_PATH

        self.mag = 15.8
        self.phot = 'v'
        self.spt = 'G2V'
        self.ut_pointer_path = str(Path(self.local_path, 'ut_pointer.yaml'))

        self.cgieetc = CGIEETC(self.mag, self.phot, self.spt,
                               self.ut_pointer_path)

        # peak_flux_ratio_pix = 0.5 for this sequence
        self.sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_1_UT'
        self.sequence_name2 = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'
        self.cfam = '1F'  # cfam associated with this sequence
        self.snr = 1.
        self.fraction = 0.1# 1e-10
        self.num_pixels = 15
        self.manual = 1e-6 # to ensure success for pc case

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        '''Successful run of the method. Includes the case of various manual
        values and None for SNR.'''

        self.cgieetc.calc_pc_exp_time_resel(self.sequence_name, self.snr,
            self.fraction, self.num_pixels, manual=self.manual)

        self.cgieetc.calc_pc_exp_time_resel(self.sequence_name, None,
            self.fraction, self.num_pixels, manual=self.manual/2)

        # run with None defaults for fraction and num_pixels
        self.cgieetc.calc_pc_exp_time_resel(
            'CGI_SEQ_DATA_SECONDARY_PR_INFOCUS', self.snr, None, None,
            manual=0.001)

    def test_biggest_scale(self):
        """Test that biggest possible value for fraction gives successful
        run.  And test that it agrees with result from calc_pc_exp_time()."""
        #ratio of peak-pixel flux to total flux:
        this_seq = self.cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        #fraction/(x*num_pixels) must be 1 or less, so test limiting case
        fraction = x*num_pixels
        snr = 1

        n1, t1, g1, snr_resel, opt1 = self.cgieetc.calc_pc_exp_time_resel(
            self.sequence_name, snr, fraction, num_pixels, manual=self.manual)
        #calc_pc_exp_time() is for one pixel, so it should agree
        n2, t2, g2, snr_pp, opt2 = self.cgieetc.calc_pc_exp_time(
            self.sequence_name, snr, manual=self.manual)
        self.assertEqual(n1, n2)
        self.assertEqual(t1, t2)
        self.assertEqual(g1, g2)
        self.assertEqual(snr_resel, snr_pp)
        self.assertEqual(opt1, opt2)

    def test_equivalent_inputs(self):
        """If the fraction is decreased by some factor, the
        number of pixels would also decrease by the same factor, and the SNR
        per pixel should not change much, as long as the effect of cosmic ray
        correction to the SNR is small, and it should be for a small number of
        pixels and a small exposure time.  The other outputs shouldn't change,
        as long as constraints aren't hit with either set of inputs."""
        num_pixels = 2
        tar_snr = 0.01
        n, t, g, snr, opt = \
            self.cgieetc.calc_pc_exp_time_resel(self.sequence_name,
            tar_snr, self.fraction, num_pixels, manual=self.manual)

        snr_pp = snr/np.sqrt(num_pixels)

        f2 = self.fraction/2
        num2 = num_pixels/2

        #comparable target SNR per resel would be (SNR per pixel)*np.sqrt(num2)
        tar_snr_pp = tar_snr/np.sqrt(num_pixels)
        tar_snr2 = tar_snr_pp*np.sqrt(num2)

        n2, t2, g2, snr2, opt2 = \
            self.cgieetc.calc_pc_exp_time_resel(self.sequence_name,
            tar_snr2, f2, num2, manual=self.manual)

        snr_pp2 = snr2/np.sqrt(num2)

        self.assertTrue(n == n2)
        self.assertTrue(np.isclose(t, t2, atol=0.5))
        self.assertTrue(np.isclose(g, g2, atol=0.01))
        self.assertTrue(np.isclose(snr_pp, snr_pp2, atol=0.01))
        self.assertTrue(opt == opt2)


    def test_failure_fixedt(self):
        """Verify unsupported constraint configurations caught"""
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                  self.snr,
                                  self.fraction,
                                  self.num_pixels, manual=self.manual)
        pass


    def test_success_fixedn(self):
        """Verify function completes without errors given valid inputs."""

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                              self.snr,
                              self.fraction,
                              self.num_pixels, manual=self.manual)
        pass


    def test_failure_fixedg(self):
        """Verify unsupported constraint configurations caught"""

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                  self.snr,
                                  self.fraction,
                                  self.num_pixels, manual=self.manual)
        pass


    def test_failure_two_or_more_fixed(self):
        """Verify unsupported constraint configurations caught"""

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                  self.snr,
                                  self.fraction,
                                  self.num_pixels, manual=self.manual)


        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                  self.snr,
                                  self.fraction,
                                  self.num_pixels, manual=self.manual)

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                  self.snr,
                                  self.fraction,
                                  self.num_pixels, manual=self.manual)

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                  self.snr,
                                  self.fraction,
                                  self.num_pixels, manual=self.manual)

        pass


    def test_success_None(self):
        """
        Verify function completes without errors given valid inputs.

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                              None,
                              self.fraction,
                              self.num_pixels, manual=self.manual)
        pass


    def test_failure_fixedt_None(self):
        """
        Verify unsupported constraint configurations caught

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                  None,
                                  self.fraction,
                                  self.num_pixels, manual=self.manual)
        pass


    def test_success_fixedn_None(self):
        """
        Verify function completes without errors given valid inputs.

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                              None,
                              self.fraction,
                              self.num_pixels, manual=self.manual)
        pass


    def test_failure_fixedg_None(self):
        """
        Verify unsupported constraint configurations caught

        'None' is a valid input for snr
        """
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                  None,
                                  self.fraction,
                                  self.num_pixels, manual=self.manual)
        pass

    def test_invalid_string(self):
        '''Invalid inputs fail as expected.'''
        #sequence_name
        for perr in strlist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time_resel(perr, self.snr,
            self.fraction, self.num_pixels, manual=self.manual)

    def test_invalid_real_positive_scalar(self):
        """invalid inputs fail as expected"""
        # snr (but allowed to be None)
        for perr in rpslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time_resel(self.sequence_name, perr,
                                           self.fraction, self.num_pixels,
                                           manual=self.manual)

        # fraction
        for perr in rpslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                           self.snr, perr, self.num_pixels,
                                           manual=self.manual)

        # scale
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                           self.snr, self.fraction,
                                           self.num_pixels, scale=perr,
                                           manual=self.manual)

        # scale_bright
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                           self.snr, self.fraction,
                                           self.num_pixels, scale_bright=perr,
                                           manual=self.manual)


    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""
        # num_pixels
        for perr in rnslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                           self.snr, self.fraction, perr,
                                           manual=self.manual)

        # manual
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                           self.snr, self.fraction,
                                           self.num_pixels, manual=perr)


    def test_fraction_less_than_1(self):
        '''Fraction must be 1 or less.'''
        fraction = 1.1
        with self.assertRaises(ValueError):
            self.cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                           self.snr, fraction,
                                           self.num_pixels, manual=self.manual)

    def test_fraction_null(self):
        '''Fraction is null.'''
        with self.assertRaises(ValueError):
            self.cgieetc.calc_pc_exp_time_resel(self.sequence_name2,
                                           self.snr, None,
                                           self.num_pixels)

    def test_num_pixels_null(self):
        '''num_pixels is null.'''
        with self.assertRaises(ValueError):
            self.cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                           self.snr, self.fraction,
                                           None)

    def test_scale_bright_less_scale(self):
        '''A ValueError is raised if scale_bright < scale.'''
        # The ratio of fraction/num_pixels must be <= the ratio of the
        # peak pixel's flux to the total flux (which is < 1).  So if
        # fraction/num_pixels is bigger than 1:
        num_pixels = self.fraction/1.1
        with self.assertRaises(ValueError):
            self.cgieetc.calc_pc_exp_time_resel(self.sequence_name,
                                           self.snr, self.fraction,
                                           num_pixels, manual=self.manual)


class TestCGIEETCCalcPCConstIntTime(unittest.TestCase):
    """
    Unit tests for calc_pc_const_int_time method of CGIEETC class.
    """

    def setUp(self):
        self.local_path = LOCAL_PATH

        self.mag = 27.8
        self.phot = 'v'
        self.spt = 'G2V'
        self.ut_pointer_path = str(Path(self.local_path, 'ut_pointer.yaml'))

        self.cgieetc = CGIEETC(self.mag, self.phot, self.spt,
                               self.ut_pointer_path)

        self.sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'
        self.cfam = '1F'  # cfam associated with this sequence
        self.t_tot = 200.
        self.scale = 1.
        self.scale_bright = 1.
        self.manual = 0.1

        self.tol = 0.05

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """Verify function completes without errors given valid inputs."""
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.calc_pc_const_int_time(self.sequence_name,
                              self.t_tot,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        cgieetc.calc_pc_const_int_time(self.sequence_name,
                              self.t_tot,
                              self.scale,
                              self.scale_bright,
                              self.manual,
                              hard_limit=False)
        pass


    def test_failure_one_or_more_fixed(self):
        """Verify unsupported constraint configurations caught"""
        # fixed g
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_const_int_time(self.sequence_name,
                                  self.t_tot,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)
        # fixed N
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_const_int_time(self.sequence_name,
                                  self.t_tot,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)
        # fixed t
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_const_int_time(self.sequence_name,
                                  self.t_tot,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)


    def test_reasonable_outputs(self):
        """Verify that outputs are reasonable."""
        int_time = 17
        manual = 1

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        n_frames, exp_time, gain, snr, t_tot_out = \
                                                cgieetc.calc_pc_const_int_time(
                                                            self.sequence_name,
                                                            int_time,
                                                            self.scale,
                                                            self.scale_bright,
                                                            manual)
        # assuming saturation is avoided, the highest number of frames is
        # usually what maximizes SNR, which minimizes exposure time:
        self.assertTrue(np.isclose(exp_time, cgieetc.excam_config['tmin'],
                                    atol=2))
        # as expected for hard_limit=True:
        self.assertTrue(t_tot_out == int_time)

        # now compare to output of calc_pc_exp_time() using max snr achieved
        # for a total integration time self.t_tot as the target SNR:
        n, t, g, snr2, _ = cgieetc.calc_pc_exp_time(self.sequence_name, snr,
                                                self.scale, self.scale_bright,
                                                manual)
        self.assertTrue(n == n_frames)
        self.assertTrue(np.isclose(t, exp_time, rtol=0.05))
        self.assertTrue(np.isclose(g, gain, rtol=0.05))
        self.assertTrue(np.isclose(snr, snr2, rtol=0.05))


    def test_nd_filter_exp_time(self):
        """Verify ND filter is being applied correctly."""
        mag = 25
        manual = self.manual/2

        int_time = 20  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        n_frames, exp_time, _, _, _ = cgieetc.calc_pc_const_int_time(
                                                            self.sequence_name,
                                                            int_time,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # using manual:
        n_frames2, exp_time2, _, _, _ = cgieetc.calc_pc_const_int_time(
                                                            self.sequence_name,
                                                            int_time,
                                                            self.scale,
                                                            self.scale_bright,
                                                            manual)
        # with a factor of 1/2 applied, the exposure time should be increased
        # by roughly a factor of 2, which would
        # make N decreased by a factor of 2:
        self.assertTrue(np.isclose(n_frames2, n_frames/2))
        self.assertTrue(np.isclose(exp_time2, exp_time*2, rtol=0.05))


    def test_scale_factor(self):
        """Verify scale factor is being applied correctly."""
        mag = 24.07
        scale = 0.5
        scale_bright = 0.5

        int_time = 17  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        n_frames, exp_time, _, _, _ = cgieetc.calc_pc_const_int_time(
                                                            self.sequence_name,
                                                            int_time,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # using scale and scale_bright:
        n_frames2, exp_time2, _, _, _ = cgieetc.calc_pc_const_int_time(
                                                            self.sequence_name,
                                                            int_time,
                                                            scale,
                                                            scale_bright,
                                                            self.manual)
        # with 0.5 applied, the exposure time should be increased by roughly
        # a factor of 2, which would make N decreased by a factor of 2:
        self.assertTrue(np.isclose(n_frames2, n_frames/2))
        self.assertTrue(np.isclose(exp_time2, exp_time*2, rtol=0.05))

    def test_flux_ratio_psf(self):
        """Verify the flux ratio parameter is being used correctly."""
        mag = 24.07

        # Unit test sequence that has a peak flux ratio of 0.5, compared to 1
        # in self.sequence_name
        sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_1_UT'

        int_time = 17.  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        n_frames, exp_time, _, _, _ = cgieetc.calc_pc_const_int_time(
                                                            self.sequence_name,
                                                            int_time,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # using sequence_name:
        n_frames2, exp_time2, _, _, _ = cgieetc.calc_pc_const_int_time(
                                                            sequence_name,
                                                            int_time,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # with 0.5 applied, the exposure time should be increased by roughly
        # a factor of 2, which would make N decreased by a factor of 2:
        self.assertTrue(np.isclose(n_frames2, n_frames/2))
        self.assertTrue(np.isclose(exp_time2, exp_time*2, rtol=0.05))

    def test_invalid_string(self):
        """invalid inputs fail as expected"""
        # sequence_name
        for perr in strlist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time(perr, self.t_tot,
                                           self.scale, self.scale_bright,
                                           self.manual)

    def test_invalid_real_positive_scalar(self):
        """invalid inputs fail as expected"""
        # scale
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time(self.sequence_name,
                                           self.t_tot, perr, self.scale_bright,
                                           self.manual)

        # scale_bright
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time(self.sequence_name,
                                           self.t_tot, self.scale, perr,
                                           self.manual)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""
        # t_tot
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time(self.sequence_name, perr,
                                           self.scale, self.scale_bright,
                                           self.manual)

        # manual
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time(self.sequence_name,
                                           self.t_tot, self.scale,
                                           self.scale_bright, perr)

    def test_scale_bright_less_scale(self):
        """scale_bright must be >= scale."""
        scale_bright = self.scale/2
        with self.assertRaises(ValueError):
            self.cgieetc.calc_pc_const_int_time(self.sequence_name,
                                           self.t_tot, self.scale,
                                           scale_bright, self.manual)

    def test_invalid_hard_limit(self):
        '''Invalid hard_limit caught.'''
        check_list = [2, 'foo', -3.4]
        for perr in check_list:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time(self.sequence_name,
                                           self.t_tot, self.scale,
                                           self.scale_bright, self.manual,
                                           hard_limit=perr)


class TestCGIEETCCalcPCConstIntTimeResel(unittest.TestCase):
    """
    Unit tests for calc_pc_const_int_time_resel method of CGIEETC class.
    """

    def setUp(self):
        self.local_path = LOCAL_PATH

        self.mag = 27.8
        self.phot = 'v'
        self.spt = 'G2V'
        self.ut_pointer_path = str(Path(self.local_path, 'ut_pointer.yaml'))

        self.cgieetc = CGIEETC(self.mag, self.phot, self.spt,
                               self.ut_pointer_path)

        self.sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_0_UT'
        self.sequence_name2 = 'CGI_SEQ_NFOV_ALIGN_LSAM_1_UT'
        self.cfam = '1F'  # cfam associated with this sequence
        self.t_tot = 200.
        self.scale = 1.
        self.scale_bright = 1.
        self.manual = 0.1
        self.fraction = 0.1
        self.num_pixels = 15

        self.tol = 0.05

        # The SLSQP optimizer sometimes has known internal weirdness about
        # bounds and scipy will raise a warning that we can't do anything
        # about.  Filter it out.
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
        pass

    def test_success(self):
        """Verify function completes without errors given valid inputs."""
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                              self.t_tot,
                              self.fraction,
                              self.num_pixels,
                              self.scale,
                              self.scale_bright,
                              self.manual)
        cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                              self.t_tot,
                              self.fraction,
                              self.num_pixels,
                              self.scale,
                              self.scale_bright,
                              self.manual,
                              hard_limit=False)
        # drawing fraction and num_pixels from sequence
        cgieetc.calc_pc_const_int_time_resel(
                              'CGI_SEQ_DATA_SECONDARY_PR_INFOCUS',
                              self.t_tot,
                              None,
                              None,
                              self.scale,
                              self.scale_bright,
                              self.manual,
                              hard_limit=False)
        pass


    def test_failure_one_or_more_fixed(self):
        """Verify unsupported constraint configurations caught"""
        # fixed g
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['gconst'] = 3
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                  self.t_tot,
                                  self.fraction,
                                  self.num_pixels,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)
        # fixed N
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['Nmax'] = cgieetc.excam_config['Nmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                  self.t_tot,
                                  self.fraction,
                                  self.num_pixels,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)
        # fixed t
        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['tmax'] = cgieetc.excam_config['tmin']
        with self.assertRaises(EXCAMOptimizeException):
            cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                  self.t_tot,
                                  self.fraction,
                                  self.num_pixels,
                                  self.scale,
                                  self.scale_bright,
                                  self.manual)

    def test_biggest_scale(self):
        """Test that biggest possible value for fraction gives successful
        run.  And the result should agree with the result from
        calc_gain_fixed_Ntime(). """
        #ratio of peak-pixel flux to total flux:
        this_seq = self.cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        #fraction/(x*num_pixels) must be 1 or less, so test limiting case
        fraction = x*num_pixels
        t_tot = 20

        n1, t1, g1, snr_resel, t_tot_out1 = \
                self.cgieetc.calc_pc_const_int_time_resel(
                    self.sequence_name, t_tot, fraction, num_pixels)
        #calc_const_int_time() is for one pixel, so it should agree
        n2, t2, g2, snr_pp, t_tot_out2 = \
            self.cgieetc.calc_pc_const_int_time(self.sequence_name,
                                                   t_tot)
        self.assertEqual(n1, n2)
        self.assertEqual(t1, t2)
        self.assertEqual(g1, g2)
        self.assertEqual(snr_resel, snr_pp)
        self.assertEqual(t_tot_out1, t_tot_out2)

    def test_equivalent_inputs(self):
        """If the fraction is decreased by some factor, the
        number of pixels would also decrease by the same factor, and the SNR
        per pixel should not change much, as long as the cosmic ray correction
        to the SNR is small, and it should be for a small number of pixels and
        a small exposure time.  The other outputs shouldn't change, as
        long as constraints aren't hit with either set of inputs."""
        num_pixels = 2
        t_tot = 20
        n, t, g, snr, t_tot_out = \
            self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
            t_tot, self.fraction, num_pixels)

        snr_pp = snr/np.sqrt(num_pixels)

        f2 = self.fraction/2
        num2 = num_pixels/2

        #comparable target SNR per resel would be (SNR per pixel)*np.sqrt(num2)
        #expected_snr2 = snr_pp*np.sqrt(num2)

        n2, t2, g2, snr2, t_tot_out2 = \
            self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
            t_tot, f2, num2)

        snr_pp2 = snr2/np.sqrt(num2)

        self.assertTrue(n == n2)
        self.assertTrue(np.isclose(t, t2, rtol=1e-2))
        self.assertTrue(np.isclose(g, g2, rtol=1e-2))
        self.assertTrue(np.isclose(snr_pp, snr_pp2, rtol=1e-1))
        self.assertTrue(t_tot_out == t_tot_out2)

    def test_reasonable_outputs(self):
        """Verify that outputs are reasonable."""
        int_time = 17
        manual = 1

        cgieetc = CGIEETC(self.mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        this_seq = cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces case of calc_pc_const_int_time()
        n_frames, exp_time, gain, snr, t_tot_out = \
                                    cgieetc.calc_pc_const_int_time_resel(
                                                            self.sequence_name,
                                                            int_time,
                                                            fraction,
                                                            num_pixels,
                                                            self.scale,
                                                            self.scale_bright,
                                                            manual)
        # assuming saturation is avoided, the highest number of frames is
        # usually what maximizes SNR, which minimizes exposure time:
        self.assertTrue(np.isclose(exp_time, cgieetc.excam_config['tmin'],
                                    atol=2))
        # as expected for hard_limit=True:
        self.assertTrue(t_tot_out == int_time)

        #now compare to output of calc_pc_exp_time() using max snr achieved for
        # a total integration time self.t_tot as the target SNR:
        n, t, g, snr2, _ = cgieetc.calc_pc_exp_time(self.sequence_name, snr,
                                                self.scale, self.scale_bright,
                                                manual)
        self.assertTrue(n == n_frames)
        self.assertTrue(np.isclose(t, exp_time, rtol=0.05))
        self.assertTrue(np.isclose(g, gain, rtol=0.05))
        self.assertTrue(np.isclose(snr, snr2, rtol=0.05))


    def test_nd_filter_exp_time(self):
        """Verify ND filter is being applied correctly."""
        mag = 25
        manual = self.manual/2

        int_time = 20  # seconds
        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        this_seq = cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces case of calc_pc_const_int_time()
        n_frames, exp_time, _, _, _ = cgieetc.calc_pc_const_int_time_resel(
                                                            self.sequence_name,
                                                            int_time,
                                                            fraction,
                                                            num_pixels,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # using manual:
        n_frames2, exp_time2, _, _, _ = cgieetc.calc_pc_const_int_time_resel(
                                                            self.sequence_name,
                                                            int_time,
                                                            fraction,
                                                            num_pixels,
                                                            self.scale,
                                                            self.scale_bright,
                                                            manual)
        # with a factor of 1/2 applied, the exposure time should be increased
        # by roughly a factor of 2, which would
        # make N decreased by a factor of 2:
        self.assertTrue(np.isclose(n_frames2, n_frames/2))
        self.assertTrue(np.isclose(exp_time2, exp_time*2, rtol=0.05))


    def test_scale_factor(self):
        """Verify scale factor is being applied correctly."""
        mag = 24.07
        scale = 0.5
        scale_bright = 0.5

        int_time = 17  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        this_seq = cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces case of calc_pc_const_int_time()
        n_frames, exp_time, _, _, _ = cgieetc.calc_pc_const_int_time_resel(
                                                            self.sequence_name,
                                                            int_time,
                                                            fraction,
                                                            num_pixels,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # using scale and scale_bright:
        n_frames2, exp_time2, _, _, _ = cgieetc.calc_pc_const_int_time_resel(
                                                            self.sequence_name,
                                                            int_time,
                                                            fraction,
                                                            num_pixels,
                                                            scale,
                                                            scale_bright,
                                                            self.manual)
        # with 0.5 applied, the exposure time should be increased by roughly
        # a factor of 2, which would make N decreased by a factor of 2:
        self.assertTrue(np.isclose(n_frames2, n_frames/2))
        self.assertTrue(np.isclose(exp_time2, exp_time*2, rtol=0.05))

    def test_flux_ratio_psf(self):
        """Verify the flux ratio parameter is being used correctly."""
        mag = 24.07

        # Unit test sequence that has a peak flux ratio of 0.5, compard to 1
        # in self.sequence_name
        sequence_name = 'CGI_SEQ_NFOV_ALIGN_LSAM_1_UT'

        int_time = 17.  # seconds

        cgieetc = CGIEETC(mag, self.phot, self.spt, self.ut_pointer_path)
        cgieetc.excam_config['overhead'] = 0
        this_seq = cgieetc.sequences[self.sequence_name]
        x = this_seq['peak_flux_ratio_pix']
        num_pixels = 1
        fraction = x*num_pixels # reproduces case of calc_pc_const_int_time()
        n_frames, exp_time, _, _, _ = cgieetc.calc_pc_const_int_time_resel(
                                                            self.sequence_name,
                                                            int_time,
                                                            fraction,
                                                            num_pixels,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # using sequence_name:
        this_seq2 = cgieetc.sequences[sequence_name]
        x2 = this_seq2['peak_flux_ratio_pix']
        num_pixels2 = 1
        fraction2 = x2*num_pixels2 #reproduces case of calc_pc_const_int_time()
        n_frames2, exp_time2, _, _, _ = cgieetc.calc_pc_const_int_time_resel(
                                                            sequence_name,
                                                            int_time,
                                                            fraction2,
                                                            num_pixels2,
                                                            self.scale,
                                                            self.scale_bright,
                                                            self.manual)
        # with 0.5 applied, the exposure time should be increased by roughly
        # a factor of 2, which would make N decreased by a factor of 2:
        self.assertTrue(np.isclose(n_frames2, n_frames/2))
        self.assertTrue(np.isclose(exp_time2, exp_time*2, rtol=0.05))

    def test_invalid_string(self):
        """invalid inputs fail as expected"""
        # sequence_name
        for perr in strlist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time_resel(perr, self.t_tot,
                                           self.fraction, self.num_pixels,
                                           self.scale, self.scale_bright,
                                           self.manual)

    def test_invalid_real_positive_scalar(self):
        """invalid inputs fail as expected"""
        # fraction
        for perr in rpslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, perr,
                                           self.num_pixels,
                                           self.scale, self.scale_bright,
                                           self.manual)

        # scale
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           self.num_pixels,
                                           perr, self.scale_bright,
                                           self.manual)

        # scale_bright
        for perr in rpslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           self.num_pixels, self.scale, perr,
                                           self.manual)

    def test_invalid_real_nonnegative_scalar(self):
        """invalid inputs fail as expected"""
        # t_tot
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                           perr, self.fraction,
                                           self.num_pixels,
                                           self.scale, self.scale_bright,
                                           self.manual)

        # num_pixels
        for perr in rnslist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           perr,
                                           self.scale, self.scale_bright,
                                           self.manual)

        # manual
        for perr in rnslist:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           self.num_pixels, self.scale,
                                           self.scale_bright, perr)

    def test_fraction_less_than_1(self):
        '''Fraction must be 1 or less.'''
        fraction = 1.1
        with self.assertRaises(ValueError):
            self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, fraction,
                                           self.num_pixels, self.scale,
                                           self.scale_bright, self.manual)

    def test_fraction_null(self):
        '''Fraction is null.'''
        with self.assertRaises(ValueError):
            self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, None,
                                           self.num_pixels)

    def test_num_pixels_null(self):
        '''num_pixels is null.'''
        with self.assertRaises(ValueError):
            self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name2,
                                           self.t_tot, self.fraction,
                                           None)

    def test_scale_bright_less_scale_2(self):
        """scale_bright must be >= scale."""
        scale_bright = self.scale/2
        with self.assertRaises(ValueError):
            self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           self.num_pixels, self.scale,
                                           scale_bright, self.manual)

    def test_invalid_hard_limit(self):
        '''Invalid hard_limit caught.'''
        check_list = [2, 'foo', -3.4]
        for perr in check_list:
            with self.assertRaises(TypeError):
                self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           self.num_pixels, self.scale,
                                           self.scale_bright, self.manual,
                                           hard_limit=perr)

    def test_scale_bright_less_scale(self):
        '''A ValueError is raised if scale_bright < scale.'''
        # The ratio of fraction/num_pixels must be <= the ratio of the
        # peak pixel's flux to the total flux (which is < 1).  So if
        # fraction/num_pixels is bigger than 1:
        num_pixels = self.fraction/1.1
        with self.assertRaises(ValueError):
            self.cgieetc.calc_pc_const_int_time_resel(self.sequence_name,
                                           self.t_tot, self.fraction,
                                           num_pixels, self.scale,
                                           self.scale_bright, self.manual)


class TestCGIEETCFunctional(unittest.TestCase):
    """
    Run a CGIEETC object on all of the valid sequences in the repo and
    verify that all complete successfully.
    """

    def test_seq_functional_all_success(self):
        """
        Run through every valid sequence in the default sequence file from
        default pointer_path, and verify that eetc does not raise an
        exception when running it.  Note this is not testing that the numbers
        are anything specific, just the functionality is successful.
        """

        mag = 2.25
        phot = 'v'
        spt = 'O5'

        # use default pointer, it better be consistent with constants.py
        tmp_eetc = CGIEETC(mag=mag, phot=phot, spt=spt)

        for seq in tmp_eetc.sequences:
            # don't need calc_exp_time too, since it calls flux rate anyway
            tmp_eetc.calc_flux_rate(seq)
            pass
        pass



if __name__ == '__main__':
    unittest.main(buffer=True)
