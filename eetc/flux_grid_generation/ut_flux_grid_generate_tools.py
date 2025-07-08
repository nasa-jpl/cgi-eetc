# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit test suite for load module.
"""
import copy
import csv
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import os
import eetc
import eetc.util.ut_check as ut_check
from eetc.flux_grid_generation.flux_grid_generate_tools import (_spectrum_load_csv, _spectrum_scale,
                                                                _spectrum_mag_estimate, _cfam_band_integrate,
                                                                _get_cfam_transmission, spec_generate_cgi_cfam)

LOCAL_PATH = os.path.abspath(os.path.dirname( __file__ ))
UT_CONFIG_PATH = str(Path(LOCAL_PATH, 'ut_config'))

#_spectrum_load_csv
# check that reference spectrum loads appropriately
class TestFluxGridGenerate(unittest.TestCase):
    """
    Unit tests for flux_grid_generate function.
    """

#def spec_generate_cgi_cfam(wvl, mag, filter_band, spt, cfam_filter,
#                           spec_dir, cfam_filter_dir, collecting_area):

    def setUp(self):
        self.spt = 'UT_SPT'
        self.spt_invalid_flux_val = 'ut_spt_invalid_flux_val'
        self.spt_invalid_wave_val = 'ut_spt_invalid_wave_val'
        self.spt_invalid_format = 'ut_spt_invalid_format'
        self.ut_config = os.path.join(UT_CONFIG_PATH,'ut_flux_grid_generate_input_params.yaml')
        self.ut_cfam_folder = os.path.join(LOCAL_PATH,'ut_cfam_filter_curves')
        self.spec_path = os.path.join(LOCAL_PATH, 'ut_bpgs_atlas_csv')
        self.ut_cfam_folder_err = os.path.join(LOCAL_PATH,
                                               'ut_cfam_filter_curves_err')
        self.spec_path_err = os.path.join(LOCAL_PATH, 'ut_bpgs_atlas_csv_err')
        self.test_mag = 0.2
        self.test_filter_band ='ut_filter'
        self.test_cfam = 'UT'
        self.test_cfam_invalid_trans_val = 'UT_INVALID_VAL_TRANS'
        self.test_cfam_invalid_wave_val = 'UT_INVALID_VAL_WAVE'
        self.test_cfam_invalid_format = 'UT_INVALID_FORMAT'
        self.collecting_area = 35895.212  # cm^2
        self.test_wvl = np.arange(3e3, 10e3, 1) # Ang

#_spectrum_load_csv
    def test_spectrum_load_csv_trans(self):
        """
        Verify spectral dummy atlas file is being loaded correctly
        """
        # matches dummy file,
        trans_answer = [1.e8, 1.e8, 1.e8, 1.e8, 1.e8, 1.e8, 1.e8,]
        output_dict = _spectrum_load_csv(self.spt, self.spec_path)

        self.assertTrue(np.array_equal(trans_answer, output_dict['flux']))

    def test_spectrum_load_csv(self):
        """
        Verify spectral dummy atlas file is being loaded correctly
        """
        # matches dummy file
        wvls_answer = [4000., 4500., 5000., 6000., 7000., 8000., 9000.]
        output_dict = _spectrum_load_csv(self.spt, self.spec_path)

        self.assertTrue(np.array_equal(wvls_answer, output_dict['wvl']))

    def test_spectrum_load_csv_invalid_spec_path(self):
        """
        Verify invalid _spectrum_load_csv fails as expected.
        """
        spt_path_err = self.spec_path + '_invalid'

        with self.assertRaises(IOError):
            _spectrum_load_csv(self.spt, spt_path_err)

    def test_spectrum_load_csv_missing_spec_file(self):
        """
        Verify invalid _spectrum_load_csv fails as expected.
        """
        spt_err = self.spt + '_invalid_file_name'

        with self.assertRaises(IOError):
            _spectrum_load_csv(spt_err, self.spec_path)

    def test_spectrum_load_csv_invalid_wave_val(self):
        """
        Verify invalid value in spectral model file fails as expected.
        """
        spt_err = self.spt_invalid_flux_val

        with self.assertRaises(ValueError):
            _spectrum_load_csv(spt_err, self.spec_path_err)

    def test_spectrum_load_csv_invalid_flux_val(self):
        """
        Verify invalid value in spectral model file fails as expected.
        """
        spt_err = self.spt_invalid_wave_val

        with self.assertRaises(ValueError):
            _spectrum_load_csv(spt_err, self.spec_path_err)

    def test_spectrum_load_csv_invalid_format(self):
        """
        Verify invalid value in spectral model file fails as expected.
        """
        spt_err = self.spt_invalid_format

        with self.assertRaises(ValueError):
            _spectrum_load_csv(spt_err, self.spec_path_err)

#_spectrum_scale
    def test_spectrum_scale(self):
        """
        Verify that spectrum is being scaled correctly to specified magnitude
        """
        spec_model = _spectrum_load_csv(self.spt, self.spec_path) #Ergs/sec/cm^2/Ang
        wvl = spec_model['wvl']
        flux = spec_model['flux']

        # check that recovered flux rate is within a percent of expectations
        tol = 0.01
        flux_answer = [1, 1, 1, 1, 1, 1, 1]

        # scale spectrum to specified band magnitude
        flux_scaled = _spectrum_scale(wvl, flux, self.test_mag, self.test_filter_band) #Ergs/sec/cm^2/Ang
        self.assertTrue(((np.abs(flux_scaled - flux_answer) / flux_scaled) < tol).all())

#_spectrum_mag_estimate
    def test_spectrum_mag_estimate(self):
        """
        Verify that spectrum is being scaled correctly to specified magnitude
        """
        spec_model = _spectrum_load_csv(self.spt, self.spec_path) #Ergs/sec/cm^2/Ang
        wvl = spec_model['wvl']
        flux = spec_model['flux']

        # check that recovered flux rate is within a percent of expectations
        tol = 0.01

        # scale spectrum to specified band magnitude
        flux_scaled = _spectrum_scale(wvl, flux, self.test_mag, self.test_filter_band) #Ergs/sec/cm^2/Ang
        mag_out = _spectrum_mag_estimate(wvl,flux_scaled, self.test_filter_band)

        self.assertTrue((np.abs(mag_out - self.test_mag) / self.test_mag) < tol)

#_cfam_band_integrate
    def test_cfam_band_integrate_flux(self):
        """
        For dummy filter curve and spectrum, check that flux integration is working properly
        """
        spec_model = _spectrum_load_csv(self.spt, self.spec_path) #Ergs/sec/cm^2/Ang
        wvl = spec_model['wvl']
        flux = spec_model['flux']

        # check that recovered flux rate is within a percent of expectations
        tol = 0.01
        the_answer = 1.0

        # scale spectrum to specified band magnitude
        flux_scaled = _spectrum_scale(wvl, flux, self.test_mag, self.test_filter_band)

        # estimate recorded flux in specified band entering telescope
        flux_at_cgi, _ = _cfam_band_integrate(wvl, flux_scaled * self.collecting_area,
                                                     self.test_cfam,
                                                     self.ut_cfam_folder)
        self.assertTrue((np.abs(flux_at_cgi - the_answer) / the_answer) < tol)

    def test_cfam_band_integrate_wave(self):
        """
        For dummy filter curve and spectrum, check that wavelength centroid is
        being computed correctly
        """
        spec_model = _spectrum_load_csv(self.spt, self.spec_path) #Ergs/sec/cm^2/Ang
        wvl = spec_model['wvl']
        flux = spec_model['flux']

        # check that recovered wavelength center is correct (it's a flat curve, so it better be)
        tol = 0.01
        the_answer = 6500.

        # scale spectrum to specified band magnitude
        flux_scaled = _spectrum_scale(wvl, flux, self.test_mag, self.test_filter_band)

        # estimate recorded flux in specified band entering telescope
        _, wave_centroid = _cfam_band_integrate(wvl, flux_scaled * self.collecting_area,
                                                     self.test_cfam,
                                                     self.ut_cfam_folder)
        self.assertTrue((np.abs(wave_centroid - the_answer) / the_answer) < tol)

    def test_spec_generate_cgi_cfam_flux(self):
        """
        For dummy filter curve and spectrum, check that wrapper function is
        producing the right answers in flux and wavelength centroid
        """
        the_answer = 1.0 # Photons/second, predetermined based on the input flux values
        tol = 0.01
        flux_at_cgi, _ = spec_generate_cgi_cfam(self.test_wvl, self.test_mag,
                                                self.test_filter_band, self.spt,
                                                self.test_cfam, self.spec_path,
                                                self.ut_cfam_folder, self.collecting_area)
        self.assertTrue((np.abs(flux_at_cgi - the_answer) / the_answer) < tol)

    def test_spec_generate_cgi_cfam_wave(self):
        the_answer = 6500. # Ang, predetermined based on the input flux values
        tol = 0.01
        _, wav_center = spec_generate_cgi_cfam(self.test_wvl, self.test_mag,
                                                self.test_filter_band, self.spt,
                                                self.test_cfam, self.spec_path,
                                                self.ut_cfam_folder, self.collecting_area)
        self.assertTrue((np.abs(wav_center - the_answer) / the_answer) < tol)


#_get_cfam_transmission
    def test_cfam_load_invalid_val_trans(self):
        """
        For dummy filter curve, check that bad transmission val fails as expected
        """
        filter_dir = self.ut_cfam_folder_err
        filter_invalid = self.test_cfam_invalid_trans_val

        with self.assertRaises(ValueError):
            _get_cfam_transmission(filter_dir, filter_invalid, self.test_wvl)

    def test_cfam_load_invalid_val_wave(self):
        """
        For dummy filter curve, check that bad transmission val fails as expected
        """
        filter_dir = self.ut_cfam_folder_err
        filter_invalid = self.test_cfam_invalid_wave_val

        with self.assertRaises(ValueError):
            _get_cfam_transmission(filter_dir, filter_invalid, self.test_wvl)

    def test_cfam_load_invalid_format(self):
        """
        For dummy filter curve, check that bad transmission val fails as expected
        """
        filter_dir = self.ut_cfam_folder_err
        filter_invalid = self.test_cfam_invalid_format

        with self.assertRaises(ValueError):
            _get_cfam_transmission(filter_dir, filter_invalid, self.test_wvl)

if __name__ == '__main__':
    unittest.main()
