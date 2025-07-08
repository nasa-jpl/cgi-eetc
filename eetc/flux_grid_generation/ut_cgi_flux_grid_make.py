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
import tempfile
import numpy as np
import os
import eetc
import eetc.util.ut_check as ut_check
from eetc.flux_grid_generation.cgi_flux_grid_make import cgi_flux_grid_generate
import yaml
from astropy.io import fits

LOCAL_PATH = os.path.abspath(os.path.dirname( __file__ ))
UT_CONFIG_PATH = str(Path(LOCAL_PATH, 'ut_config'))

VALID_CONFIG_KEYS = ['magref', 'magbands',
                    'input_files','output_flux_grid_file',
                    'output_wave_grid_file','ota_collecting_area']

# check that flux grid is being produced correctly
class TestCgiFluxGridMake(unittest.TestCase):
    """
    Unit tests for cgi_flux_grid_make function.
    """
    def setUp(self):
        self.input_file = os.path.join(UT_CONFIG_PATH,'ut_flux_grid_generate_input_params.yaml')
        self.input_file_invalid_key = os.path.join(UT_CONFIG_PATH,'ut_flux_invalid_key.yaml')
        self.input_file_invalid_val = os.path.join(UT_CONFIG_PATH,'ut_flux_invalid_val.yaml')
        self.input_file_missing_key = os.path.join(UT_CONFIG_PATH,'ut_flux_missing_key.yaml')
        self.input_file_extra_key = os.path.join(UT_CONFIG_PATH,'ut_flux_extra_key.yaml')

        with open(self.input_file, 'r') as stream:
            ut_input_dicts = (yaml.safe_load(stream))

        # load in input parameters from config file for comparison later
        self.refmag = ut_input_dicts['magref']
        self.refmagband = ut_input_dicts['magbands']
        # reading off filenames from ut_bpgs_atlas_csv
        self.refspt = 'UT_SPT'
        # reading off filenames from ut_cfam_filter_curves
        self.cfams = 'UT'

        self.flux_fd, self.outputfluxgridfits = tempfile.mkstemp()
        self.wave_fd, self.outputwavegridfits = tempfile.mkstemp()

        self.headerkeys = ['REFBAND', 'REFMAG', 'SPECTYPE','CFAMCOLS', 'PHOTEXTS',
                            'NAXIS1', 'SIMPLE', 'BITPIX', 'NAXIS2', 'NAXIS']

    def tearDown(self):
        os.close(self.flux_fd)
        os.unlink(self.outputfluxgridfits)
        os.close(self.wave_fd)
        os.unlink(self.outputwavegridfits)
        pass


    def test_fits_generated_flux_header(self):
        """
        Verify that flux grid file has the right header keywords
        """
        cgi_flux_grid_generate(input_file=self.input_file,
                               output_flux_grid_file=self.outputfluxgridfits,
                               output_wave_grid_file=self.outputwavegridfits,
        )
        fits_file_saved = self.outputfluxgridfits
        header = dict(fits.getheader(fits_file_saved))
        self.assertTrue(set(header.keys()) == set(self.headerkeys))

    def test_fits_generated_flux_header_vals_mag(self):
        """
        Verify that flux grid file has the right header keyword values
        """
        cgi_flux_grid_generate(input_file=self.input_file,
                               output_flux_grid_file=self.outputfluxgridfits,
                               output_wave_grid_file=self.outputwavegridfits,
        )
        fits_file_saved = self.outputfluxgridfits
        header = dict(fits.getheader(fits_file_saved))
        self.assertTrue(header['REFMAG'] == self.refmag)

    def test_fits_generated_flux_header_vals_spt(self):
        """
        Verify that flux grid file has the right header keyword values
        """
        cgi_flux_grid_generate(input_file=self.input_file,
                               output_flux_grid_file=self.outputfluxgridfits,
                               output_wave_grid_file=self.outputwavegridfits,
        )
        fits_file_saved = self.outputfluxgridfits
        header = dict(fits.getheader(fits_file_saved))
        self.assertTrue(header['SPECTYPE'] == self.refspt)

    def test_fits_generated_flux_header_vals_cfams(self):
        """
        Verify that flux grid file has the right header keyword values
        """
        cgi_flux_grid_generate(input_file=self.input_file,
                               output_flux_grid_file=self.outputfluxgridfits,
                               output_wave_grid_file=self.outputwavegridfits,
        )
        fits_file_saved = self.outputfluxgridfits
        header = dict(fits.getheader(fits_file_saved))
        self.assertTrue(header['CFAMCOLS'] == self.cfams)

    def test_fits_generated_flux_header_vals(self):
        """
        Verify that flux grid file has the right header keyword values
        """
        cgi_flux_grid_generate(input_file=self.input_file,
                               output_flux_grid_file=self.outputfluxgridfits,
                               output_wave_grid_file=self.outputwavegridfits,
        )
        fits_file_saved = self.outputfluxgridfits
        header = dict(fits.getheader(fits_file_saved))
        self.assertTrue([header['REFBAND']] == self.refmagband)

    def test_fits_generated_flux_value(self):
        """
        Verify that flux grid file has the right correct value
        """
        cgi_flux_grid_generate(input_file=self.input_file,
                               output_flux_grid_file=self.outputfluxgridfits,
                               output_wave_grid_file=self.outputwavegridfits,
        )
        tol = 0.01 # 1% tolerance
        flux_estimate = fits.getdata(self.outputfluxgridfits)[0]
        the_answer = 0.00477977
        self.assertTrue(np.abs(flux_estimate - the_answer) / the_answer < tol)

    def test_fits_generated_wave_header(self):
        """
        Verify that wave grid file has the right header keywords
        """
        cgi_flux_grid_generate(input_file=self.input_file,
                               output_flux_grid_file=self.outputfluxgridfits,
                               output_wave_grid_file=self.outputwavegridfits,
        )
        fits_file_saved = self.outputwavegridfits
        header = dict(fits.getheader(fits_file_saved))
        self.assertTrue(set(header.keys()) == set(self.headerkeys))

    def test_fits_generated_wave_header_vals_mag(self):
        """
        Verify that wave grid file has the right header keyword values
        """
        cgi_flux_grid_generate(input_file=self.input_file,
                               output_flux_grid_file=self.outputfluxgridfits,
                               output_wave_grid_file=self.outputwavegridfits,
        )
        fits_file_saved = self.outputwavegridfits
        header = dict(fits.getheader(fits_file_saved))
        self.assertTrue(header['REFMAG'] == self.refmag)

    def test_fits_generated_wave_header_vals_spt(self):
        """
        Verify that wave grid file has the right header keyword values
        """
        cgi_flux_grid_generate(input_file=self.input_file,
                               output_flux_grid_file=self.outputfluxgridfits,
                               output_wave_grid_file=self.outputwavegridfits,
        )
        fits_file_saved = self.outputwavegridfits
        header = dict(fits.getheader(fits_file_saved))
        self.assertTrue(header['SPECTYPE'] == self.refspt)

    def test_fits_generated_flux_header_vals_cfams(self):
        """
        Verify that wave grid file has the right header keyword values
        """
        cgi_flux_grid_generate(input_file=self.input_file,
                               output_flux_grid_file=self.outputfluxgridfits,
                               output_wave_grid_file=self.outputwavegridfits,
        )
        fits_file_saved = self.outputwavegridfits
        header = dict(fits.getheader(fits_file_saved))
        self.assertTrue(header['CFAMCOLS'] == self.cfams)

    def test_fits_generated_wave_header_vals(self):
        """
        Verify that wave grid file has the right header keyword values
        """
        cgi_flux_grid_generate(input_file=self.input_file,
                               output_flux_grid_file=self.outputfluxgridfits,
                               output_wave_grid_file=self.outputwavegridfits,
        )
        fits_file_saved = self.outputwavegridfits
        header = dict(fits.getheader(fits_file_saved))
        self.assertTrue([header['REFBAND']] == self.refmagband)

    def test_fits_generated_wave_value(self):
        """
        Verify that wave grid file has the right correct value
        """
        cgi_flux_grid_generate(input_file=self.input_file,
                               output_flux_grid_file=self.outputfluxgridfits,
                               output_wave_grid_file=self.outputwavegridfits,
        )
        tol = 0.01 # 1% tolerance
        wave_estimate = fits.getdata(self.outputwavegridfits)[0]
        the_answer = 6500.
        self.assertTrue(np.abs(wave_estimate - the_answer) / the_answer < tol)

    # fails as expected if a .csv file or .txt is missing or malformed with respect to your specification.
    def test_invalid_file_location(self):
        """invalid input fail as expected -- file missing"""
        pointer_path_err = self.input_file + '_invalid'

        with self.assertRaises(IOError):
            cgi_flux_grid_generate(input_file=pointer_path_err,
                            output_flux_grid_file=self.outputfluxgridfits,
                            output_wave_grid_file=self.outputwavegridfits,
            )


    def test_invalid_pointer_keys(self):
        """Verify invalid pointer file keys fail as expected."""
        pointer_path_invalid = self.input_file_invalid_key

        with self.assertRaises(KeyError):
            cgi_flux_grid_generate(input_file=pointer_path_invalid,
                            output_flux_grid_file=self.outputfluxgridfits,
                            output_wave_grid_file=self.outputwavegridfits,
            )


    def test_missing_pointer_keys(self):
        """Verify missing pointer file key fails as expected."""
        pointer_path_invalid = self.input_file_missing_key

        with self.assertRaises(KeyError):
            cgi_flux_grid_generate(input_file=pointer_path_invalid,
                            output_flux_grid_file=self.outputfluxgridfits,
                            output_wave_grid_file=self.outputwavegridfits,
            )


    def test_extra_pointer_keys(self):
        """Verify extra pointer file key fails as expected."""
        pointer_path_invalid = self.input_file_extra_key

        with self.assertRaises(KeyError):
            cgi_flux_grid_generate(input_file=pointer_path_invalid,
                            output_flux_grid_file=self.outputfluxgridfits,
                            output_wave_grid_file=self.outputwavegridfits,
            )


    def test_invalid_pointer_vals(self):
        """Verify invalid pointer file values fail as expected."""
        pointer_path_invalid = self.input_file_invalid_val

        with self.assertRaises(TypeError):
            cgi_flux_grid_generate(input_file=pointer_path_invalid,
                            output_flux_grid_file=self.outputfluxgridfits,
                            output_wave_grid_file=self.outputwavegridfits,
            )


    def test_invalid_flux_grid_file(self):
        """Verify invalid flux grid file values fail as expected."""
        for err in [1j, (1.,), [5, 5], -1, 0, 1.0]: #strlist w/o None
            with self.assertRaises(TypeError):
                cgi_flux_grid_generate(input_file=self.input_file,
                                output_flux_grid_file=err,
                                output_wave_grid_file=self.outputwavegridfits,
            )


    def test_invalid_flux_grid_file(self):
        """Verify invalid wave grid file values fail as expected."""
        for err in [1j, (1.,), [5, 5], -1, 0, 1.0]: #strlist w/o None
            with self.assertRaises(TypeError):
                cgi_flux_grid_generate(input_file=self.input_file,
                                output_flux_grid_file=self.outputfluxgridfits,
                                output_wave_grid_file=err,
            )

    def test_load_abs_rel(self):
        """Verify we can load from both absolute and relative paths"""

        substitute = copy.deepcopy(self.input_file)
        with open(self.input_file) as stream:
            file_dict = yaml.safe_load(stream)
        substitute = copy.deepcopy(file_dict)
        basepath = os.path.dirname(os.path.dirname(os.path.abspath(
                                                            self.input_file)))
        # replace all the usual relative path inputs with absolute
        substitute['input_files']['cfam_filter_curve_path'] = \
            os.path.join(basepath,
                         substitute['input_files']['cfam_filter_curve_path'])
        substitute['input_files']['stellar_model_spec_path'] = \
            os.path.join(basepath,
                         substitute['input_files']['stellar_model_spec_path'])
        substitute['output_flux_grid_file'] =  os.path.join(basepath,
                         substitute['output_flux_grid_file'])
        substitute['output_wave_grid_file'] =  os.path.join(basepath,
                         substitute['output_wave_grid_file'])

        try:
            (fd, name) = tempfile.mkstemp()
            with open(name, 'w') as FILE:
                yaml.dump(substitute, FILE)
                pass
            # success if this completes without errors
            cgi_flux_grid_generate(input_file=name,
                                output_flux_grid_file=self.outputfluxgridfits,
                                output_wave_grid_file=self.outputwavegridfits,
            )
        finally:
            os.close(fd)
            os.unlink(name)
            pass
        pass


if __name__ == '__main__':
    unittest.main()
