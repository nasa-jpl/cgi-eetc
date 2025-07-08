# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit test suite for load module.
"""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList

import eetc
import eetc.util.ut_check as ut_check
from eetc.load import (load_sequences, load_thpt_configs, load_excam_config,
                       load_flux_grid, load_wave_grid,
                       load_locam_config)
from eetc.load import (_load_thptcurve, _unpack_excam_config,
                       _unpack_locam_config)


LOCAL_PATH = eetc.lib_dir
CONFIG_PATH = str(Path(LOCAL_PATH, 'config'))
UT_CONFIG_PATH = str(Path(LOCAL_PATH, 'ut_config'))

# Lists for checking valid inputs
strlist = ut_check.strlist
rslist = ut_check.rslist
rnslist = ut_check.rnslist
rpslist = ut_check.rpslist
nsilist = ut_check.nsilist
psilist = ut_check.psilist
oneDlist = ut_check.oneDlist
twoDlist = ut_check.twoDlist


def _make_hdul():
    """
    Generate Header Data Unit List (HDUList) from existing file.
    """
    flux_grid_path = str(Path(LOCAL_PATH, 'flux_grid_generation', 'grid_files',
                              'flux_grid.fits'))

    with fits.open(flux_grid_path) as hdul_raw:
        hdul = copy.deepcopy(hdul_raw)

    # Make sure values are unique for testing purposes
    for i, hdu in enumerate(hdul):
        # flux_grid_vals
        hdul[i].data = np.ones_like(hdu.data) * i
        hdul[i].data[0, 1] = 10.  # For orientation check
        # spts
        spts = hdul[i].header['SPECTYPE'].split(',')
        if i < len(spts):
            spts[i], spts[-1] = spts[-1], spts[i]  # Swap with last element
        spts_str = ''.join(f'{spt},' for spt in spts).strip(',')
        hdul[i].header['SPECTYPE'] = spts_str

        # cfams
        cfams = hdul[i].header['CFAMCOLS'].split(',')
        if i < len(cfams):
            cfams[i], cfams[-1] = cfams[-1], cfams[i]  # Swap with last element
        cfams_str = ''.join(f'{cfam},' for cfam in cfams).strip(',')
        hdul[i].header['CFAMCOLS'] = cfams_str
        # ref_mag
        hdul[i].header['REFMAG'] += i

    return hdul


def _hdul_to_dict(hdul):
    """
    Take HDUList and pack it into a dictionary, with keys corresponding to phot
    filters and values corresponding to HDU objects.

    The header of the flux grid PrimaryHDU contains a string of phot filters
    separated by commas. Each of these phot filters corresponds to a HDU, in
    order. This function will assign the phot filters in the correct order to
    the HDUs.
    """
    phots = hdul[0].header['PHOTEXTS'].split(',')

    hdul_dict = dict(zip(phots, hdul))

    return hdul_dict


def _dict_to_hdul(hdul_dict):
    """
    Convert HDUList dictionary back into HDUList. Make sure list order matches
    order of phot filters.
    """
    phots = list(hdul_dict.values())[0].header['PHOTEXTS'].split(',')

    hdul = []
    for phot in phots:
        hdul.append(hdul_dict[phot])

    return HDUList(hdul)


class TestLoadSequences(unittest.TestCase):
    """
    Unit tests for load_sequences function.
    """
    def setUp(self):
        self.sequences_path = str(Path(UT_CONFIG_PATH, 'ut_sequences.yaml'))
        self.sequences_path_num = str(Path(UT_CONFIG_PATH,
                                           'ut_sequences_bad_num_pix.yaml'))
        self.sequences_path_frac = str(Path(UT_CONFIG_PATH,
                                           'ut_sequences_bad_frac.yaml'))
        self.sequences_path_frac1 = str(Path(UT_CONFIG_PATH,
                                           'ut_sequences_bad_frac1.yaml'))
        self.sequences_path_peak = str(Path(UT_CONFIG_PATH,
                                           'ut_sequences_bad_peak.yaml'))
        self.flux_grid_path = str(Path(LOCAL_PATH, 'flux_grid_generation',
                                       'grid_files', 'flux_grid.fits'))
        self.wave_grid_path = str(Path(LOCAL_PATH, 'flux_grid_generation',
                                       'grid_files', 'wave_grid.fits'))

        # Set values to match the contents of ut_sequences
        self.spam_lsam = 'open_spam_open_lsam'
        self.fpam = 'open'
        self.fsam = 'open'
        wave_grid = load_wave_grid(self.wave_grid_path)
        _, _, cfam_list, _ = wave_grid[list(wave_grid.keys())[0]]
        self.validating_cfam = set(cfam_list)
        self.dpam = 'pupil_lens'

        self.mode = 'excam_imaging'
        self.peak_flux_ratio_pix = 1.

        self.valid_modes = set(['excam_imaging', 'locam'])
        self.valid_spam_lsam = set([self.spam_lsam])
        self.valid_fpam = set(['open', 'nfov_dm_reflect'])
        self.valid_fsam = set([self.fsam])
        self.valid_dpam = set(['pupil_lens', 'defocus_1', 'imaging_lens',
                               'open'])


    def test_success(self):
        """Given valid inputs, completes without error"""
        load_sequences(
            self.sequences_path,
            self.valid_modes,
            self.valid_spam_lsam,
            self.valid_fpam,
            self.valid_fsam,
            self.validating_cfam,
            self.valid_dpam,
        )
        pass


    def test_invalid_string(self):
        """
        Verify invalid inputs fail as expected.
        """
        # sequences_path
        for perr in strlist:
            with self.assertRaises(TypeError):
                load_sequences(
                    perr,
                    self.valid_modes,
                    self.valid_spam_lsam,
                    self.valid_fpam,
                    self.valid_fsam,
                    self.validating_cfam,
                    self.valid_dpam,
                )


    def test_invalid_valid_modes(self):
        """
        Verify invalid inputs fail as expected.
        """

        # set
        perrlist = [[self.mode], {self.mode:0}, 0, None]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                load_sequences(
                    self.sequences_path,
                    perr,
                    self.valid_spam_lsam,
                    self.valid_fpam,
                    self.valid_fsam,
                    self.validating_cfam,
                    self.valid_dpam,
                )

        # contents
        perrlist2 = [0, 1j, None, [5, 5], [5, 'txt']]

        for perr in perrlist2:
            with self.assertRaises(TypeError):
                load_sequences(
                    self.sequences_path,
                    set(perr),
                    self.valid_spam_lsam,
                    self.valid_fpam,
                    self.valid_fsam,
                    self.validating_cfam,
                    self.valid_dpam,
                )

    def test_invalid_valid_spam_lsam(self):
        """
        Verify invalid inputs fail as expected.
        """

        # set
        perrlist = [[self.spam_lsam], {self.spam_lsam:0}, 0, None]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                load_sequences(
                    self.sequences_path,
                    self.valid_modes,
                    perr,
                    self.valid_fpam,
                    self.valid_fsam,
                    self.validating_cfam,
                    self.valid_dpam,
                )

        # contents
        perrlist2 = [0, 1j, None, [5, 5], [5, 'txt']]

        for perr in perrlist2:
            with self.assertRaises(TypeError):
                load_sequences(
                    self.sequences_path,
                    self.valid_modes,
                    set(perr),
                    self.valid_fpam,
                    self.valid_fsam,
                    self.validating_cfam,
                    self.valid_dpam,
                )


    def test_invalid_valid_fpam(self):
        """
        Verify invalid inputs fail as expected.
        """

        # set
        perrlist = [[self.fpam], {self.fpam:0}, 0, None]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                load_sequences(
                    self.sequences_path,
                    self.valid_modes,
                    self.valid_spam_lsam,
                    perr,
                    self.valid_fsam,
                    self.validating_cfam,
                    self.valid_dpam,
                )

        # contents
        perrlist2 = [0, 1j, None, [5, 5], [5, 'txt']]

        for perr in perrlist2:
            with self.assertRaises(TypeError):
                load_sequences(
                    self.sequences_path,
                    self.valid_modes,
                    self.valid_spam_lsam,
                    set(perr),
                    self.valid_fsam,
                    self.validating_cfam,
                    self.valid_dpam,
                )


    def test_invalid_valid_fsam(self):
        """
        Verify invalid inputs fail as expected.
        """

        # set
        perrlist = [[self.fsam], {self.fsam:0}, 0, None]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                load_sequences(
                    self.sequences_path,
                    self.valid_modes,
                    self.valid_spam_lsam,
                    self.valid_fpam,
                    perr,
                    self.validating_cfam,
                    self.valid_dpam,
                )

        # contents
        perrlist2 = [0, 1j, None, [5, 5], [5, 'txt']]

        for perr in perrlist2:
            with self.assertRaises(TypeError):
                load_sequences(
                    self.sequences_path,
                    self.valid_modes,
                    self.valid_spam_lsam,
                    self.valid_fpam,
                    set(perr),
                    self.validating_cfam,
                    self.valid_dpam,
                )

    def test_invalid_validating_cfam(self):
        """
        Verify invalid inputs fail as expected.
        """

        # set
        perrlist = [[self.dpam], {self.dpam:0}, 0, None]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                load_sequences(
                    self.sequences_path,
                    self.valid_modes,
                    self.valid_spam_lsam,
                    self.valid_fpam,
                    self.valid_fsam,
                    perr,
                    self.valid_dpam,
                )

        # contents
        perrlist2 = [0, 1j, None, [5, 5], [5, 'txt']]

        for perr in perrlist2:
            with self.assertRaises(TypeError):
                load_sequences(
                    self.sequences_path,
                    self.valid_modes,
                    self.valid_spam_lsam,
                    self.valid_fpam,
                    self.valid_fsam,
                    set(perr),
                    self.valid_dpam,
                )

    def test_invalid_valid_dpam(self):
        """
        Verify invalid inputs fail as expected.
        """

        # set
        perrlist = [[self.dpam], {self.dpam:0}, 0, None]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                load_sequences(
                    self.sequences_path,
                    self.valid_modes,
                    self.valid_spam_lsam,
                    self.valid_fpam,
                    self.valid_fsam,
                    self.validating_cfam,
                    perr,
                )

        # contents
        perrlist2 = [0, 1j, None, [5, 5], [5, 'txt']]

        for perr in perrlist2:
            with self.assertRaises(TypeError):
                load_sequences(
                    self.sequences_path,
                    self.valid_modes,
                    self.valid_spam_lsam,
                    self.valid_fpam,
                    self.valid_fsam,
                    self.validating_cfam,
                    set(perr),
                )




    def test_invalid_file_location(self):
        """
        Verify invalid file location fails as expected.
        """
        sequences_path_err = self.sequences_path + '_invalid'

        with self.assertRaises(IOError):
            load_sequences(
                sequences_path_err,
                self.valid_modes,
                self.valid_spam_lsam,
                self.valid_fpam,
                self.valid_fsam,
                self.validating_cfam,
                self.valid_dpam,
            )


    def test_invalid_settings_keys(self):
        """
        Verify valid_settings_keys validation.
        """
        sequences_path_invalid = \
            str(Path(LOCAL_PATH, 'ut_config',
                     'ut_sequences_invalid_settings_keys.yaml'))

        with self.assertRaises(KeyError):
            load_sequences(
                sequences_path_invalid,
                self.valid_modes,
                self.valid_spam_lsam,
                self.valid_fpam,
                self.valid_fsam,
                self.validating_cfam,
                self.valid_dpam,
            )
        pass

    def test_invalid_peak_flux_ratio_pix(self):
        """
        Verify valid peak_flux_ratio_pix.
        """

        with self.assertRaises(TypeError):
            load_sequences(
                self.sequences_path_peak,
                self.valid_modes,
                self.valid_spam_lsam,
                self.valid_fpam,
                self.valid_fsam,
                self.validating_cfam,
                self.valid_dpam,
            )
        pass

    def test_invalid_fraction(self):
        """
        Verify valid fraction.
        """

        with self.assertRaises(TypeError):
            load_sequences(
                self.sequences_path_frac,
                self.valid_modes,
                self.valid_spam_lsam,
                self.valid_fpam,
                self.valid_fsam,
                self.validating_cfam,
                self.valid_dpam,
            )
        pass

    def test_invalid_fraction1(self):
        """
        Verify fraction <= 1.
        """

        with self.assertRaises(ValueError):
            load_sequences(
                self.sequences_path_frac1,
                self.valid_modes,
                self.valid_spam_lsam,
                self.valid_fpam,
                self.valid_fsam,
                self.validating_cfam,
                self.valid_dpam,
            )
        pass

    def test_invalid_num_pixels(self):
        """
        Verify valid num_pixels.
        """

        with self.assertRaises(TypeError):
            load_sequences(
                self.sequences_path_num,
                self.valid_modes,
                self.valid_spam_lsam,
                self.valid_fpam,
                self.valid_fsam,
                self.validating_cfam,
                self.valid_dpam,
            )
        pass


class TestLoadThptConfigs(unittest.TestCase):
    """
    Unit tests for load_thpt_configs function.
    """

    def setUp(self):
        self.thpt_configs_path = str(Path(UT_CONFIG_PATH,
                                          'ut_thpt_configs.yaml'))
        self.thpt_coatings_path = str(Path(UT_CONFIG_PATH,
                                           'ut_thpt_coatings.yaml'))
        self.thpt_data_path = str(Path(UT_CONFIG_PATH,
                                       'ut_thpt_data.yaml'))
        self.thptcurves_dir_path = str(Path(LOCAL_PATH, 'ut_thptcurves'))
        self.thptcurve_path = str(Path(UT_CONFIG_PATH,
                                       'ut_thptcurve.txt')) # tests failures

        # Set up reference coating_config
        self.num_alum = 0
        self.num_IOI = 1
        self.num_lens_glass = 2
        self.num_broadband_ar = 3
        self.num_spectroscopy_unit = 4
        self.num_hrc = 5

        self.coating_config = {
            'alum': self.num_alum,
            'IOI': self.num_IOI,
            'lens_glass': self.num_lens_glass,
            'broadband_ar': self.num_broadband_ar,
            'spectroscopy_unit': self.num_spectroscopy_unit,
            'hrc': self.num_hrc
        }

        # Set up reference coating_thptcurves
        self.thptcurve_al_path = 'alum.txt'
        self.thptcurve_ioi_path = 'IOI-6-8301R8deg.txt'
        self.thptcurve_lens_glass_path = 'lens_transmission.txt'
        self.thptcurve_ar_glass_path = 'ar_glass.dat'
        self.thptcurve_spec_unit_path = 'spectroscopy_unit.txt'
        self.thptcurve_hrc_path = 'hrc.txt'
        self.thptcurve_emccd_path = 'e2v_excam.txt'

        self.coating_thptcurves = {
            'thpt_al': self.thptcurve_al_path,
            'thpt_fss99': self.thptcurve_ioi_path,
            'thpt_glass': self.thptcurve_lens_glass_path,
            'thpt_ar_glass': self.thptcurve_ar_glass_path,
            'thpt_spec_unit': self.thptcurve_spec_unit_path,
            'thpt_hrc': self.thptcurve_hrc_path,
            'thpt_emccd': self.thptcurve_emccd_path
        }

        # Set up reference setting_thptcurves
        # Set up reference settings with unique setting values for each element
        self.dms_flat_path = 'flat1.txt'
        self.dms_hlc_flattened_with_pattern_path = 'flat2.txt'
        self.dms_path = {
            'flat': self.dms_flat_path,
            'hlc_flattened_with_pattern': \
            self.dms_hlc_flattened_with_pattern_path
        }
        self.spam_open_path = 'open.txt'
        self.spam_spec_path = 'spec.txt'
        self.spam_wfov_path = 'wfov.txt'
        self.spam_path = {
            'open': self.spam_open_path,
            'spec': self.spam_spec_path,
            'wfov': self.spam_wfov_path
        }
        self.fpam_open_path = 'open.txt'
        self.fpam_aligned_path = 'aligned.txt'
        self.fpam_path = {
            'open': self.fpam_open_path,
            'aligned': self.fpam_aligned_path
        }
        self.lsam_open_path = 'open.txt'
        self.lsam_nfov_path = 'nfov.txt'
        self.lsam_spec_path = 'spec.txt'
        self.lsam_wfov_path = 'wfov.txt'
        self.lsam_aligned_path = 'aligned.txt'
        self.lsam_path = {
            'open': self.lsam_open_path,
            'nfov': self.lsam_nfov_path,
            'spec': self.lsam_spec_path,
            'wfov': self.lsam_wfov_path,
            'aligned': self.lsam_aligned_path
        }
        self.fsam_open_path = 'open.txt'
        self.fsam_aligned_path = 'aligned.txt'
        self.fsam_path = {
            'open': self.fsam_open_path,
            'aligned': self.fsam_aligned_path
        }
        self.dpam_imaging_lens_path = 'imaging_lens.txt'
        self.dpam_pupil_lens_path = 'pupil_lens.txt'
        self.dpam_path = {
            'imaging_lens': self.dpam_imaging_lens_path,
            'pupil_lens': self.dpam_pupil_lens_path
        }
        self.setting_thptcurves = {
            'spam': self.spam_path,
            'fpam': self.fpam_path,
            'lsam': self.lsam_path,
            'fsam': self.fsam_path,
            'dpam': self.dpam_path
        }

        # Set up reference thpt_configs
        self.thpt_configs = {
            'hlc': self.coating_config,
            'lowfs': self.coating_config,
            'spec': self.coating_config,
        }
        self.thpt_data = dict()
        self.thpt_data.update(self.coating_thptcurves)
        self.thpt_data.update(self.setting_thptcurves)

    def test_invalid_string(self):
        """
        Verify invalid inputs fail as expected.
        """
        # thpt_configs_path
        for perr in strlist:
            with self.assertRaises(TypeError):
                load_thpt_configs(
                    perr,
                    self.thpt_coatings_path,
                    self.thpt_data_path,
                    self.thptcurves_dir_path,
                )

        # thpt_coatings_path
        for perr in strlist:
            with self.assertRaises(TypeError):
                load_thpt_configs(
                    self.thpt_configs_path,
                    perr,
                    self.thpt_data_path,
                    self.thptcurves_dir_path,
                )

        # thpt_data_path
        for perr in strlist:
            with self.assertRaises(TypeError):
                load_thpt_configs(
                    self.thpt_data_path,
                    self.thpt_coatings_path,
                    perr,
                    self.thptcurves_dir_path,
                )

        # thptcurves_dir_path
        for perr in strlist:
            with self.assertRaises(TypeError):
                load_thpt_configs(
                    self.thpt_configs_path,
                    self.thpt_coatings_path,
                    self.thpt_data_path,
                    perr,
                )

    def test_invalid_file_location(self):
        """
        Verify invalid file location fails as expected.
        """
        thpt_configs_path_err = self.thpt_configs_path + '_invalid'
        thpt_coatings_path_err = self.thpt_coatings_path + '_invalid'
        thpt_data_path_err = self.thpt_configs_path + '_invalid'
        thptcurves_dir_path_err = self.thptcurves_dir_path + '_invalid'

        # thpt_configs_path
        with self.assertRaises(IOError):
            load_thpt_configs(
                thpt_configs_path_err,
                self.thpt_coatings_path,
                self.thpt_data_path,
                self.thptcurves_dir_path,
            )

        # thpt_coatings_path
        with self.assertRaises(IOError):
            load_thpt_configs(
                self.thpt_configs_path,
                thpt_coatings_path_err,
                self.thpt_data_path,
                self.thptcurves_dir_path,
            )

        # thpt_data_path
        with self.assertRaises(IOError):
            load_thpt_configs(
                self.thpt_configs_path,
                self.thpt_coatings_path,
                thpt_data_path_err,
                self.thptcurves_dir_path,
            )

        # thptcurves_dir_path
        with self.assertRaises(IOError):
            load_thpt_configs(
                self.thpt_configs_path,
                self.thpt_coatings_path,
                self.thpt_data_path,
                thptcurves_dir_path_err,
            )


#-----------------------------------------------
# thpt_configs checks

    def test_invalid_coating_type(self):
        """
        Verify valid_coating_types validation.
        """
        thpt_configs_path_invalid = \
            str(Path(UT_CONFIG_PATH,
                     'ut_thpt_configs_invalid_coating.yaml'))

        with self.assertRaises(TypeError):
            load_thpt_configs(
                thpt_configs_path_invalid,
                self.thpt_coatings_path,
                self.thpt_data_path,
                self.thptcurves_dir_path,
            )

    def test_summing_coatings(self):
        """
        Verify summing of coating types is done correctly in load_thpt_configs.
        """
        test = load_thpt_configs(
            self.thpt_configs_path,
            self.thpt_coatings_path,
            self.thpt_data_path,
            self.thptcurves_dir_path,
        )
        # can be verified by inspection of thpt_configs_path
        # example with counts found in the mode
        self.assertEqual(test[0]['ut_only']['IOI'], 1)
        self.assertEqual(test[0]['ut_only']['hrc'], 2)
        # no qe_excam in the test, so verify we get a 0
        self.assertEqual(test[0]['ut_only']['qe_excam'], 0)

#----------------------------------------------------
# thpt_coating checks


    def test_invalid_coating_file(self):
        """
        Missing coating file caught
        """
        thpt_coatings_path_invalid = \
            str(Path(LOCAL_PATH, 'ut_config',
                     'ut_thpt_coatings_invalid_file.yaml'))

        with self.assertRaises(IOError):
            load_thpt_configs(
                self.thpt_configs_path,
                thpt_coatings_path_invalid,
                self.thpt_data_path,
                self.thptcurves_dir_path,
            )




#----------------------------------------------------
# thpt_data checks


    def test_invalid_top_level(self):
        """
        Missing top-level key caught
        """
        thpt_data_path_invalid = \
            str(Path(LOCAL_PATH, 'ut_config',
                     'ut_thpt_data_invalid_folder.yaml'))

        with self.assertRaises(TypeError):
            load_thpt_configs(
                self.thpt_configs_path,
                self.thpt_coatings_path,
                thpt_data_path_invalid,
                self.thptcurves_dir_path,
            )

    def test_invalid_thptfile(self):
        """
        Missing file caught
        """
        thpt_data_path_invalid = \
            str(Path(LOCAL_PATH, 'ut_config',
                     'ut_thpt_data_invalid_thptfile.yaml'))

        with self.assertRaises(IOError):
            load_thpt_configs(
                self.thpt_configs_path,
                self.thpt_coatings_path,
                thpt_data_path_invalid,
                self.thptcurves_dir_path,
            )


    def test_load_thptcurve_invalid_thptcurve_path(self):
        """
        Verify invalid thptcurve_path fails as expected.
        """
        thpcurve_path_err = self.thptcurve_path + '_invalid'

        with self.assertRaises(IOError):
            _load_thptcurve(thpcurve_path_err)


    @patch('eetc.load.np.loadtxt')
    def test_load_thptcurve_invalid_vals(self, mock_loadtxt):
        """
        Verify invalid thptcurve values fail as expected.
        """
        thpt_len = 11
        lams = np.linspace(5500., 8570., thpt_len)
        thpts = np.linspace(0., 1., thpt_len)

        # lams
        for err in oneDlist:
            mock_loadtxt.return_value = (err, thpts)
            with self.assertRaises(TypeError):
                _load_thptcurve(self.thptcurve_path)

        # thpts
        for err in oneDlist:
            mock_loadtxt.return_value = (lams, err)
            with self.assertRaises(TypeError):
                _load_thptcurve(self.thptcurve_path)

        # different sized arrays
        mock_loadtxt.return_value = (lams, np.append(thpts, [1.]))
        with self.assertRaises(ValueError):
            _load_thptcurve(self.thptcurve_path)

        # invalid thpt values
        thpts0 = thpts.copy()
        thpts0[0] = -0.1
        mock_loadtxt.return_value = (lams, thpts0)
        with self.assertRaises(ValueError):
            _load_thptcurve(self.thptcurve_path)
        thpts1 = thpts.copy()
        thpts1[0] = 1.1
        mock_loadtxt.return_value = (lams, thpts1)
        with self.assertRaises(ValueError):
            _load_thptcurve(self.thptcurve_path)


class TestLoadLocamConfig(unittest.TestCase):
    """
    unit tests for loading in LOCAM config
    """

    def setUp(self):
        self.locam_config_path = str(Path(LOCAL_PATH, 'ut_config',
                                          'ut_locam_config.yaml'))

        self.darke = 8.33e-4
        self.cic = 0.02
        self.alpha0 = 0.7
        self.fwc = 50000
        self.alpha1 = 0.7
        self.fwc_em = 90000
        self.g_max_comm = 7500
        self.g_max_age = 200
        self.e_max_age = 12800
        self.tframe = 0.000441
        self.n = 4

        # Set up reference locam_config
        self.locam_config = {
            'darke': self.darke,
            'cic': self.cic,
            'alpha0': self.alpha0,
            'fwc': self.fwc,
            'alpha1': self.alpha1,
            'fwc_em': self.fwc_em,
            'g_max_comm': self.g_max_comm,
            'g_max_age': self.g_max_age,
            'e_max_age': self.e_max_age,
            'tframe': self.tframe,
            'n': self.n,
        }

    def test_invalid_string(self):
        """
        Verify invalid inputs fail as expected.
        """
        # excam_config_path
        for perr in strlist:
            with self.assertRaises(TypeError):
                load_locam_config(perr)

    def test_invalid_file_location(self):
        """
        Verify invalid file location fails as expected.
        """
        locam_config_path_err = self.locam_config_path + '_invalid'

        with self.assertRaises(IOError):
            load_locam_config(locam_config_path_err)

    def test_invalid_locam_config_keys(self):
        """
        Verify locam_config_keys validation.
        """
        locam_config_path_invalid = \
            str(Path(LOCAL_PATH, 'ut_config',
                     'ut_locam_config_invalid_keys.yaml'))

        with self.assertRaises(KeyError):
            load_locam_config(locam_config_path_invalid)


    @patch('eetc.load._unpack_locam_config')
    def test_invalid_locam_config_vals(self, mock_unpack_locam_config):
        """
        Verify invalid locam_config values fail as expected.
        """
        # darke
        for err in rnslist:
            mock_unpack_locam_config.return_value = (
                err,
                self.cic,
                self.alpha0,
                self.fwc,
                self.alpha1,
                self.fwc_em,
                self.g_max_comm,
                self.g_max_age,
                self.e_max_age,
                self.tframe,
                self.n,
            )
            with self.assertRaises(TypeError):
                load_locam_config(self.locam_config_path)

        # cic
        for err in rnslist:
            mock_unpack_locam_config.return_value = (
                self.darke,
                err,
                self.alpha0,
                self.fwc,
                self.alpha1,
                self.fwc_em,
                self.g_max_comm,
                self.g_max_age,
                self.e_max_age,
                self.tframe,
                self.n,
            )
            with self.assertRaises(TypeError):
                load_locam_config(self.locam_config_path)

        # alpha0
        for err in rpslist:
            mock_unpack_locam_config.return_value = (
                self.darke,
                self.cic,
                err,
                self.fwc,
                self.alpha1,
                self.fwc_em,
                self.g_max_comm,
                self.g_max_age,
                self.e_max_age,
                self.tframe,
                self.n,
            )
            with self.assertRaises(TypeError):
                load_locam_config(self.locam_config_path)

        # alpha0 < 1
        mock_unpack_locam_config.return_value = (
            self.darke,
            self.cic,
            2,
            self.fwc,
            self.alpha1,
            self.fwc_em,
            self.g_max_comm,
            self.g_max_age,
            self.e_max_age,
            self.tframe,
            self.n,
        )
        with self.assertRaises(ValueError):
            load_locam_config(self.locam_config_path)

        # fwc
        for err in psilist:
            mock_unpack_locam_config.return_value = (
                self.darke,
                self.cic,
                self.alpha0,
                err,
                self.alpha1,
                self.fwc_em,
                self.g_max_comm,
                self.g_max_age,
                self.e_max_age,
                self.tframe,
                self.n,
            )
            with self.assertRaises(TypeError):
                load_locam_config(self.locam_config_path)

        # alpha1
        for err in rpslist:
            mock_unpack_locam_config.return_value = (
                self.darke,
                self.cic,
                self.alpha0,
                self.fwc,
                err,
                self.fwc_em,
                self.g_max_comm,
                self.g_max_age,
                self.e_max_age,
                self.tframe,
                self.n,
            )
            with self.assertRaises(TypeError):
                load_locam_config(self.locam_config_path)

        # alpha1 < 1
        mock_unpack_locam_config.return_value = (
            self.darke,
            self.cic,
            self.alpha0,
            self.fwc,
            2,
            self.fwc_em,
            self.g_max_comm,
            self.g_max_age,
            self.e_max_age,
            self.tframe,
            self.n,
        )
        with self.assertRaises(ValueError):
            load_locam_config(self.locam_config_path)

        # fwc_em
        for err in psilist:
            mock_unpack_locam_config.return_value = (
                self.darke,
                self.cic,
                self.alpha0,
                self.fwc,
                self.alpha1,
                err,
                self.g_max_comm,
                self.g_max_age,
                self.e_max_age,
                self.tframe,
                self.n,
            )
            with self.assertRaises(TypeError):
                load_locam_config(self.locam_config_path)

        # g_max_comm
        for err in rpslist:
            mock_unpack_locam_config.return_value = (
                self.darke,
                self.cic,
                self.alpha0,
                self.fwc,
                self.alpha1,
                self.fwc_em,
                err,
                self.g_max_age,
                self.e_max_age,
                self.tframe,
                self.n,
            )
            with self.assertRaises(TypeError):
                load_locam_config(self.locam_config_path)

        # g_max_comm >= 1
        mock_unpack_locam_config.return_value = (
            self.darke,
            self.cic,
            self.alpha0,
            self.fwc,
            self.alpha1,
            self.fwc_em,
            0.9,
            self.g_max_age,
            self.e_max_age,
            self.tframe,
            self.n,
        )
        with self.assertRaises(ValueError):
            load_locam_config(self.locam_config_path)

        # g_max_age
        for err in rpslist:
            mock_unpack_locam_config.return_value = (
                self.darke,
                self.cic,
                self.alpha0,
                self.fwc,
                self.alpha1,
                self.fwc_em,
                self.g_max_comm,
                err,
                self.e_max_age,
                self.tframe,
                self.n,
            )
            with self.assertRaises(TypeError):
                load_locam_config(self.locam_config_path)


        # g_max_age >= 1
        mock_unpack_locam_config.return_value = (
            self.darke,
            self.cic,
            self.alpha0,
            self.fwc,
            self.alpha1,
            self.fwc_em,
            self.g_max_comm,
            0.9,
            self.e_max_age,
            self.tframe,
            self.n,
        )
        with self.assertRaises(ValueError):
            load_locam_config(self.locam_config_path)


        # e_max_age
        for err in psilist:
            mock_unpack_locam_config.return_value = (
                self.darke,
                self.cic,
                self.alpha0,
                self.fwc,
                self.alpha1,
                self.fwc_em,
                self.g_max_comm,
                self.g_max_age,
                err,
                self.tframe,
                self.n,
            )
            with self.assertRaises(TypeError):
                load_locam_config(self.locam_config_path)

        # tframe
        for err in rnslist:
            mock_unpack_locam_config.return_value = (
                self.darke,
                self.cic,
                self.alpha0,
                self.fwc,
                self.alpha1,
                self.fwc_em,
                self.g_max_comm,
                self.g_max_age,
                self.e_max_age,
                err,
                self.n,
            )
            with self.assertRaises(TypeError):
                load_locam_config(self.locam_config_path)

        # n
        for err in rnslist:
            mock_unpack_locam_config.return_value = (
                self.darke,
                self.cic,
                self.alpha0,
                self.fwc,
                self.alpha1,
                self.fwc_em,
                self.g_max_comm,
                self.g_max_age,
                self.e_max_age,
                self.tframe,
                err,
            )
            with self.assertRaises(TypeError):
                load_locam_config(self.locam_config_path)

    def test_unpack_locam_config(self):
        """
        Verify locam_config is unpacked correctly.
        """
        darke, cic, alpha0, fwc, alpha1, fwc_em, g_max_comm, g_max_age, \
            e_max_age, tframe, n = _unpack_locam_config(self.locam_config)

        self.assertEqual(darke, self.darke)
        self.assertEqual(cic, self.cic)
        self.assertEqual(alpha0, self.alpha0)
        self.assertEqual(fwc, self.fwc)
        self.assertEqual(alpha1, self.alpha1)
        self.assertEqual(fwc_em, self.fwc_em)
        self.assertEqual(g_max_comm, self.g_max_comm)
        self.assertEqual(g_max_age, self.g_max_age)
        self.assertEqual(e_max_age, self.e_max_age)
        self.assertEqual(tframe, self.tframe)
        self.assertEqual(n, self.n)



class TestLoadExcamConfig(unittest.TestCase):
    """
    Unit tests for load_excam_config function.
    """

    def setUp(self):
        self.excam_config_path = str(Path(UT_CONFIG_PATH,
                                          'ut_excam_config.yaml'))

        self.darke = 8.33e-4
        self.cic = 0.02
        self.rn = 200.
        self.X = 5.0e+04
        self.a = 1.69e-10
        self.Lij = 512
        self.alpha0 = 0.75
        self.fwc = 60000
        self.alpha1 = 0.85
        self.fwc_em = 100000
        self.Nmin = 1
        self.Nmax = 49
        self.tmin = 0.264
        self.tmax = 120.
        self.gmax = 5000.
        self.gconst = None
        self.n = 4
        self.Nem = 604
        self.tol = 1e-30
        self.delta_constr = 1e-4
        self.overhead = 3
        self.pc_ecount_max = 0.1
        self.T_factor = 5

        # Set up reference excam_config
        self.excam_config = {
            'darke': self.darke,
            'cic': self.cic,
            'rn': self.rn,
            'X': self.X,
            'a': self.a,
            'Lij': self.Lij,
            'alpha0': self.alpha0,
            'fwc': self.fwc,
            'alpha1': self.alpha1,
            'fwc_em': self.fwc_em,
            'Nmin': self.Nmin,
            'Nmax': self.Nmax,
            'tmin': self.tmin,
            'tmax': self.tmax,
            'gmax': self.gmax,
            'gconst': self.gconst,
            'n': self.n,
            'Nem': self.Nem,
            'tol': self.tol,
            'delta_constr': self.delta_constr,
            'overhead': self.overhead,
            'pc_ecount_max': self.pc_ecount_max,
            'T_factor': self.T_factor,
        }

    def test_invalid_string(self):
        """
        Verify invalid inputs fail as expected.
        """
        # excam_config_path
        for perr in strlist:
            with self.assertRaises(TypeError):
                load_excam_config(perr)

    def test_invalid_file_location(self):
        """
        Verify invalid file location fails as expected.
        """
        excam_config_path_err = self.excam_config_path + '_invalid'

        with self.assertRaises(IOError):
            load_excam_config(excam_config_path_err)

    def test_invalid_excam_config_keys(self):
        """
        Verify excam_config_keys validation.
        """
        excam_config_path_invalid = \
            str(Path(LOCAL_PATH, 'ut_config',
                     'ut_excam_config_invalid_keys.yaml'))

        with self.assertRaises(KeyError):
            load_excam_config(excam_config_path_invalid)

    @patch('eetc.load._unpack_excam_config')
    def test_invalid_excam_config_vals(self, mock_unpack_excam_config):
        """
        Verify invalid excam_config values fail as expected.
        """
        # darke
        for err in rnslist:
            mock_unpack_excam_config.return_value = (
                err, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # cic
        for err in rnslist:
            mock_unpack_excam_config.return_value = (
                self.darke, err, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # rn
        for err in rnslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, err, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # X
        for err in rnslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, err, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # a
        for err in rnslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, err,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # Lij
        for err in psilist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                err, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # alpha0
        for err in rpslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, err, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)
        # alpha0 > 1
        mock_unpack_excam_config.return_value = (
            self.darke, self.cic, self.rn, self.X, self.a,
            self.Lij, 2, self.fwc, self.alpha1, self.fwc_em,
            self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
        )
        with self.assertRaises(ValueError):
            load_excam_config(self.excam_config_path)

        # fwc
        for err in psilist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, err, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # alpha1
        for err in rpslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, err, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)
        # alpha1 > 1
        mock_unpack_excam_config.return_value = (
            self.darke, self.cic, self.rn, self.X, self.a,
            self.Lij, self.alpha0, self.fwc, 2, self.fwc_em,
            self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
        )
        with self.assertRaises(ValueError):
            load_excam_config(self.excam_config_path)

        # fwc_em
        for err in psilist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, err,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # Nmin
        for err in psilist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                err, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # Nmax
        for err in psilist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, err, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)
        # Nmax < Nmin
        mock_unpack_excam_config.return_value = (
            self.darke, self.cic, self.rn, self.X, self.a,
            self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
            2, 1, self.tmin, self.tmax, self.gmax,
            self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
            self.overhead, self.pc_ecount_max, self.T_factor
        )
        with self.assertRaises(ValueError):
            load_excam_config(self.excam_config_path)

        # tmin
        for err in rnslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, err, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # tmax
        for err in rnslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, err, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)
        # tmax < tmin
        mock_unpack_excam_config.return_value = (
            self.darke, self.cic, self.rn, self.X, self.a,
            self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
            self.Nmin, self.Nmax, 2., 1., self.gmax,
            self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
            self.overhead, self.pc_ecount_max, self.T_factor
        )
        with self.assertRaises(ValueError):
            load_excam_config(self.excam_config_path)

        # gmax
        for err in rnslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, err,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)
        # gmax < 1
        mock_unpack_excam_config.return_value = (
            self.darke, self.cic, self.rn, self.X, self.a,
            self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
            self.Nmin, self.Nmax, self.tmin, self.tmax, 0.9,
            self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
            self.overhead, self.pc_ecount_max, self.T_factor
        )
        with self.assertRaises(ValueError):
            load_excam_config(self.excam_config_path)

        # gconst
        # all of rpslist except for None since gconst can be None
        for err in [1j, (1.,), [5, 5], 'txt', -1, 0]:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                err, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)
        # gconst < 1
        mock_unpack_excam_config.return_value = (
            self.darke, self.cic, self.rn, self.X, self.a,
            self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
            self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
            0.9, self.n, self.Nem, self.tol, self.delta_constr,
            self.overhead, self.pc_ecount_max, self.T_factor
        )
        with self.assertRaises(ValueError):
            load_excam_config(self.excam_config_path)
        # gconst > gmax
        mock_unpack_excam_config.return_value = (
            self.darke, self.cic, self.rn, self.X, self.a,
            self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
            self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
            self.gmax+1, self.n, self.Nem, self.tol, self.delta_constr,
            self.overhead, self.pc_ecount_max, self.T_factor
        )
        with self.assertRaises(ValueError):
            load_excam_config(self.excam_config_path)

        # n
        for err in rnslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, err, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # Nem
        for err in psilist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, err, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # tol
        for err in rnslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, err, self.delta_constr,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # delta_constr
        for err in rnslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, err,
                self.overhead, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # overhead
        for err in rnslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                err, self.pc_ecount_max, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)


        # pc_ecount_max
        for err in rpslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, err, self.T_factor
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

        # T_factor
        for err in rpslist:
            mock_unpack_excam_config.return_value = (
                self.darke, self.cic, self.rn, self.X, self.a,
                self.Lij, self.alpha0, self.fwc, self.alpha1, self.fwc_em,
                self.Nmin, self.Nmax, self.tmin, self.tmax, self.gmax,
                self.gconst, self.n, self.Nem, self.tol, self.delta_constr,
                self.overhead, self.pc_ecount_max, err
            )
            with self.assertRaises(TypeError):
                load_excam_config(self.excam_config_path)

    def test_unpack_excam_config(self):
        """
        Verify excam_config is unpacked correctly.
        """
        darke, cic, rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em, Nmin, \
            Nmax, tmin, tmax, gmax, gconst, n, Nem, tol, delta_constr, \
            overhead, pc_ecount_max, T_factor \
            = _unpack_excam_config(self.excam_config)

        self.assertEqual(darke, self.darke)
        self.assertEqual(cic, self.cic)
        self.assertEqual(rn, self.rn)
        self.assertEqual(X, self.X)
        self.assertEqual(a, self.a)
        self.assertEqual(Lij, self.Lij)
        self.assertEqual(alpha0, self.alpha0)
        self.assertEqual(fwc, self.fwc)
        self.assertEqual(alpha1, self.alpha1)
        self.assertEqual(fwc_em, self.fwc_em)
        self.assertEqual(Nmin, self.Nmin)
        self.assertEqual(Nmax, self.Nmax)
        self.assertEqual(tmin, self.tmin)
        self.assertEqual(tmax, self.tmax)
        self.assertEqual(gmax, self.gmax)
        self.assertEqual(gconst, self.gconst)
        self.assertEqual(n, self.n)
        self.assertEqual(Nem, self.Nem)
        self.assertEqual(tol, self.tol)
        self.assertEqual(delta_constr, self.delta_constr)
        self.assertEqual(overhead, self.overhead)
        self.assertEqual(pc_ecount_max, self.pc_ecount_max)
        self.assertEqual(T_factor, self.T_factor)


class TestLoadFluxGrid(unittest.TestCase):
    """
    Unit tests for load_flux_grid function.
    """

    def setUp(self):
        self.flux_grid_path = str(Path(LOCAL_PATH, 'flux_grid_generation',
                                       'grid_files', 'flux_grid.fits'))
        self.hdul = _make_hdul()

        # Exclude tuple and list checks; they are not allowed in fits headers
        self.strlist_notuple = [el for el in strlist if not
                                isinstance(el, (tuple, list))]
        self.rslist_notuple = [el for el in rslist if not
                               isinstance(el, (tuple, list))]
        self.rpslist_notuple = [el for el in rpslist if not
                                isinstance(el, (tuple, list))]

    @patch('eetc.load.fits.open')
    def test_phot_keys(self, mock_open):
        """
        Verify phot values in PrimaryHDU header are assigned in the correct
        order to flux grid lists.
        """
        hdul = copy.deepcopy(self.hdul)
        hdul_dict = _hdul_to_dict(hdul)

        mock_open.return_value = hdul

        flux_grid = load_flux_grid(self.flux_grid_path)

        for phot, grid in flux_grid.items():
            flux_grid_vals, spts, cfams, ref_mag = grid
            hdu = hdul_dict[phot]
            # flux_grid_vals
            self.assertTrue((flux_grid_vals == hdu.data).all())
            # spts
            self.assertEqual(spts, hdu.header['SPECTYPE'].split(','))
            # cfams
            self.assertEqual(cfams, hdu.header['CFAMCOLS'].split(','))
            # ref_mag
            self.assertEqual(ref_mag, hdu.header['REFMAG'])

    @patch('eetc.load.fits.open')
    def test_invalid_phots_str(self, mock_open):
        """
        Verify invalid phots_str value fails as expected.
        """
        for err in self.strlist_notuple:
            hdul_err = copy.deepcopy(self.hdul)
            hdul_err[0].header['PHOTEXTS'] = err
            mock_open.return_value = hdul_err
            with self.assertRaises(TypeError):
                load_flux_grid(self.flux_grid_path)


    @patch('eetc.load.fits.open')
    def test_invalid_header_vals(self, mock_open):
        """
        Verify invalid header values fail as expected.
        """
        # cfams_str
        for err in self.strlist_notuple:
            hdul_err = copy.deepcopy(self.hdul)
            hdul_err[0].header['CFAMCOLS'] = err
            mock_open.return_value = hdul_err
            with self.assertRaises(TypeError):
                load_flux_grid(self.flux_grid_path)

        # spts_str
        for err in self.strlist_notuple:
            hdul_err = copy.deepcopy(self.hdul)
            hdul_err[0].header['SPECTYPE'] = err
            mock_open.return_value = hdul_err
            with self.assertRaises(TypeError):
                load_flux_grid(self.flux_grid_path)

        # ref_mag
        for err in self.rslist_notuple:
            hdul_err = copy.deepcopy(self.hdul)
            hdul_err[0].header['REFMAG'] = err
            mock_open.return_value = hdul_err
            with self.assertRaises(TypeError):
                load_flux_grid(self.flux_grid_path)


    @patch('eetc.load.fits.open')
    def test_invalid_data(self, mock_open):
        """
        Verify invalid data array fails as expected.
        """
        # Exclude non-array checks; hdu data must have at least one dimension
        twoDlist_arrays = [el for el in twoDlist if isinstance(el, np.ndarray)]

        for err in twoDlist_arrays:
            hdul_err = copy.deepcopy(self.hdul)
            hdul_err[0].data = err
            mock_open.return_value = hdul_err
            with self.assertRaises(TypeError):
                load_flux_grid(self.flux_grid_path)


class TestLoadWaveGrid(unittest.TestCase):
    """
    Unit tests for load_wave_grid function.
    """

    def setUp(self):
        self.wave_grid_path = str(Path(LOCAL_PATH, 'flux_grids',
                                       'wave_grid.fits'))
        self.hdul = _make_hdul()

        # Exclude tuple and list checks; they are not allowed in fits headers
        self.strlist_notuple = [el for el in strlist if not
                                isinstance(el, (tuple, list))]
        self.rslist_notuple = [el for el in rslist if not
                               isinstance(el, (tuple, list))]
        self.rpslist_notuple = [el for el in rpslist if not
                                isinstance(el, (tuple, list))]

    @patch('eetc.load.fits.open')
    def test_phot_keys(self, mock_open):
        """
        Verify phot values in PrimaryHDU header are assigned in the correct
        order to flux grid lists.
        """
        hdul = copy.deepcopy(self.hdul)
        hdul_dict = _hdul_to_dict(hdul)

        mock_open.return_value = hdul

        wave_grid = load_wave_grid(self.wave_grid_path)

        for phot, grid in wave_grid.items():
            wave_grid_vals, spts, cfams, ref_mag = grid
            hdu = hdul_dict[phot]
            # wave_grid_vals
            self.assertTrue((wave_grid_vals == hdu.data).all())
            # spts
            self.assertEqual(spts, hdu.header['SPECTYPE'].split(','))
            # cfams
            self.assertEqual(cfams, hdu.header['CFAMCOLS'].split(','))
            # ref_mag
            self.assertEqual(ref_mag, hdu.header['REFMAG'])

    @patch('eetc.load.fits.open')
    def test_invalid_phots_str(self, mock_open):
        """
        Verify invalid phots_str value fails as expected.
        """
        for err in self.strlist_notuple:
            hdul_err = copy.deepcopy(self.hdul)
            hdul_err[0].header['PHOTEXTS'] = err
            mock_open.return_value = hdul_err
            with self.assertRaises(TypeError):
                load_wave_grid(self.wave_grid_path)


    @patch('eetc.load.fits.open')
    def test_invalid_header_vals(self, mock_open):
        """
        Verify invalid header values fail as expected.
        """
        # cfams_str
        for err in self.strlist_notuple:
            hdul_err = copy.deepcopy(self.hdul)
            hdul_err[0].header['CFAMCOLS'] = err
            mock_open.return_value = hdul_err
            with self.assertRaises(TypeError):
                load_wave_grid(self.wave_grid_path)

        # spts_str
        for err in self.strlist_notuple:
            hdul_err = copy.deepcopy(self.hdul)
            hdul_err[0].header['SPECTYPE'] = err
            mock_open.return_value = hdul_err
            with self.assertRaises(TypeError):
                load_wave_grid(self.wave_grid_path)

        # ref_mag
        for err in self.rslist_notuple:
            hdul_err = copy.deepcopy(self.hdul)
            hdul_err[0].header['REFMAG'] = err
            mock_open.return_value = hdul_err
            with self.assertRaises(TypeError):
                load_wave_grid(self.wave_grid_path)


    @patch('eetc.load.fits.open')
    def test_invalid_data(self, mock_open):
        """
        Verify invalid data array fails as expected.
        """
        # Exclude non-array checks; hdu data must have at least one dimension
        twoDlist_arrays = [el for el in twoDlist if isinstance(el, np.ndarray)]

        for err in twoDlist_arrays:
            hdul_err = copy.deepcopy(self.hdul)
            hdul_err[0].data = err
            mock_open.return_value = hdul_err
            with self.assertRaises(TypeError):
                load_wave_grid(self.wave_grid_path)


if __name__ == '__main__':
    unittest.main()
