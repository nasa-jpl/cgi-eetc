# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
#---CGI EETC flux grid generator----
'''
Wrapper function for computing stellar flux rates and weighted CFAM filter
wavelength centroids for CGI engineering exposure time calculator
'''
import argparse
import os
import glob
from astropy.io import fits
import yaml
import numpy as np
from eetc.util.validate import validate_dict_keys
import eetc.util.check as check
from eetc.flux_grid_generation.flux_grid_generate_tools import spec_generate_cgi_cfam
import eetc

# sort out paths
LOCAL_PATH = os.path.dirname(os.path.realpath(__file__))

# YAML file that has grid generation parameters
YAML_GRID_INPUT = os.path.join(os.path.join(LOCAL_PATH,'config'),
                               'flux_grid_generate_input_params.yaml')

VALID_POINTER_KEYS = [
    'magref', 'magbands', 'wvl',
    'input_files', 'output_flux_grid_file',
    'output_wave_grid_file', 'ota_collecting_area'
]

def _abs_or_rel_path(path, rel=None):
    """
    Takes what might be an absolute or relative path and returns an
    absolute one.  What the path is relative to is specified by the
    optional input rel, which must be an absolute path itself.
    """
    check.string(path, path, TypeError)
    if rel is None:
        rel = os.path.dirname(os.path.realpath(__file__))
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(rel, path)

class EETCFLUXGRIDMAKEException(Exception):
    """Exception class for flux_grid_generate_tools module."""

# read in input parameters from yaml file
def cgi_flux_grid_generate(input_file=YAML_GRID_INPUT,
                           output_flux_grid_file=None,
                           output_wave_grid_file=None):
    '''
    This function produces a grid of flux rates at the top of the telescope through the
    specified CFAM filter, and does not include OTA or other CGI optics. Downstream
    functions (cgi_eetc) include both CGI and OTA throughput curves.

    Generates full grid of flux rates for all CFAM filters, as function of
    target spectral type and photometric filter, and saves as a FITS file
    formatted in the following way:

    Flux[CFAM (column), photometric filter (FITS extension), spectral type (row))]

    Metadata describing grid basis vectors is saved in the FITS
    file header using a suite of header keywords:

    ['REFBAND']:   photometric filter at which fluxes were computed
    ['REFMAG']:    photometric magnitude at which fluxes were computed
    ['SPECTYPE']:   array of stellar spectral types at which fluxes were computed
    ['CFAMCOLS']:  CFAM filter column names in flux grid
    ['PHOTEXTS']:  photometric filter FITS extensions

    Input config file must have the following dictionary keys (values are examples):

        magref: 6 (double)
        magbands:  ['b','v','r','i'] (string of filter bands; must be these four or some subset of them)
        # master wavelength array parameters [Ang]
        wvl:
            min:  3000.
            max:  10000.
            step: 1.
        input_files:
            cfam_filters: 'config/cfam_filter_wavelengths.yaml'
            cfam_filter_curve_path: 'cfam_filter_curves'
            stellar_model_spec_path: 'bpgs_atlas_csv'
        output_flux_grid_file: 'grid_files/flux_grid.fits'
        output_wave_grid_file: 'grid_files/wave_grid.fits'
        ota_collecting_area: 35895.212  # cm^2

    CFAM filter curves assumed to be csv files,
    with naming convention of '_filter_name_.csv'
    formatted in two columns -- wavelength
    in nanometers, and % transmission:

    lambda_nm,%T
    563.5,1.451493979
    564,4.515580654
    ...,...

    Stellar model spectra are assumed to be csv files,
    with naming convention '_spec_type_.txt',
    formatted in two columns -- wavelength
    in Angstrom, f_lambda in [Erg/sec/cm^2/Ang]:

    wavelength [Ang], f_lambda [Erg/sec/cm^2/Ang]
    2.2900e+02,0.0000e+00
    2.3400e+02,0.0000e+00
    2.4300e+02,0.0000e+00
    ...,...

    Parameters
    ----------
    input_file : string
        Input yaml file detailing grid generation parameters (discussed above).
        Path can be relative to the flux_grid_generation folder or absolute.
    output_flux_grid_file : string
        File path to write flux grid output to, or None.  If None, will read
        the path from the file in input_file.  Path can be relative to the
        flux_grid_generation folder or absolute.
    output_wave_grid_file : string
        File path to write wave grid output to, or None.  If None, will read
        the path from the file in input_file.  Path can be relative to the
        flux_grid_generation folder or absolute.

    S Halverson - JPL - 15-Dec-2019
    '''

    # check format of input file
    input_file = _abs_or_rel_path(input_file)
    try:
        with open(input_file, 'r') as stream:
            input_dicts = (yaml.safe_load(stream))
    except FileNotFoundError:
        raise IOError('pointer file not found')
    validate_dict_keys(input_dicts, VALID_POINTER_KEYS)

    # parse yaml dictionary filenames and pointers
    #---------------------------
    # master flux grid file
    if output_flux_grid_file is not None:
        check.string(output_flux_grid_file, 'output_flux_grid_file', TypeError)
        grid_fits_filename = output_flux_grid_file
    else:
        grid_fits_filename = _abs_or_rel_path(
                                        input_dicts['output_flux_grid_file'])

    # wave centroid grid file name
    if output_wave_grid_file is not None:
        check.string(output_wave_grid_file, 'output_wave_grid_file', TypeError)
        grid_wave_fits_filename = output_wave_grid_file
    else:
        grid_wave_fits_filename = _abs_or_rel_path(
                                        input_dicts['output_wave_grid_file'])

    # path to stellar spectra
    spec_dir = _abs_or_rel_path(
                        input_dicts['input_files']['stellar_model_spec_path'])

    # path to CFAM filter curve csv files
    cfam_filter_dir = _abs_or_rel_path(
                        input_dicts['input_files']['cfam_filter_curve_path'])

    # OTA collecting area
    collecting_area = input_dicts['ota_collecting_area'] # cm^2
    check.real_positive_scalar(collecting_area, 'collecting_area', TypeError)

    # check to make sure there are spectra in spec_dir
    spec_csv_list = glob.glob(os.path.join(spec_dir,'*.txt'))
    if len(spec_csv_list) < 1:
        raise EETCFLUXGRIDMAKEException('Spectrum txt files not found in '
                                        'specified directory')

    #------------------------------

    # flux grid meta-data
    #---------------------------------
    # reference magnitude at which flux rates are computed
    mag_reference = input_dicts['magref']
    check.real_scalar(mag_reference, 'mag_reference', TypeError)

    # reference set of photometric bands
    mag_reference_band_arr = input_dicts['magbands'] # reference bands
    check.oneD_array(mag_reference_band_arr, 'mag_reference_band_arr', TypeError)
    mag_reference_band_arr = [x.lower() for x in mag_reference_band_arr]

    # reference stellar spectral type array
    target_spt_master = []
    target_spt_list = os.listdir(spec_dir)
    for spt in target_spt_list:
        if spt.endswith('.txt'):
            # cut of the '.txt' and uppercase:
            target_spt_master.append(spt[:-4].upper())

    # master wavelength array used to generate spectra
    wvl_min = input_dicts['wvl']['min']
    check.real_positive_scalar(wvl_min, 'wvl_min', TypeError)

    wvl_max = input_dicts['wvl']['max']
    check.real_positive_scalar(wvl_max, 'wvl_max', TypeError)

    wvl_step_size = input_dicts['wvl']['step']
    check.real_positive_scalar(wvl_step_size, 'wvl_step_size', TypeError)

    if wvl_min > wvl_max:
        raise EETCFLUXGRIDMAKEException('Specified beginning wavelength larger than ending')
    wvl = np.arange(wvl_min, wvl_max, wvl_step_size)
    #---------------------------------

    # CFAM filter list from file names
    # ----------------------------
    cfam_filter_arr = []
    cfam_filter_list = os.listdir(cfam_filter_dir)
    for cfam in cfam_filter_list:
        if cfam.endswith('.csv'):
            # cut of the '.csv' and uppercase:
            cfam_filter_arr.append(cfam[:-4].upper())

    # ----------------------------

    # grid generation
    #----------------------------------------
    nspt = len(target_spt_master)
    ncfam_filters = len(cfam_filter_arr)

    # output flux at entrance to telescope, integrated in CFAM filter bands
    # target filter band loop
    for ind_mag, mag_reference_band in enumerate(mag_reference_band_arr):

        # master flux rate matrix
        flux_rate_master_grid = np.empty([nspt, ncfam_filters])
        wave_centroid_master_grid = np.empty([nspt, ncfam_filters])

        # for each spectral type, calculate flux rate (photons/sec) through CFAM filter
        for ind_spt, target_spt in enumerate(target_spt_master):

            # cfam filter loop
            for ind_cfam, cfam_filter in enumerate(cfam_filter_arr):

                # calculate flux rate, scaled to reference magnitude in reference band
                flux_rate, wvl_centroid = spec_generate_cgi_cfam(wvl, mag_reference,
                                                                       mag_reference_band,
                                                                       target_spt, cfam_filter,
                                                                       spec_dir, cfam_filter_dir,
                                                                       collecting_area)

                # store filter fluxes in master grid matrix
                flux_rate_master_grid[ind_spt, ind_cfam] = flux_rate # photons / s
                wave_centroid_master_grid[ind_spt, ind_cfam] = wvl_centroid # Angstrom

        # if on first filter, save grid to fits new file
        if ind_mag == 0:
            # fits.writeto('output_file.fits', data, header, clobber=True)
            hdr = fits.Header()
            hdr['REFBAND'] = mag_reference_band # photometric filter at which fluxes were computed
            hdr['REFMAG'] = mag_reference # photometric magnitude at which fluxes were computed
            hdr['SPECTYPE'] = ','.join(target_spt_master) # spectral type
            hdr['CFAMCOLS'] = ','.join(cfam_filter_arr) # CFAM filter column names in flux grid
            hdr['PHOTEXTS'] = ','.join(mag_reference_band_arr) # photometric filter fits extensions

            # flux grid fits file
            fits.writeto(grid_fits_filename, flux_rate_master_grid, hdr, overwrite=True)

            # wavelength centroid grid fits file
            fits.writeto(grid_wave_fits_filename, wave_centroid_master_grid, hdr, overwrite=True)

        # else append existing fits file with next filter
        else:
            #append reference magnitude definitions to header
            hdr = fits.Header()
            hdr['REFBAND'] = mag_reference_band # photometric filter at which fluxes were computed
            hdr['REFMAG'] = mag_reference # photometric magnitude at which fluxes were computed
            hdr['SPECTYPE'] = ','.join(target_spt_master) # spectral type
            hdr['CFAMCOLS'] = ','.join(cfam_filter_arr) # CFAM filter column names in flux grid
            hdr['PHOTEXTS'] = ','.join(mag_reference_band_arr) # photometric filter fits extensions
            fits.append(grid_fits_filename, flux_rate_master_grid, hdr)
            fits.append(grid_wave_fits_filename, wave_centroid_master_grid, hdr)
#-----------------------------

if __name__ == "__main__":
    # use argparse so we can do functional tests without changing the code
    ap = argparse.ArgumentParser(
        prog='python cgi_flux_grid_make.py',
        description="Compute grid of spectral flux rates as a function of " +
                    "stellar type and magnitude"
    )

    # get relevant argument(s)
    ap.add_argument('--input_file', default=YAML_GRID_INPUT, help="Input config file.",
                    type=str)
    args = ap.parse_args()
    _input_file = args.input_file

    # run function
    cgi_flux_grid_generate(input_file=_input_file)
