# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
'''
Package for creating and scaling synthetic spectra for CGI EETC
S Halverson - JPL - 29-Sep-2019
'''
import math
import os
import yaml
from astropy import constants as const
from astropy import units as u
import numpy as np
import eetc.util.check as check
import eetc

try: # numpy 2.0 and later
    from numpy import trapezoid
except ImportError: # numpy 1.x
    from numpy import trapz as trapezoid

# global variables
#---------------------------------------
# sort out paths
LOCAL_PATH = os.path.abspath(os.path.dirname( __file__ ))

# manitude zero point definitions
MAG_ZERO_POINT_FILE = os.path.join(LOCAL_PATH,'config',
                                   'vega_magnitude_zeropoints.yaml')
#---------------------------------------

class EETCFLUXGRIDGENERATEException(Exception):
    """Exception class for flux_grid_generate_tools module."""

# Loads BRUZUAL-PERSSON-GUNN-STRYKER stellar spectrum from specified path
# for a given spectral type
#----------------------------------------
def _spectrum_load_csv(spt, spec_path, wvl=None):

    """
    Loads spectrum from library of stellar spectra, assuming a two
    column csv file (wavelength, flux), ending with '.txt'

    Currently using a local copy of the BPGS atlas (folder of csv files):

    https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/bpgs/

    Parameters
    ----------
    spt : string
        Stellar spectral type -- must match file name, which ends with '.txt'

    spec_path : string
        Directory path for library of model spectra

    wvl : array
        Array of wavelengths to interpolate spectrum onto (optional)

    Returns
    -------
    output_dict : dictionary
        Output spectrum, referenced by {'flux','wvl'}
        'flux': output flux spectrum, in units of [Ergs/sec/cm^2/A]
        'wvl': output wavelength array of 'flux' [Ang]

    S Halverson - JPL - 29-Sep-2019
    """

    # input checks
    check.string(spt, 'spt', TypeError)
    check.string(spec_path, 'spec_path', TypeError)

    # path to fits file directory containing spectra
    spec_csv = os.path.join(spec_path, spt + '.txt')

    # Read file
    try:
        wvl_model, flux_model = np.loadtxt(spec_csv, delimiter=',', unpack=True)
    except FileNotFoundError:
        raise IOError('spectrum file not found')

    # Check arrays
    check.oneD_array(wvl_model, 'lams', TypeError)
    check.oneD_array(flux_model, 'thpts', TypeError)
    if len(wvl_model) != len(flux_model):
        raise ValueError('wavelength and flux arrays in model spectrum file must '
                         'be the same length')
    if (flux_model < 0).any():
        raise ValueError('flux_model values must be >= 0')
    if (np.nanmin(wvl_model) > 4000.) or (np.nanmax(wvl_model) < 9000.):
        raise ValueError('wvl_model does not cover required range for interpolation')

    # restrict wavelength range to optical (cuts down on array sizes)
    flux_model = flux_model[(wvl_model > 3000.) & (wvl_model < 10000.)] #Ergs/sec/cm^2/A
    wvl_model = wvl_model[(wvl_model > 3000.) & (wvl_model < 10000.)]   #Ang

    if len(flux_model) == 0:
        raise EETCFLUXGRIDGENERATEException('No wavelengths between 3000-10000 Ang in spectrum')

    # if wvl array is provided, interpolate spectrum onto specified array -- else use default
    if np.any(wvl):
        # interpolate elememnt transmission curve onto reference wavelength array
        flux_model = np.interp(wvl, wvl_model, flux_model)
        wvl_master = wvl
    else:
        wvl_master = wvl_model

    output_dict = {'flux':flux_model, 'wvl':wvl_master}
    return output_dict
#----------------------------------------

# Scales provided spectrum to specified magnitude in photometric band
#-------------------------------------------------=
def _spectrum_scale(wvl, spec, mag, filter_band, mag_zero_point_file=MAG_ZERO_POINT_FILE):

    """
    Performs scaling of input spectrum to specified magnitude,
    returns scaled stellar spectrum.

    Requires relevant photometric filter response curve to be saved as text file
    under 'astro_filters_Generic_Jonson.N.txt', where 'N' is the filter band.

    Parameters
    ----------
    wvl : :obj:`ndarray` of :obj:`float`
        Wavelength array of input spectrum [Ang]

    spec : :obj:`ndarray` of :obj:`float`
        Input spectrum [Ergs/sec/cm^2/Ang]

    mag : float
        Target magnitude to scale input spectrum to

    filter_band : string
        filter band of 'mag' - must have corresponding transmission file

    mag_zero_point_file : string
        YAML file conaining magnitude zero point offsets for Vega and
        pointers to generic filter response curve text files

    Returns
    -------
    flxu_scaled : :obj:`ndarray` of :obj:`float`
        Scaled spectrum in flux units [Ergs/sec/cm^2/Ang]

    S Halverson - JPL - 29-Sep-2019
    """
    # input checks
    check.oneD_array(wvl, 'wvl', TypeError)
    check.oneD_array(spec, 'spec', TypeError)
    check.real_positive_scalar(mag, 'mag', TypeError)
    check.string(filter_band, 'filter_band', TypeError)
    check.string(mag_zero_point_file, 'mag_zero_point_file', TypeError)

    # capitalize for convinience
    filter_band = filter_band.capitalize()

    # Vega zero point magnitude numbers for specified band
    with open(mag_zero_point_file, 'r') as stream:
        data_filters = yaml.safe_load(stream)

    # reference data
    # lambda_filter = data_filters[filter_band]['lambda0']  	#Ang, central wavelength of filter
    flux_filter_0 = data_filters[filter_band]['flux0'] 	#erg/sec/cm^2/A, Vega zeropoint
    lambda_wid_filter = data_filters[filter_band]['filter_wid']   	#A, effective filter width

    # load in filter data from relevant text tile
    filter_file = os.path.join(LOCAL_PATH, data_filters[filter_band]['filter_file'])
    filter_data = np.loadtxt(filter_file)
    wvl_filter = filter_data[:, 0]    #Ang
    trans_filter = filter_data[:, 1]    #filter transmission

    # interpolate filter transmission curve onto reference wavelength array
    filter_trans = np.interp(wvl, wvl_filter, trans_filter)
    filter_trans[(wvl > np.max(wvl_filter))] = 0.
    filter_trans[(wvl < np.min(wvl_filter))] = 0.

    # multiply filter profile by spectrum
    flux = spec * filter_trans  	#Ergs/sec/cm^2/Ang

    # integrate combined spectrum
    filter_flux_filter = trapezoid(flux, wvl)  #Ergs/sec/cm^2

    # get effective magnitude of model spectrum
    offset_filter = 2.5 * math.log10(flux_filter_0 * lambda_wid_filter)	#mag, Vega zero point offset
    mag_model = (-2.5) * math.log10(filter_flux_filter) + offset_filter	#mag

    # compare vmag from magnitude estimate routine to library version
    mag_diff = mag_model - mag	#mag
    multi_fac = 10. ** (0.4 * mag_diff)	#scaling factor

    # final output spectrum
    flux_scaled = multi_fac * spec #Ergs/sec/cm^2/A

    # independantly check if the output magnitude matches the desired input
    # mag_model_out = (-2.5) * math.log10(trapezoid(flux_scaled * filter_trans, wvl)) +
    #                  offset_filter
    return flux_scaled
#-------------------------------------------------

# Estimates photometric magnitude of provided spectrum
#-------------------------------------------------=
def _spectrum_mag_estimate(wvl, spec, filter_band, mag_zero_point_file=MAG_ZERO_POINT_FILE):
    """
    Estimates the brightness magnitude of an input spectrum within a specified
    photometric filter.

    Input units must be Ergs/sec/cm^2/Ang (spec) and Ang (wvl)

    Parameters
    ----------
    wvl : :obj:`ndarray` of :obj:`float`
        Wavelength array of input spectrum [Ang]

    spec : :obj:`ndarray` of :obj:`float`
        Input spectrum [Ergs/sec/cm^2/Ang]

    mag : float
        Target magnitude to scale input spectrum to

    filter_band : string
        filter band of 'mag' - must have corresponding transmission file

    mag_zero_point_file : string
        YAML file conaining magnitude zero point offsets for Vega

    Returns
    -------
    flxu_scaled : :obj:`float`
        Magnitude of input spectrum in specified band

    S Halverson - JPL - 29-Sep-2019
    """
    #  input checks
    check.oneD_array(wvl, 'wvl', TypeError)
    check.oneD_array(spec, 'spec', TypeError)
    check.string(filter_band, 'filter_band', TypeError)
    check.string(mag_zero_point_file, 'mag_zero_point_file', TypeError)

    # capitalize for convinience
    filter_band = filter_band.capitalize()

    # Vega zero point magnitude numbers for specified band
    with open(mag_zero_point_file, 'r') as stream:
        data_filters = yaml.safe_load(stream)

    # reference data
    flux_filter_0 = data_filters[filter_band]['flux0'] 	#erg/sec/cm^2/A, Vega zeropoint
    lambda_wid_filter = data_filters[filter_band]['filter_wid']   	#A, effective filter width

    # load in filter data from relevant text tile
    filter_file = os.path.join(LOCAL_PATH, data_filters[filter_band]['filter_file'])
    filter_data = np.loadtxt(filter_file)
    wvl_filter = filter_data[:, 0]    #Ang
    trans_filter = filter_data[:, 1]    #filter transmission

    # interpolate filter transmission curve onto reference wavelength array
    filter_trans = np.interp(wvl, wvl_filter, trans_filter)
    filter_trans[(wvl > np.max(wvl_filter))] = 0.
    filter_trans[(wvl < np.min(wvl_filter))] = 0.

    # multiply filter profile by spectrum
    flux = spec * filter_trans  	#Ergs/sec/cm^2/Ang

    # integrate combined spectrum
    filter_flux_filter = trapezoid(flux, wvl)   #Ergs/sec/cm^2

    # get effective magnitude of the 'model' spectrum
    offset_filter = 2.5 * math.log10(flux_filter_0 * lambda_wid_filter)	#mag, Vega zero point offset
    mag_spectrum = (-2.5) * math.log10(filter_flux_filter) + offset_filter	#mag

    return mag_spectrum
#-------------------------------------------------=

# Estimates amount of integrated flux in photometric filter, given an incident flux
#-------------------------------------------------
def _cfam_band_integrate(wvl, spec, filter_name, filter_dir):
    """
    Integrates flux spectrum inside specified CGI filter band and returns
    photons / second

    Parameters
    ----------
    wvl : :obj:`ndarray` of :obj:`float`
        Wavelength array of input spectrum [Ang]

    spec : :obj:`ndarray` of :obj:`float`
        Input spectrum at entrance to OTA [Ergs/sec/Ang]

    filter_name : string
        Name of optic plane containing filter (default is CFAM).
        Must follow directory structure: optic_plane/filter.txt

    filter_dir : string
        directory name of folder containing CFAM filter files

    Returns
    -------
    flux_rate : float
        Integrated flux rate in CFAM filter at entrance to OTA (photons/s)

    cfam_center_wvl : float
        Flux-weighted wavelength of spectrum through CFAM filter

    S Halverson - JPL - 29-Sep-2019
    """
    # input checks
    check.oneD_array(wvl, 'wvl', TypeError)
    check.oneD_array(spec, 'spec', TypeError)
    check.string(filter_name, 'filter_name', TypeError)
    check.string(filter_dir, 'filter_dir', TypeError)

    # get transmission profile of specified filter
    filter_data = _get_cfam_transmission(filter_dir, filter_name, wvl)
    filter_trans = filter_data['trans']

    # get flux-weighted wavelength midpoint of filter.
    band_wvl = np.average(wvl, weights=filter_trans)  # ang
    band_wvl_cm = band_wvl * 1e-8 * u.cm  # cm

    # calculate average photon energy in band
    # c = 2.997e10    # cm/s
    # h = 6.6261e-27  # cm^2 g/s
    photon_energy = (const.h.cgs * const.c.cgs / band_wvl_cm) # erg / photon

    # spectrum multiplied by the filter transmission
    flux_band = spec * filter_trans * u.erg / (u.AA * u.s)   # Ergs/sec/Ang
    flux_band_photons = flux_band / photon_energy   # photons / s / Ang

    # integrate modulated spectrum in wavelength
    flux_rate = trapezoid(flux_band_photons, wvl * u.AA)   # photons/sec

    # integrate modulated spectrum in flux -- returns central wavelength
    cfam_center_wvl = (np.nansum(flux_band_photons * (wvl * u.AA)) /
                       np.nansum(flux_band_photons))

    return flux_rate.value, cfam_center_wvl.value #photons/sec
#-------------------------------------------------

# pulls efficiency for single CGI element or generic filter
#-------------------------------------------------------
def _get_cfam_transmission(filter_dir, filter_name, wvl):
    '''
    Parameters
    ----------
    filter_dir : string
        Name of master directory containing all element efficiency files.

    filter_name : string
        Name of CFAM filter -- must match file name (e.g. '1A' for 1A.csv)
        Filter file assumed to have a 4 line header.

    wvl : :obj:`ndarray` of :obj:`float`
        Reference wavelength array for which to load element efficiency curve
        onto [Ang]. Linear interpolation assumed if individual transmission
        files have different samplings.

    Returns
    -------
    output_element : dictionary of arrays
        Output dictionary strucuted with 'element' (name),
        'wvl' (wavelength array), and 'trans' (transmission array of element)

    S Halverson - JPL - 29-Sep-2019
    '''
    # check inputs
    check.oneD_array(wvl, 'wvl', TypeError)
    check.string(filter_dir, 'filter_dir', TypeError)
    check.string(filter_name, 'filter_name', TypeError)

    # load in filter data from relevant text tile
    element_efficiency_file = os.path.join(filter_dir, filter_name + '.csv')

    # Read file
    try:
        wvl_file, trans_file = np.loadtxt(element_efficiency_file,
                                          delimiter=',', skiprows=4,
                                          unpack=True)
    except FileNotFoundError:
        raise IOError('CFAM filter curve file not found')

    wvl_file *=  10.   # convert from nm to Ang
    trans_file *= 1e-2    # convert from percentage to fraction

    # Check arrays
    check.oneD_array(wvl_file, 'wvl_file', TypeError)
    check.oneD_array(trans_file, 'trans_file', TypeError)
    if len(wvl_file) != len(trans_file):
        raise ValueError('CFAM wavelength and transmission arrays must '
                         'be the same length')
    if (trans_file < 0).any() or (trans_file > 1).any():
        raise ValueError('trans_file values must be >0 and <100')
    if (np.min(wvl_file) < np.min(wvl)) or (np.max(wvl_file) > np.max(wvl)):
        raise ValueError('CFAM file wavelengths outside of bounds of stellar '
                         'model')

    wvl_master = wvl_file
    transmission_out = trans_file

    # if wavelength grid is specified, interpolate filter response
    if wvl is not None:
        wvl_master = wvl
        # interpolate elememnt transmission curve onto reference wavelength array
        transmission_out = np.interp(wvl_master, wvl_file, trans_file)

        # zero out any points beyond the range of the file -- helps with interpolation errors
        transmission_out[(wvl_master > np.max(wvl_file))] = 0.
        transmission_out[(wvl_master < np.min(wvl_file))] = 0.
    else:
        print('No wavelengths provided -- returning natively sampled CFAM curve')
    # final output
    output_element = {'element':filter_name, 'wvl':wvl_master, 'trans':transmission_out}
    return output_element
#-------------------------------------------------------

# Estimates amount of integrated flux in CFAM filter for specified target
#-------------------------------------------------
def spec_generate_cgi_cfam(wvl, mag, filter_band, spt, cfam_filter,
                           spec_dir, cfam_filter_dir, collecting_area):

    """
    Calculates flux estimate of synthetic spectrum at specified
    photometric magnitude in specified filter at entrance of OTA (pre-CGI)
    through specific CFAM filter

    Parameters
    ----------
    wvl : float array
        Wavelength array of input spectrum [Ang]

    mag : float
        Target magnitude to scale input spectrum

    filter_band : string
        filter band of 'mag' - must have corresponding transmission file

    spt : string
        Model spectral type

    cfam_filter : string
        CFAM filter name

    spec_dir : string
        directory containing individual stellar spectra csv files

    cfam_filter_dir : string
        directory containing CFAM filter curves csv files

    collecting_area : float
        total OTA collecting area (including obscuration)

    Returns
    -------
    flux_at_instrument_entrance : float
        Flux rate through CFAM filter of specified target

    wvl_centroid : float
        Calculated flux-weighted wavelength centroid of specified target
        through specified CFAM filter

    mag_output_dict : dict
        Dictionary containing the estimated photometric magnitudes in other
        bands.

    flux_entrance : array of floats
        Synthetic spectrum scaled to specified magnitude entering OTA

    S Halverson - JPL - 29-Sep-2019
    """
    # input checks
    check.oneD_array(wvl, 'wvl', TypeError)
    check.real_positive_scalar(mag, 'mag', TypeError)
    check.string(filter_band, 'filter_band', TypeError)
    check.string(spt, 'spt', TypeError)
    check.string(cfam_filter, 'cfam_filter', TypeError)
    check.string(spec_dir, 'spec_dir', TypeError)
    check.string(cfam_filter_dir, 'cfam_filter_dir', TypeError)
    check.real_positive_scalar(collecting_area, 'collecting_area', TypeError)

    # load in model spectrum of appropriate temperature
    spec_model = _spectrum_load_csv(spt, spec_dir) #Ergs/sec/cm^2/Ang
    wvl_file = spec_model['wvl']
    flux_file = spec_model['flux']

    # interpolate stellar spectrum onto reference wavelength array
    flux = np.interp(wvl, wvl_file, flux_file)

    # zero out any wavelength values beyond the range of the file wavelengths
    flux[(wvl > np.nanmax(wvl_file))] = 0.
    flux[(wvl < np.nanmin(wvl_file))] = 0.

    # scale spectrum to specified band magnitude
    flux_scaled = _spectrum_scale(wvl, flux, mag, filter_band) #Ergs/sec/cm^2/Ang

    # spectrum at OTA entrance
    flux_entrance = flux_scaled * collecting_area #Ergs/sec/A

    # estimate recorded flux in specified band entering telescope
    flux_at_cgi, wvl_centroid = _cfam_band_integrate(wvl, flux_entrance,
                                cfam_filter, cfam_filter_dir) # photons / sec

    return flux_at_cgi, wvl_centroid
#-------------------------------------------------
