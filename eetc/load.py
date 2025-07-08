# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Functions for loading, unpacking, parsing, and validating eetc files.
"""
import copy
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

import eetc.util.check as check
from eetc.util.validate import validate_dict_keys
from eetc.constants import (
    valid_sequence_keys,
    valid_thpt_data_keys,
    header_primary_keys,
    header_custom_keys,
    header_ext_keys,
    valid_excam_config_keys,
    valid_locam_config_keys,
)

def load_sequences(sequences_path, valid_modes, valid_spam_lsam, valid_fpam,
                   valid_fsam, validating_cfam, valid_dpam):
    """
    Load sequences file.

    Validity-checking arguments are determined on-the-fly by the calling
    function based on read-in data from configuration files, to allow for
    future expansion without software change.  (As opposed to reading them in
    from constants.py, which is done for checking values that have something
    hardcoded about them.)

    Parameters
    ----------
    sequences_path : str
        Full path of sequences yaml file.

    valid_modes : set
        set of strings, each of which is a mode (set of optical surfaces)
        supported by thpt_config data.

    valid_spam_lsam : set
        set of strings, each of which is a SPAM/LSAM position supported by
        thpt_data data.

    valid_fpam : set
        set of strings, each of which is a FPAM position supported by
        thpt_data data.

    valid_fsam : set
        set of strings, each of which is a FSAM position supported by
        thpt_data data.

    validating_cfam : set
        set of strings, each of which is a CFAM filter name used to create the
        flux grid.  This is the list against which to check, as opposed
        to the other inputs which are lists to be checked against valid lists
        coming from constants.py.

    valid_dpam : set
        set of strings, each of which is a DPAM position supported by
        thpt_data data.


    Returns
    -------
    sequences : dict
        Dictionary containing sequence subdictionaries.

        Keys are sequence names, values are sequence subdictionaries.

        Each sequence subdictionary should contain the following keys and
        values:

        mode : str
            Observation mode. This sets the throughput of the full CGI +
            front-end system (OTA, CVS) and all of the static optics in CGI,
            including the 'throughput' associated with the camera QE.

        spam_lsam : str
            Configuration of the SPAM and LSAM masks.  Combined into one
            element as two masks sometimes block the same area, and so the
            attentuation due to both masks can't be computed from the product
            of each mask individually--it needs to be solved jointly.

        fpam : str
            Configuration of the FPAM masks and ND filters

        fsam : str
            Configuration of the FSAM masks and ND filters

        cfam : str
            Configuration of the CFAM color filters

        dpam : str
            Configuration of the DPAM lenses, prisms, and polarizers

        peak_flux_ratio_pix : float
            Ratio of peak flux per pixel to integrated flux for specified
            observation mode.

        fraction : float
            Fraction of total image area flux corresponding to resel.  >=0 and
            <=1.

        num_pixels : float
            Number of pixels corresponding to resel.  >= 0.

    """
    # Check inputs
    check.string(sequences_path, 'sequences_path', TypeError)

    if not isinstance(valid_modes, set):
        raise TypeError('valid_modes must be a set')
    for m in valid_modes:
        check.string(m, str(m) + ' in valid_modes', TypeError)
        pass

    if not isinstance(valid_spam_lsam, set):
        raise TypeError('valid_spam_lsam must be a set')
    for m in valid_spam_lsam:
        check.string(m, str(m) + ' in valid_spam_lsam', TypeError)
        pass

    if not isinstance(valid_fpam, set):
        raise TypeError('valid_fpam must be a set')
    for m in valid_fpam:
        check.string(m, str(m) + ' in valid_fpam', TypeError)
        pass

    if not isinstance(valid_fsam, set):
        raise TypeError('valid_fsam must be a set')
    for m in valid_fsam:
        check.string(m, str(m) + ' in valid_fsam', TypeError)
        pass

    if not isinstance(validating_cfam, set):
        raise TypeError('valid_dpam must be a set')
    for m in validating_cfam:
        check.string(m, str(m) + ' in validating_cfam', TypeError)
        pass

    if not isinstance(valid_dpam, set):
        raise TypeError('valid_dpam must be a set')
    for m in valid_dpam:
        check.string(m, str(m) + ' in valid_dpam', TypeError)
        pass

    # Load sequences dictionary from file and validate
    try:
        with open(sequences_path) as stream:
            sequences_dict = yaml.safe_load(stream)
    except FileNotFoundError:
        raise IOError('sequences file not found')

    # make all sequences uppercase to avoid case-sensitivity issues
    sequences = {}
    for k, v in sequences_dict.items():
        sequences.update({k.upper(): v})
        pass
    # Do not check sequence names!  Any sequence with valid contents is valid.
    # This allows us to add new sequences without re-releasing software.

    # Unpack sequence subdictionaries from sequences dictionary and validate
    for _, sequence in sequences.items():
        # Check each sequence for missing keys, but not extra keys
        misskeys = set(valid_sequence_keys) - set(sequence.keys())
        if misskeys != set():
            raise KeyError('Missing keys in sequence: ' + str(misskeys))

        # Unpack values from sequence subdictionary and validate
        check.string(sequence['mode'], 'mode', TypeError)
        if sequence['mode'] not in valid_modes:
            raise TypeError('invalid mode')

        check.string(sequence['spam_lsam'], 'spam_lsam', TypeError)
        if sequence['spam_lsam'] not in valid_spam_lsam:
            raise TypeError('invalid spam_lsam')

        check.string(sequence['fpam'], 'fpam', TypeError)
        if sequence['fpam'] not in valid_fpam:
            raise TypeError('invalid fpam')

        check.string(sequence['fsam'], 'fsam', TypeError)
        if sequence['fsam'] not in valid_fsam:
            raise TypeError('invalid fsam')

        check.string(sequence['cfam'], 'cfam', TypeError)
        # Special case; read from grid header (in cgi_eetc.py call)
        if sequence['cfam'].upper() not in validating_cfam:
            raise TypeError('invalid cfam')

        check.string(sequence['dpam'], 'dpam', TypeError)
        if sequence['dpam'] not in valid_dpam:
            raise TypeError('invalid dpam')

        check.real_positive_scalar(
            sequence['peak_flux_ratio_pix'],
            'peak_flux_ratio_pix',
            TypeError,
        )

        if sequence['fraction'] is not None:
            check.real_positive_scalar(
                sequence['fraction'],
                'fraction',
                TypeError,
            )
            if sequence['fraction'] > 1:
                raise ValueError('fraction must be <= 1.')


        if sequence['num_pixels'] is not None:
            check.real_nonnegative_scalar(
                sequence['num_pixels'],
                'num_pixels',
                TypeError,
            )

    return sequences


def load_thpt_configs(thpt_configs_path,
                      thpt_coatings_path,
                      thpt_data_path,
                      thptcurves_dir_path):
    """
    Load thpt_configs file and throughput curve files stored in thptcurves
    directory. Return dictionaries containing coating configurations, coating
    throughput curves, and setting throughput curves.

    Note that the returned dictionaries are not exactly what is read out of the
    file. The file contains keys with values that are the paths relative to
    thptcurves_dir_path, including the filename, of the
    throughput curve files, while the returned dictionaries contain the same
    keys but replace the filenames with the actual throughput curve arrays from
    the throughput curve files.

    Parameters
    ----------
    thpt_configs_path : str
        Full path of thpt_configs yaml file.

    thpt_coatings_path : str
        Full path of thpt_coatings yaml file.

    thpt_data : str
        Full path of thpt_data yaml file.

    thptcurves_dir_path : str
        Full path of directory containing throughput curve files.

    Returns
    -------
    coating_configs : dict
        Dictionary containing coating configurations for each mode.

        Keys are modes, values are coating configuration subdictionaries.

        Each coating configuration subdictionary contains the number of
        coatings of each type.

        Keys are coating names, values are integers specifying number of that
        type of coating.

    coating_thptcurves : dict
        Dictionary containing throughput curves for each type of coating.

        Keys are coating names, values are tuples of the form (lams, thpts),
        where lams is a 1D array of wavelengths (angstroms) and thpts is a 1D
        array of throughputs corresponding to each wavelength.

    setting_thptcurves : dict
        Dictionary containing throughput curves for each element and its
        possible settings.

        Keys are element names, values are subdictionaries containing
        throughput curves for each possible setting.

        For these subdictionaries, keys are setting names, values are tuples of
        the form (lams, thpts), where lams is a 1D array of wavelengths
        (angstroms) and thpts is a 1D array of throughputs corresponding to
        each wavelength.

    """
    # Check inputs
    check.string(thpt_configs_path, 'thpt_configs_path', TypeError)
    check.string(thpt_coatings_path, 'thpt_coatings_path', TypeError)
    check.string(thpt_data_path, 'thpt_data_path', TypeError)
    check.string(thptcurves_dir_path, 'thptcurves_dir_path', TypeError)
    if not Path(thptcurves_dir_path).is_dir():
        raise IOError('thptcurves_dir directory not found')

    # Load dicts from file and validate as appropriate
    try:
        with open(thpt_configs_path, 'r') as stream:
            thpt_configs = yaml.safe_load(stream)
    except FileNotFoundError:
        raise IOError('thpt_configs file not found')
    # No validation of the list of surfaces--they can be anything as long as
    # each surface has a valid coating, which we will check later

    try:
        with open(thpt_coatings_path, 'r') as stream:
            thpt_coatings = yaml.safe_load(stream)
    except FileNotFoundError:
        raise IOError('thpt_coatings file not found')
    # No validation of the list of static-optic coatings-they can be anything
    # as long as each coating maps to a file, which we will check later

    try:
        with open(thpt_data_path, 'r') as stream:
            thpt_data = yaml.safe_load(stream)
    except FileNotFoundError:
        raise IOError('thpt_data file not found')
    # No validation of the throughput options listed per surface--as long as
    # the curves exist to support the provided options, it's valid.

    coating_thptcurves = thpt_coatings
    setting_thptcurves = thpt_data
    valid_coating_types = thpt_coatings.keys()

    for i in coating_thptcurves.keys():
        check.string(coating_thptcurves[i], str(coating_thptcurves[i]),
                     TypeError)
        fn_coating = Path(thptcurves_dir_path, coating_thptcurves[i])
        if not fn_coating.is_file():
            raise IOError(str(fn_coating) + ' is not a file')
        # Replace filenames in coating_thptcurves dictionary with loaded arrays
        coating_thptcurves[i] = _load_thptcurve(fn_coating)

    # Split dictionary into parts and validate and enumerate coating lists
    coating_configs = dict()
    for layout, optics in thpt_configs.items(): # optics is another dict
        count_num_coatings = {k: 0 for k in valid_coating_types}
        for v in optics.values():
            if v not in valid_coating_types:
                raise TypeError('Invalid coating type')
            count_num_coatings[v] += 1
            pass
        coating_configs.update({layout: count_num_coatings})
        pass

    # Want to check the following:
    # - the set of top-level values match the required thpt_data format
    # - the set of second-level values each map to a file in
    #  thptcurves_dir_path when combined with the element
    # - the file itself is a valid throughput curve (i.e. in [0, 1])
    # The final check is done as part of _interp_thptcurve().  Checking only
    # these things means that anyone who updates the thpt_data contents
    # along with right curve files can run it, without hardcoding any
    # expectations.
    #
    # The second-level keys do not have to map to anything now, but the
    # sequence file will later validate against them to verify that we actually
    # have a curve for every sequence in that file

    for element, element_dict in setting_thptcurves.items():
        if element not in valid_thpt_data_keys:
            raise TypeError('thpt_data file contains invalid top-level keys')
        for setting, filename in element_dict.items():
            check.string(filename, 'filename', TypeError)
            fn = Path(thptcurves_dir_path, filename)
            if not fn.is_file():
                raise IOError(str(fn) + ' is not a file')
            element_dict[setting] = _load_thptcurve(fn)
            pass
        pass

    return coating_configs, coating_thptcurves, setting_thptcurves


def _load_thptcurve(thptcurve_path):
    """
    Load a throughput curve from a file.


    Internal only; this is tied up in the details of the dict definition
    """
    # Read file
    try:
        lams, thpts = np.loadtxt(thptcurve_path, unpack=True)
    except FileNotFoundError:
        raise IOError('throughput file not found')

    # Check arrays
    check.oneD_array(lams, 'lams', TypeError)
    check.oneD_array(thpts, 'thpts', TypeError)
    if len(lams) != len(thpts):
        raise ValueError('lam and thpt arrays in optical element file must '
                         'be the same length')
    if (thpts > 1).any() or (thpts < 0).any():
        raise ValueError('thpt values must be >= 0 and <= 1')

    return lams, thpts


def load_excam_config(excam_config_path):
    """
    Load excam configuration file.

    Parameters
    ----------
    excam_config_path : str
        Full path of excam_config yaml file.

    Returns
    -------
    excam_config : dict
        Dictionary with excam configuration values.

        Keys are excam property names, values are floats or ints corresponding
        to the excam properties.

    """
    # Check inputs
    check.string(excam_config_path, 'excam_config_path', TypeError)

    # Load dictionary from file and validate
    try:
        with open(excam_config_path, 'r') as stream:
            excam_config = yaml.safe_load(stream)
    except FileNotFoundError:
        raise IOError('excam_config file not found')
    validate_dict_keys(excam_config, valid_excam_config_keys)

    # Unpack values from dictionary and validate
    darke, cic, rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em, Nmin, \
        Nmax, tmin, tmax, gmax, gconst, n, Nem, tol, delta_constr, \
        overhead, pc_ecount_max, T_factor = \
        _unpack_excam_config(excam_config)

    check.real_nonnegative_scalar(darke, 'darke', TypeError)
    check.real_nonnegative_scalar(cic, 'cic', TypeError)
    check.real_nonnegative_scalar(rn, 'rn', TypeError)
    check.real_nonnegative_scalar(X, 'X', TypeError)
    check.real_nonnegative_scalar(a, 'a', TypeError)
    check.positive_scalar_integer(Lij, 'Lij', TypeError)
    check.real_positive_scalar(alpha0, 'alpha0', TypeError)
    if alpha0 > 1:
        raise ValueError('alpha0 must be between 0 and 1 (fraction of total)')
    check.positive_scalar_integer(fwc, 'fwc', TypeError)
    check.real_positive_scalar(alpha1, 'alpha1', TypeError)
    if alpha1 > 1:
        raise ValueError('alpha1 must be between 0 and 1 (fraction of total)')
    check.positive_scalar_integer(fwc_em, 'fwc_em', TypeError)
    check.positive_scalar_integer(Nmin, 'Nmin', TypeError)
    check.positive_scalar_integer(Nmax, 'Nmax', TypeError)
    if Nmax < Nmin:
        raise ValueError('Nmax must be >= Nmin')
    check.real_nonnegative_scalar(tmin, 'tmin', TypeError)
    check.real_nonnegative_scalar(tmax, 'tmax', TypeError)
    if tmax < tmin:
        raise ValueError('tmax must be >= tmin')
    check.real_nonnegative_scalar(gmax, 'gmax', TypeError)
    if gmax < 1:
        raise ValueError('gmax must be >= 1')
    if gconst is not None:
        check.real_positive_scalar(gconst, 'gconst', TypeError)
        if gconst < 1:
            raise ValueError('gconst must be >= 1')
        if gconst > gmax:
            raise ValueError('gconst must be <= gmax')
    check.real_nonnegative_scalar(n, 'n', TypeError)
    check.positive_scalar_integer(Nem, 'Nem', TypeError)
    check.real_nonnegative_scalar(tol, "tol", TypeError)
    check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)
    check.real_nonnegative_scalar(overhead, 'overhead', TypeError)
    check.real_positive_scalar(pc_ecount_max, 'pc_ecount_max', TypeError)
    check.real_positive_scalar(T_factor, 'T_factor', TypeError)

    return excam_config


def _unpack_excam_config(excam_config):
    """
    Decompose a 'excam_config' dictionary into its components

    Internal only; this is tied up in the details of the dict definition
    """

    darke = excam_config['darke']
    cic = excam_config['cic']
    rn = excam_config['rn']
    X = excam_config['X']
    a = excam_config['a']
    Lij = excam_config['Lij']
    alpha0 = excam_config['alpha0']
    fwc = excam_config['fwc']
    alpha1 = excam_config['alpha1']
    fwc_em = excam_config['fwc_em']
    Nmin = excam_config['Nmin']
    Nmax = excam_config['Nmax']
    tmin = excam_config['tmin']
    tmax = excam_config['tmax']
    gmax = excam_config['gmax']
    gconst = excam_config['gconst']
    n = excam_config['n']
    Nem = excam_config['Nem']
    tol = excam_config['tol']
    delta_constr = excam_config['delta_constr']
    overhead = excam_config['overhead']
    pc_ecount_max = excam_config['pc_ecount_max']
    T_factor = excam_config['T_factor']

    return darke, cic, rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em, \
        Nmin, Nmax, tmin, tmax, gmax, gconst, n, Nem, tol, delta_constr, \
        overhead, pc_ecount_max, T_factor


def load_locam_config(locam_config_path):
    """
    Load LOCAM configuration file.

    Parameters
    ----------
    locam_config_path : str
        Full path of locam_config yaml file.

    Returns
    -------
    locam_config : dict
        Dictionary with LOCAM configuration values.

        Keys are LOCAM property names, values are floats or ints corresponding
        to the LOCAM properties.

    """
    # Check inputs
    check.string(locam_config_path, 'locam_config_path', TypeError)

    # Load dictionary from file and validate
    try:
        with open(locam_config_path, 'r') as stream:
            locam_config = yaml.safe_load(stream)
    except FileNotFoundError:
        raise IOError('locam_config file not found')
    validate_dict_keys(locam_config, valid_locam_config_keys)

    # Unpack values from dictionary and validate
    darke, cic, alpha0, fwc, alpha1, fwc_em, g_max_comm, g_max_age, \
        e_max_age, tframe, n = _unpack_locam_config(locam_config)

    check.real_nonnegative_scalar(darke, 'darke', TypeError)
    check.real_nonnegative_scalar(cic, 'cic', TypeError)
    check.real_positive_scalar(alpha0, 'alpha0', TypeError)
    if alpha0 > 1:
        raise ValueError('alpha0 must be between 0 and 1 (fraction of total)')
    check.positive_scalar_integer(fwc, 'fwc', TypeError)
    check.real_positive_scalar(alpha1, 'alpha1', TypeError)
    if alpha1 > 1:
        raise ValueError('alpha1 must be between 0 and 1 (fraction of total)')
    check.positive_scalar_integer(fwc_em, 'fwc_em', TypeError)
    check.real_positive_scalar(g_max_comm, 'g_max_comm', TypeError)
    if g_max_comm < 1:
        raise ValueError('g_max_comm must be >= 1')
    check.real_positive_scalar(g_max_age, 'g_max_age', TypeError)
    if g_max_age < 1:
        raise ValueError('g_max_age must be >= 1')
    check.positive_scalar_integer(e_max_age, 'e_max_age', TypeError)
    check.real_nonnegative_scalar(tframe, 'tframe', TypeError)
    check.real_nonnegative_scalar(n, 'n', TypeError)

    return locam_config


def _unpack_locam_config(locam_config):
    """
    Decompose a 'locam_config' dictionary into its components

    Internal only; this is tied up in the details of the dict definition
    """

    darke = locam_config['darke']
    cic = locam_config['cic']
    alpha0 = locam_config['alpha0']
    fwc = locam_config['fwc']
    alpha1 = locam_config['alpha1']
    fwc_em = locam_config['fwc_em']
    g_max_comm = locam_config['g_max_comm']
    g_max_age = locam_config['g_max_age']
    e_max_age = locam_config['e_max_age']
    tframe = locam_config['tframe']
    n = locam_config['n']

    return darke, cic, alpha0, fwc, alpha1, fwc_em, g_max_comm, g_max_age, \
        e_max_age, tframe, n


def load_flux_grid(flux_grid_path):
    """
    Load flux grid file.

    Flux grid is 3D and should follow the following format:

    Spectral type (row) X CFAM filter (column) X photometric filter
    (fits extension)

    and contain the following header keywords to describe dimensions and
    extensions:

    ['REFBAND']:   photometric filter at which fluxes were computed
    ['REFMAG']:    photometric magnitude at which fluxes were computed
    ['SPECTYPE']:  spectral type array at which fluxes were computed
    ['CFAMCOLS']:  CFAM filter column names in flux grid
    ['PHOTEXTS']:  photometric filter fits extensions

    Parameters
    ----------
    flux_grid_path : str
        Full path of flux_grid fits file.

    Returns
    -------
    flux_grid : dict
        Dictonary containing flux grid data.

        Keys are photometric filters, values are tuples containing the
        following values:

        flux_grid_vals : array_like
            Array of flux values as a function of stellar spectral type and
            CFAM filter.

        spts : array_like
            Array of stellar spectral types corresponding to
            flux_grid_vals rows.

        cfams : array_like
            Array of CFAM filter names corresponding to flux_grid_vals columns.

        ref_mag : float
            Magnitude used to generate flux_grid_vals.

    S Halverson - JPL - 29-Sep-2019

    """
    # Check inputs
    check.string(flux_grid_path, 'flux_grid_file', TypeError)

    # Load HDUList (Header Data Unit List) and validate
    try:
        with fits.open(flux_grid_path) as hdul_raw:
            hdul = copy.deepcopy(hdul_raw)
    except FileNotFoundError:
        raise IOError('flux_grid file not found')

    # Cycle through HDUs and validate
    flux_grid_list = []
    for i, hdu in enumerate(hdul):
        # Unpack header and data
        header = dict(hdu.header)
        flux_grid_vals = hdu.data

        # astropy will only add EXTEND keyword if there are extensions
        if len(hdul) == 1: # primary only
            header_primary_keys.pop('EXTEND')

        # Validate flux_grid
        # Catch any dimension errors here, since different dimensions will
        # change the header values
        check.twoD_array(flux_grid_vals, 'flux_grid', TypeError)

        # Validate header keys based on header type. Also get phot filter
        # string from the primary header
        if i == 0:
            # Primary header comes first
            validate_dict_keys(header,
                               header_primary_keys + header_custom_keys)
            phots_str = header['PHOTEXTS']
            check.string(phots_str, 'phots_str', TypeError)
            phots = phots_str.split(',')
            # phots do not need to be validated; header defines what is valid
        else:
            # All other headers are image headers
            validate_dict_keys(header, header_ext_keys + header_custom_keys)

        # Unpack header and validate
        cfams_str = header['CFAMCOLS']
        ref_mag = header['REFMAG']
        spt_str = header['SPECTYPE']

        check.string(cfams_str, 'cfams_str', TypeError)
        check.real_scalar(ref_mag, 'ref_mag', TypeError)
        check.string(spt_str, 'spt_str', TypeError)

        # no need to validate these; the header of the grids define what is
        # valid
        cfams = cfams_str.split(',')
        spts = spt_str.split(',')


        # Check that header info and flux_grid are consistent
        # Reconstruct effective temperature array from fits header
        # This is the basis vector for flux grid rows
        if flux_grid_vals.shape[0] != len(spts):
            raise ValueError('number of rows in flux grid must match number '
                             'of spectral types')
        # cfams is the basis vector for flux grid cols
        if flux_grid_vals.shape[1] != len(cfams):
            raise ValueError('number of cols in flux grid must match number '
                             'of cfam filters')

        # Pack data into tuple and append to list in order of phot filter
        flux_grid_list.append((flux_grid_vals, spts, cfams, ref_mag))

    # Make dictionary with keys from phots and values from flux_grid_list
    flux_grid = dict(zip(phots, flux_grid_list))

    return flux_grid


def load_wave_grid(wave_grid_path):
    """
    Load wavelength grid file (spectrally averaged wavelength)

    wave grid is 3D and should follow the following format:

    Spectral type (row) X CFAM filter (column) X photometric filter
    (fits extension)

    and contain the following header keywords to describe dimensions and
    extensions:

    ['REFBAND']:   photometric filter at which wavelengths were computed
    ['REFMAG']:    photometric magnitude at which wavelengths were computed
    ['SPECTYPE']:  spectral type array at which wavelengths were computed
    ['CFAMCOLS']:  CFAM filter column names in wave grid
    ['PHOTEXTS']:  photometric filter fits extensions

    Parameters
    ----------
    wave_grid_path : str
        Full path of wave_grid fits file.

    Returns
    -------
    wave_grid : dict
        Dictonary containing wave grid data.

        Keys are photometric filters, values are tuples containing the
        following values:

        wave_grid_vals : array_like
            Array of wave values as a function of stellar spectral type and
            CFAM filter.

        spts : array_like
            Array of stellar spectral types corresponding to
            wave_grid_vals rows.

        cfams : array_like
            Array of CFAM filter names corresponding to wave_grid_vals columns.

        ref_mag : float
            Magnitude used to generate wave_grid_vals.

    S Halverson - JPL - 29-Sep-2019

    """
    # Check inputs
    check.string(wave_grid_path, 'wave_grid_file', TypeError)

    # Load HDUList (Header Data Unit List) and validate
    try:
        with fits.open(wave_grid_path) as hdul_raw:
            hdul = copy.deepcopy(hdul_raw)
    except FileNotFoundError:
        raise IOError('wave_grid file not found')

    # Cycle through HDUs and validate
    wave_grid_list = []
    for i, hdu in enumerate(hdul):
        # Unpack header and data
        header = dict(hdu.header)
        wave_grid_vals = hdu.data

        # Validate wave_grid
        # Catch any dimension errors here, since different dimensions will
        # change the header values
        check.twoD_array(wave_grid_vals, 'wave_grid', TypeError)

        # Validate header keys based on header type. Also get phot filter
        # string from the primary header
        if i == 0:
            # Primary header comes first
            validate_dict_keys(header,
                               header_primary_keys + header_custom_keys)
            phots_str = header['PHOTEXTS']
            check.string(phots_str, 'phots_str', TypeError)
            phots = phots_str.split(',')
            # phots do not need to be validated; header defines what is valid
        else:
            # All other headers are image headers
            validate_dict_keys(header, header_ext_keys + header_custom_keys)

        # Unpack header and validate
        cfams_str = header['CFAMCOLS']
        ref_mag = header['REFMAG']
        spt_str = header['SPECTYPE']

        check.string(cfams_str, 'cfams_str', TypeError)
        check.real_scalar(ref_mag, 'ref_mag', TypeError)
        check.string(spt_str, 'spt_str', TypeError)

        # no need to validate these; the header of the grids define what is
        # valid
        cfams = cfams_str.split(',')
        spts = spt_str.split(',')

        # Check that header info and wave_grid are consistent
        # Reconstruct effective temperature array from fits header
        # This is the basis vector for wave grid rows
        if wave_grid_vals.shape[0] != len(spts):
            raise ValueError('number of rows in wave grid must match number '
                             'of spectral types')
        # cfams is the basis vector for wave grid cols
        if wave_grid_vals.shape[1] != len(cfams):
            raise ValueError('number of cols in wave grid must match number '
                             'of cfam filters')

        # Pack data into tuple and append to list in order of phot filter
        wave_grid_list.append((wave_grid_vals, spts, cfams, ref_mag))

    # Make dictionary with keys from phots and values from wave_grid_list
    wave_grid = dict(zip(phots, wave_grid_list))

    return wave_grid
