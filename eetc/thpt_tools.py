# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Functions for calculating throughput based on provided throughput curves and
optical element settings.
"""
from scipy.interpolate import interp1d


class ThptToolsException(Exception):
    """Exception class for thpt_tools module."""


def calc_thpt(coating_thptcurves, setting_thptcurves, coating_config, lam):
    """
    Calculate the estimated throughput of RST + CGI system for a provided
    wavelength.

    Note the total throughput estimate does not include the CFAM filter, which
    is included in the precomputed EETC flux rate grids.

    Also note that this is designed for internal use within CGIEETC and assumes
    all inputs are already validated. Therefore no input checks are done.

    Parameters
    ----------
    coating_thptcurves : dict
        Dictionary containing throughput curves for each type of coating.

        Keys are coating names, values are tuples of the form (lams, thpts),
        where lams is a 1D array of wavelengths (angstroms) and thpts is a 1D
        array of throughputs corresponding to each wavelength.

    setting_thptcurves : dict
        Dictionary containing throughput curves for each element.

        Keys are element names, values are tuples of the form (lams, thpts),
        where lams is a 1D array of wavelengths (angstroms) and thpts is a 1D
        array of throughputs corresponding to each wavelength.

    coating_config : dict
        Dictionary containing number of coatings of each type.

        Keys are coating names, values are integers specifying number of that
        type of coating.

    lam : float
        Wavelength to be used in throughput curve interpolations (angstroms).

    Returns
    -------
    thpt : float
        Estimated RST + CGI throughput.

    Notes
    -----
    Adapted from John Krist's cgisim_roman_throughput.py in cgisim

    S Halverson - JPL - 29-Sep-2019
    S Miller - UAH - 1-Dec-2021

    """

    # Interpolate to get throughput for each coating and evaluate at lambda
    thpt_lam = dict()
    for k, v in coating_thptcurves.items():
        thpt_lam[k] = _interp_thptcurve(v, lam)
        pass

    # Match number of coatings with coating throughput curves and calculate
    # total coating throughput
    thpt_coatings = 1 # start with perfect throughput
    for k, v in coating_config.items(): # v = number of surfaces of that type
        thpt_coatings *= (thpt_lam[k] ** v)
        pass
    # Quantum efficiency of the EMCCD detector is not technically a throughput,
    # but it is easiest to include it here and treat it like one.  Done by
    # adding it as a surface in the mode

    # Interpolate to get throughput for each element and evaluate at lambda
    thpt_pams = 1  # Start with perfect throughput
    for _, thptcurve in setting_thptcurves.items():
        thpt_pams *= _interp_thptcurve(thptcurve, lam)

    # Calculate total throughput
    thpt = thpt_coatings * thpt_pams

    return thpt


def _interp_thptcurve(thptcurve, lam):
    """
    Find throughput for a given lambda using linear interpolation.

    Convenience function for internal use within calc_thpt.

    Note that if lam is greater than the last lam provided, the throughput at
    the end of the array is used. Likewise if lam is less than the first lam
    provided, the throughput at the beginning of the array is used.

    Parameters
    ----------
    thptcurve : tuple
        Tuple of the form (lams, thpts), where lams is a 1D array of
        wavelengths (angstroms) and thpts is a 1D array of throughputs
        corresponding to each wavelength.

    lam : float
        Wavelength to be used in interpolation (angstroms).

    Returns
    -------
    thpt : float
        Throughput for given wavelength.

    """
    # Unpack throughput curve
    lams, thpts = thptcurve

    if (lam > lams.max()) or (lam < lams.min()):
        raise ThptToolsException('Requested wavelength falls outside the '
        'wavelength bounds of the throughput curve.  Check wavelengths '
        'for modes in yaml files.')

    # Interpolate throughtput curve over provided wavelength array
    f = interp1d(lams, thpts, kind='linear', bounds_error=False,
                 fill_value=(thpts[0], thpts[-1]), assume_sorted=False)
    thpt = float(f(lam))

    # These should not be possible for a valid throughput curve (i.e. a curve
    # that ranges from 0 to 1 inclusive), but still check in case something
    # weird happens inside the interpolator
    if thpt > 1:
        raise ThptToolsException('interpolation resulted in throughput > 1')
    if thpt < 0:
        raise ThptToolsException('interpolation resulted in throughput < 0')

    return thpt
