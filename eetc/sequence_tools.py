# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Miscellaneous tools to produce numbers to be used with sequences
"""

import numpy as np
from scipy.optimize import curve_fit

import eetc.util.check as check

def get_peak_flux_ratio_pix(array):
    """
    Return the ratio of peak flux in a pixel, to total flux in all pixels.
    This function assumes the array is from a frame that has no bias or
    is bias-subtracted.

    This is intended for use with PSFs and other sharply-featured images,
    though it can technically be run on any 2D array.

    Will ignore any NaNs in the array for the purposes of calculation.  Will
    raise a ValueError if the array is all NaNs or all zeros, though.

    Arguments:
     array: finite and real-valued 2D array to calculate on.  Must have at
     least some positive values present.
     If it is not partially positive, real-valued, 2D, or an array, a TypeError
     will be raised.

    Returns:
     floating point value between 0 and 1, inclusive of 1.  (It is not possible
     to get a zero output as all-zero data raises a ValueError.)

    """

    check.twoD_array(array, 'array', TypeError)
    if not np.isrealobj(array):
        raise TypeError('Array must be real-valued')
    if (array < 0).all():
        raise TypeError('Array must have some positive numbers')
    if np.isnan(array).all():
        raise ValueError('Array cannot be all NaNs')
    if (array == 0).all():
        raise ValueError('Array cannot be all zeroes')
    if (not np.isnan(array).any()) and (not np.isfinite(array).all()):
        raise TypeError('Array must be finite')
    if (array[~np.isnan(array)] == 0).all():
        raise ValueError('Array cannot be all zeroes in between the NaNs')

    return np.nanmax(array)/np.nansum(array)

def get_num_pixels_and_fraction(array, thresh=0.8):
    """
    Given an array with a PSF, this function finds the number of
    pixels represented by the spatial resolution element ("resel") for the PSF.
    It also finds the fraction of the total array's flux corresponding to the
    resel.  The array should have only one PSF present for reliable results.
    This function assumes the array is from a frame that has no bias or
    is bias-subtracted.

    The fit function for the PSF is a rotated elliptic Gaussian (different
    widths in general in x and y and also rotated to an arbitrary angle).  The
    fraction is found by summing the non-NaN pixels within the fitted
    Gaussian's 2-D FWHM and dividing by the non-NaN sum of the whole array.

    The function will ignore any NaNs in the array for the purposes of
    calculation and will raise a ValueError if the array is all NaNs or all
    zeros.

    Parameters
    ----------
    array : 2-D array
    The input array containing the PSF of interest. All elements must be
    real-valued and finite, and it must have at least some positive values.

    thresh : float
    Fit's adjusted R^2 value must be equal to or greater than this specified
    threshold.  Must be > 0 and <= 1.  Exception raised if thresh not met.
    Defaults to 0.8.

    Returns
    -------
    num_pixels : float
        Number of pixels corresponding to resel.  >= 0.

    fraction : float
        Fraction of total array area flux corresponding to resel.  >=0 and
        <=1.

    Rsq_adj : float
        Adjusted R^2 value from the fit.
    """

    check.twoD_array(array, 'array', TypeError)
    if not np.isrealobj(array):
        raise TypeError('Array must be real-valued')
    if (array < 0).all():
        raise TypeError('Array must have some positive numbers')
    if np.isnan(array).all():
        raise ValueError('Array cannot be all NaNs')
    if (array == 0).all():
        raise ValueError('Array cannot be all zeroes')
    if (not np.isnan(array).any()) and (not np.isfinite(array).all()):
        raise TypeError('Array must be finite')
    if (array[~np.isnan(array)] == 0).all():
        raise ValueError('Array cannot be all zeroes in between the NaNs')
    check.real_positive_scalar(thresh, 'thresh', TypeError)
    if thresh > 1:
        raise ValueError('thresh must be <= 1.')

    try:
        offset, A, x0, y0, sx, sy, theta, fit, Rsq_adj = \
            _rot_gauss_fit(array)
    except:
        raise Exception('Could not find a fit to the PSF.')

    if Rsq_adj < thresh:
        raise ValueError('The adjusted R^2 of the fit cannot be less than '
                         'thresh.')

    FWHMx = 2*np.sqrt(2*np.log(2))*sx
    FWHMy = 2*np.sqrt(2*np.log(2))*sy


    Y = np.arange(0,len(array))
    X = np.arange(0,len(array[0]))
    X, Y = np.meshgrid(X,Y)

    # pixels contained within elliptic FWHM
    rsX = np.cos(theta)*(X-x0)-np.sin(theta)*(Y-y0)
    rsY = np.sin(theta)*(X-x0)+np.cos(theta)*(Y-y0)
    ellipse = rsX**2/(FWHMx/2)**2 + rsY**2/(FWHMy/2)**2
    ell_ind = np.where(ellipse <= 1)
    pre_resel = array[ell_ind]
    resel = pre_resel[~np.isnan(pre_resel)]
    num_pixels = resel.size
    fraction = np.sum(resel)/np.nansum(array)

    return num_pixels, fraction, Rsq_adj

# Gaussian with different widths in x and y directions, and then rotated by
# theta
def _rot_gauss_spot(xy, offset, A, x0, y0, sx, sy, theta):
    (x, y) = xy
    return (offset +
        A*np.e**(-(np.cos(theta)*(x-x0)-np.sin(theta)*(y-y0))**2/(2*sx**2) -
                (np.sin(theta)*(x-x0)+np.cos(theta)*(y-y0))**2/(2*sy**2)))

def _rot_gauss_fit(data):
    y = np.arange(0,len(data))
    x = np.arange(0,len(data[0]))
    X, Y = np.meshgrid(x,y)
    # block out nan values before fitting
    good_ind = np.where(~np.isnan(data))
    # this flattens X and Y and data, too:
    X = X[good_ind]
    Y = Y[good_ind]
    XY = np.vstack((X, Y))
    data = data[good_ind]

    # initial guess values
    offset0 = np.median(data)
    A0 = np.max(data) - offset0
    peak_ind = np.argmax(data)
    x00 = X[peak_ind]
    y00 = Y[peak_ind]
    sx0 = 0.5
    sy0 = 0.5
    theta0 = 0
    init_guess = (offset0, A0, x00, y00, sx0, sy0, theta0)
    ub = [np.max(data), np.max(data), X.max(), Y.max(), X.max()/2,
          Y.max()/2, np.pi]
    lb = [0, 0, 0, 0, 0, 0, 0]
    bounds = (lb, ub)
    popt, _ = curve_fit(_rot_gauss_spot, XY, data.ravel(),
                            bounds=bounds, p0=init_guess, maxfev=1e5,
                            xtol=1e-15,)

    gauss_val = _rot_gauss_spot(XY, *popt)
    residual = data - gauss_val
    ss_r = np.sum((np.mean(data) - gauss_val)**2)
    ss_e = np.sum(residual**2)
    Rsq = ss_r/(ss_e+ss_r)
    # adjusted coefficient of determination, adjusted R^2:
    # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
    # The closer to 1, the better the fit.
    # Can have negative values. Can never be above 1.
    num_params = 7 # fit parameters
    data_pts = data.size
    Rsq_adj = 1 - (1 - Rsq)*(data_pts - 1)/(data_pts - num_params)

    return list(popt) + [gauss_val, Rsq_adj]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from emccd_detect.emccd_detect import EMCCDDetect
    from astropy.io import fits
    import os
    import eetc

    ar = np.zeros((50, 50))

    x = np.arange(50)
    y = np.arange(50)
    X, Y = np.meshgrid(x, y)
    A = 2000
    sx = 4
    sy = 2
    offset = 100
    ar += _rot_gauss_spot((X,Y), offset, A, 20, 29, sx, sy, np.pi/2)
    if True:
        bias = 2000 # in e-
        eperdn = 7 #e-/DN
        bias_dn = bias/eperdn
        em_gain = 3
        emccd = EMCCDDetect(em_gain=em_gain, full_well_image=50000,
                            full_well_serial=100000, dark_current=8.33e-4,
                            cic=0.02,read_noise=120, bias=bias, qe=0.9,
                            cr_rate=0,
                            pixel_pitch=3e-6, eperdn=7, nbits=12,
                            numel_gain_register=604, meta_path=None)
        ar_n = emccd.sim_sub_frame(ar, 1).astype(float)
        #ar = eperdn*(ar - bias_dn)/em_gain # ungained e-
        # bias subtracted in next line (and converted to units of photons)
        ar = emccd.get_e_frame(ar_n)/emccd.qe
        ar = ar.astype(float)
        ar[25:32, 23:24] = np.nan
        hdr = fits.Header()
        prim = fits.PrimaryHDU(header=hdr)
        img = fits.ImageHDU(ar)
        hdul = fits.HDUList([prim, img])
        here = os.path.abspath(os.path.dirname(__file__))
        hdul.writeto(os.path.join(here, 'noised_PSF.fits'),
                                  overwrite=True)

    print(get_num_pixels_and_fraction(ar, 0.65))