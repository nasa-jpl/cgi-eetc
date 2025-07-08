# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Package for computing CCD settings and exposure times for a given observing
scenario.

Constrains signal before gain to be n standard deviations
(accounting for ENF and read noise) below alpha0*fwc, and also constrains the
signal after gain to be n standard deviations (accounting for ENF and read
noise) below alpha1*fwc_em. Also uses fluxe for SNR constraint and fluxe_bright
for fwc constraints.

The number of darks taken
for the master dark is large compared to the maximum number of frames that can
be taken for an analog observation, and thus for analog SNR calculations in all
the functions below, the contribution to the noise from the darks is ignored.
The number of darks divided by the maximum number of photon-counted (PC)
frames that can be taken for an observation is not as big as in the analog
case; for the PC cases below, for a reasonable and conservative estiamte
without knowing the exact number of darks that will be taken,
it is assumed that the number of darks taken for
the master dark is the same as the number of frames taken for a PC observation.
"""
import argparse
import warnings

import numpy as np
import scipy.optimize
from mpmath import hyper

import eetc.util.check as check

# The SLSQP optimizer sometimes has known internal weirdness about bounds and
# scipy will raise a warning that we can't do anything about.  Filter it out.
warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')

class EXCAMOptimizeException(Exception):
    """Thin class for optimizer-specific failures"""
    pass


def _ENF(g, Nem):
    """
    Returns the ENF.
    """
    return np.sqrt(2*(g-1)*g**(-(Nem+1)/Nem) + 1/g)


def _SNR_CR_den(g, Nem, rn, fluxe, tfr, darke, cic):
    """
    Returns the denominator of the SNR given camera settings and noise
    properties, including cosmic ray effects (denominator of the
    _SNR_CR function).
    """
    return np.sqrt(rn**2/g**2 +
                   ((_ENF(g, Nem))**2)*(fluxe*tfr + darke*tfr + cic))


def _SNR_CR(g, tfr, N, fluxe, darke, cic, rn, X, a, Lij, sign, Nem):
    """
    Compute the SNR given camera settings and noise properties, including
    cosmic ray effects.

    Internal function to feed the optimizer only.  Does not enforce any of the
    bounds you might expect, as the optimizer will eventually enforce them, so
    use for non-optimizer applications at your own risk.

    Parameters
    ----------
    g : float
        camera gain, unitless.

    tfr : float
        exposure time per frame, in seconds.

    N : float
        number of frames, unitless.

    fluxe : float
        flux, in electrons per second (i.e. photon flux (phi) * QE (eta)).

    darke : float
        dark current, in electrons per second.

    cic : float
        clock-induced charge, in electrons.

    rn : float
        read noise, in electrons.

    X : float
        cosmic ray hits/meter**2/sec.

    a : float
        pixel area in meters**2/pixel.

    Lij : int
        for a target pixel, the number of pixels including itself which can
        cause that target pixel to be made useless if the other pixel is hit.

    sign : {1, -1}
        1 or -1, to return a positive or negative value (for minimization or
        maximization, respectively)

    Nem : int
        number of gain multiplying elements.

    Returns
    -------
    float
        SNR value (single floating-point number)

    """
    # no input checks for functions that will be run by the optimizer

    num = np.sqrt(N*np.exp(-X*a*Lij*tfr))*fluxe*tfr
    den = _SNR_CR_den(g, Nem, rn, fluxe, tfr, darke, cic)
    if den == 0:
        return 0 # in the limit den terms go to zero, num terms go faster

    return sign*num/den

def _SNR_CR_resel(g, tfr, N, fluxe, darke, cic, rn, X, a, Lij, sign, Nem,
                    num_pixels):
    """
    Computes the photon-counting SNR per spatial resolution element ('resel')
    comprised of 'num_pixels' pixels, given camera settings and noise
    properties, including cosmic ray effects.  Details can be found in
    snr_resel.pdf in the doc folder of eetc.

    Internal function to feed the optimizer only.  Does not enforce any of the
    bounds you might expect, as the optimizer will eventually enforce them, so
    use for non-optimizer applications at your own risk.

    Parameters
    ----------
    g : float
        camera gain, unitless.

    tfr : float
        exposure time per frame, in seconds.

    N : float
        number of frames, unitless.

    fluxe : float
        flux, in electrons per second (i.e. photon flux (phi) * QE (eta)).

    darke : float
        dark current, in electrons per second.

    cic : float
        clock-induced charge, in electrons.

    rn : float
        read noise, in electrons.

    X : float
        cosmic ray hits/meter**2/sec.

    a : float
        pixel area in meters**2/pixel.

    Lij : int
        for a target pixel, the number of pixels including itself which can
        cause that target pixel to be made useless if the other pixel is hit.

    sign : {1, -1}
        1 or -1, to return a positive or negative value (for minimization or
        maximization, respectively)

    Nem : int
        number of gain multiplying elements.

    num_pixels : float
        number of pixels considered.

    Returns
    -------
    float
        SNR value (single floating-point number)
    """
    if num_pixels == 1:
        return _SNR_CR(g, tfr, N, fluxe, darke, cic, rn, X, a, Lij, sign, Nem)
    else:
        return (_SNR_CR(g, tfr, N, fluxe, darke, cic, rn, X,
                (4*num_pixels/np.pi)**0.5*a, Lij, sign, Nem) * num_pixels**0.5)
    # return (_SNR_CR(g, tfr, N, fluxe, darke, cic, rn, X, a, Lij, sign, Nem) *
    #     num_pixels**0.5 * (np.exp(-X*tfr*(num_pixels**0.5 - 1)*a*Lij))**0.5)

def calc_gain_exptime(target_snr, fluxe, fluxe_bright, darke, cic, rn, X, a,
                      Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax, tmin, tmax,
                      gmax, overhead, opt_choice=0, n=4, Nem=604, tol=1e-30,
                      delta_constr=1e-4, num_pixels=1):
    """
    Run 1-2 optimizations to find the best EXCAM settings.
    If you don't have a target SNR and simply want to
    maximize the SNR, you may specify opt_choice=1 to skip to the 2nd
    optimizer.

    This function runs an optimization to find the combination of gain,
    exposure time, and number of frames which provides an SNR at or better than
    a target SNR with the smallest wall-clock time.  If there is no feasible
    combination which does so, it runs a second optimization to find the best
    SNR it can get given the constraints.

    Despite the long list of inputs, almost all of them are fixed properties of
    the EXCAM detector or the CGI system as a whole.  The only 3 that will
    usually be changed are the target SNR (target_snr), incoming flux
    (fluxe, in units of electrons), and the incoming peak flux (fluxe_bright,
    in units of electrons).  Both of these depend on the astrophysical target
    and the use case of CGI at that time.

    Inputs that violate the specifications given below will raise a TypeError
    or a ValueError.

    If both optimizations are somehow infeasible, or if compound constraints
    (e.g. electrons generation rate vs. full-well level) that depend on
    fluxe_bright are violated, an EXCAMOptimizeException will be raised.

    Parameters
    ----------
    target_snr : float
        Desired SNR in 'num_pixels' pixels, where 'num_pixels' is the input
        specifying the number of pixels.  This can be useful if considering a
        spatial resolution element ('resel'). >= 0.

    fluxe : float
        observation target's average flux, in electrons per second
        (i.e. photon flux (phi) * QE (eta)), in a pixel.  >= 0.

    fluxe_bright : float
        flux of the peak pixel, in electrons per second
        (i.e. photon flux (phi) * QE (eta)).  >= 0.

    darke : float
        dark current, in electrons per second.  >= 0.

    cic : float
        clock-induced charge, in electrons.  >= 0.

    rn : float
        read noise, in electrons.  >= 0.

    X : float
        cosmic ray hits/meter**2/sec.  >= 0.

    a : float
        pixel area in meters**2/pixel.  >= 0.

    Lij : int
        for a target pixel, the number of pixels including itself which can
        cause that target pixel to be made useless if hit with a cosmic ray.
        >= 1.

    alpha0 : float
        fraction of the per-pixel full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc : int
        number of electrons to fill the per-pixel full well.  >= 1.

    alpha1 : float
        fraction of the EM gain full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc_em : int
        number of electrons to fill the per-pixel EM gain full well.  >= 1.

    Nmin : int
        minimum number of allowed exposures.  >= 1.

    Nmax : int
        maximum number of allowed exposures.  >= 1.  Must be > Nmin.

    tmin : float
        minimum exposure time length, in seconds/frame.  >= 0.

    tmax : float
        maximum exposure time length, in seconds/frame.  >= 0. Must be > tmin.

    gmax : float
        maximum permitted gain.  > 1.  (Minimum gain is 1.)

    overhead : float
        Overhead per frame, in seconds, that is not spent observing.  Used to
        compute wall-clock time in optimization figure of merit.  >= 0.

    opt_choice : {0, 1}, optional
        0 to try the first optimization first and then the second if the first
        fails, 1 to go directly to the second optimization (i.e., to maximize
        the SNR without trying to minimize the total integrated exposure time).
        Defaults to 0.

    n : float, optional
        number of standard deviations the signal after gain is below the max
        fwc_em. Defaults to 4.

    Nem : int, optional
        number of gain multiplying elements.  Defaults to 604, which is the
        right hardware number for CGI cameras and very unlikely to change.

    tol : float, optional
        tolerance level used used by optimizations.  >= 0.  Recommended to be
        1e-30 for good results for any given input.  (This has been tested and
        verified over the relevant parameter space.)  Defaults to 1e-30.

    delta_constr:  float, optional
        constraint bounds relaxed via delta_constr so that optimization has
        success if constraints are satisfied within this fraction (to
        accommodate floating-point error).  >=0. For example,
        delta_constr = 0.01 means that the SNR that results from
        the optimizer should be 0.99*target_snr or bigger in order to have a
        successful return for the optimizer. Defaults to 1e-4,
        which has been tested and confirmed as a good choice for
        consistency between the first and second optimization schemes.

    num_pixels : float, optional
        The number of pixels over which to calculate the SNR.  >= 0.  Defaults
        to 1.

    Returns
    -------
    gain : float
        Optimal gain setting for detector

    exptime : float
        Exposure time for each frame in 'n_frames' to reach 'snr_out'

    n_frames : float
        Total number of exposures to reach 'snr_out'

    snr_out : float
        Expected SNR when using above exposure settings

    optflag : int
        0 or 1: 0 if the first optimization succeeded, 1 if the first failed
        but the second succeeded.

    """

    # Check inputs
    check.real_nonnegative_scalar(target_snr, 'target_snr', TypeError)
    check.real_nonnegative_scalar(fluxe, 'fluxe', TypeError)
    check.real_nonnegative_scalar(fluxe_bright, 'fluxe_bright', TypeError)
    if fluxe_bright < fluxe:
        raise ValueError('fluxe_bright must be greater than or equal to fluxe')
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
    if Nmax <= Nmin:
        raise ValueError('Nmax must be > Nmin')
    check.real_nonnegative_scalar(tmin, 'tmin', TypeError)
    check.real_positive_scalar(tmax, 'tmax', TypeError)
    if tmax <= tmin:
        raise ValueError('tmax must be > tmin')
    check.real_positive_scalar(gmax, 'gmax', TypeError)
    if gmax <= 1:
        raise ValueError('gmax must be > 1')
    check.real_nonnegative_scalar(overhead, 'overhead', TypeError)
    check.nonnegative_scalar_integer(opt_choice, 'opt_choice', TypeError)
    if ((opt_choice != 0) and (opt_choice != 1)):
        raise ValueError('opt_choice must be 0 or 1')
    check.real_nonnegative_scalar(n, 'n', TypeError)
    check.positive_scalar_integer(Nem, 'Nem', TypeError)
    check.real_nonnegative_scalar(tol, "tol", TypeError)
    check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)
    check.real_nonnegative_scalar(num_pixels, 'num_pixels', TypeError)

    # Check that there are no unsatisfiable constraints
    # minimal case for rail constraint (parallel CIC negligible):
    if (fluxe_bright*tmin + darke*tmin) + \
        n*np.sqrt(fluxe_bright*tmin + darke*tmin) > alpha0*fwc:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')

    _gmin = 1

    # in case alpha1*fwc_em is input as smaller than alpha0*fwc:
    #minimal case for em_rail constraint, serial cic ~ total cic:
    if _gmin*(fluxe_bright*tmin + darke*tmin + cic) + \
        n*_ENF(_gmin, Nem)*_gmin*\
        np.sqrt(fluxe_bright*tmin + darke*tmin + cic) > alpha1*fwc_em:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')

    # [g, tfr, N] = v
    bounds = scipy.optimize.Bounds(lb=np.array([_gmin, tmin, Nmin]),
                                ub=np.array([gmax, tmax, Nmax]))

    em_rail = lambda v: v[0]*(fluxe_bright*v[1] + darke*v[1] + cic) + (
                        n*_ENF(v[0], Nem)*v[0]*np.sqrt(fluxe_bright*v[1] +
                        darke*v[1] + cic)
                        )
    # constraint 1:  prevention of EM FWC saturation
    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1*fwc_em)

    nowsnr = lambda v: _SNR_CR_resel(v[0], v[1], v[2], fluxe, darke, cic, rn,
                        X, a, Lij, sign=1, Nem=Nem, num_pixels=num_pixels)
    # constraint 2:  target SNR
    nconst2 = scipy.optimize.NonlinearConstraint(nowsnr,
                                        target_snr, np.inf)

    rail = lambda v: fluxe_bright*v[1] + darke*v[1] + (
                    n*np.sqrt(fluxe_bright*v[1] + darke*v[1])
                    )
    # constraint 3:  prevention of image FWC saturation
    nconst3 = scipy.optimize.NonlinearConstraint(rail, 0, alpha0*fwc)

    _tmp_opt_choice = 0 #used to go to 2nd optimization if first fails


    if opt_choice == 0:
        FOM = lambda v: (v[1] + overhead)*v[2] # no gain in wall-clock time

        res1 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([_gmin, tmin, Nmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3),
                                    )

        #same thing, but starting point of gmax
        res2 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([gmax, tmin, Nmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3),
                                    )

        g1 = res1.x[0]
        tfr1 = res1.x[1]
        N1 = int(np.ceil(res1.x[2]))
        g2 = res2.x[0]
        tfr2 = res2.x[1]
        N2 = int(np.ceil(res2.x[2]))
        snr1 = nowsnr(np.array([g1, tfr1, N1]))
        snr2 = nowsnr(np.array([g2, tfr2, N2]))

        #constraints met
        res1_cond = (
        (rail(np.array([g1, tfr1])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g1, tfr1])) <= (1+delta_constr)*(alpha1*fwc_em)) and
        (snr1 >= (1-delta_constr)*target_snr)
                    )

        res2_cond = (
        (rail(np.array([g2, tfr2])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g2, tfr2])) <= (1+delta_constr)*(alpha1*fwc_em)) and
        (snr2 >= (1-delta_constr)*target_snr)
                    )

        if res1_cond and res2_cond:
            if N1*tfr1 <= N2*tfr2:
                return (g1, tfr1, N1, snr1, 0)
            if N1*tfr1 > N2*tfr2:
                return (g2, tfr2, N2, snr2, 0)
        elif res1_cond and not res2_cond:
            return (g1, tfr1, N1, snr1, 0)
        elif res2_cond and not res1_cond:
            return (g2, tfr2, N2, snr2, 0)

        _tmp_opt_choice = 1

    if (opt_choice == 1) or (_tmp_opt_choice == 1):

        def _SNR_CR1(v, fluxe, darke, cic, rn, X, a, Lij, sign, Nem,
                    num_pixels):
            g, tfr, N = v
            return _SNR_CR_resel(g, tfr, N, fluxe, darke, cic, rn, X, a, Lij,
                                sign, Nem, num_pixels)

        res3 = scipy.optimize.minimize(fun=_SNR_CR1,
                                   x0=np.array([_gmin, tmin, Nmin]),
                                   args=(fluxe, darke, cic, rn, X, a, Lij, -1,
                                    Nem, num_pixels),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3),
                                   )


        #same thing, but starting point of gmax
        res4 = scipy.optimize.minimize(fun=_SNR_CR1,
                                   x0=np.array([gmax, tmin, Nmin]),
                                   args=(fluxe, darke, cic, rn, X, a, Lij, -1,
                                    Nem, num_pixels),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3),
                                   )

        g3 = res3.x[0]
        tfr3 = res3.x[1]
        N3 = int(np.ceil(res3.x[2]))
        g4 = res4.x[0]
        tfr4 = res4.x[1]
        N4 = int(np.ceil(res4.x[2]))
        snr3 = nowsnr(np.array([g3, tfr3, N3]))
        snr4 = nowsnr(np.array([g4, tfr4, N4]))

        #constraints met
        res3_cond = (
        (rail(np.array([g3, tfr3])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g3, tfr3])) <= (1+delta_constr)*(alpha1*fwc_em))
                    )

        res4_cond = (
        (rail(np.array([g4, tfr4])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g4, tfr4])) <= (1+delta_constr)*(alpha1*fwc_em))
                    )

        if res3_cond and res4_cond:
            if snr3 >= snr4:
                return (g3, tfr3, N3, snr3, 1)
            if snr3 < snr4:
                return (g4, tfr4, N4, snr4, 1)
        elif res3_cond and not res4_cond:
            return (g3, tfr3, N3, snr3, 1)
        elif res4_cond and not res3_cond:
            return (g4, tfr4, N4, snr4, 1)

    raise EXCAMOptimizeException('Both optimizations failed, cannot produce ' +
                                 'camera settings')

def calc_gain_fixed_time(target_snr, fluxe, fluxe_bright, darke, cic, rn, X, a,
                      Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax, t,
                      gmax, overhead, opt_choice=0, n=4, Nem=604, tol=1e-30,
                      delta_constr=1e-4, num_pixels=1):
    """
    Runs 1-2 optimizations to find the best EXCAM settings.
    If you don't have a target SNR and simply want to maximize the SNR,
    you may specify opt_choice=1 to skip to the 2nd optimizer.

    For a given frame exposure time (t), this function runs an optimization
    to find the combination of gain and number of frames which provides an SNR
    at or better than a target SNR with the smallest wall-clock time.  If there
    is no feasible combination which does so, it runs a second optimization to
    find the best SNR it can get given the constraints.

    Despite the long list of inputs, almost all of them are fixed properties of
    the EXCAM detector or the CGI system as a whole.  The only 3 that will
    usually be changed are the target SNR (target_snr), incoming flux
    (fluxe, in units of electrons), and the incoming peak flux (fluxe_bright,
    in units of electrons).  Both of these depend on the astrophysical target
    and the use case of CGI at that time.

    Inputs that violate the specifications given below will raise a TypeError
    or a ValueError.

    If both optimizations are somehow infeasible, or if compound constraints
    (e.g. electrons generation rate vs. full-well level) that depend on
    fluxe_bright are violated, an EXCAMOptimizeException will be raised.

    Parameters
    ----------
    target_snr : float
        Desired SNR in 'num_pixels' pixels, where 'num_pixels' is the input
        specifying the number of pixels.  This can be useful if considering a
        spatial resolution element ('resel'). >= 0.

    fluxe : float
        observation target's average flux, in electrons per second
        (i.e. photon flux (phi) * QE (eta)), in a pixel.  >= 0.

    fluxe_bright : float
        flux of the peak pixel, in electrons per second
        (i.e. photon flux (phi) * QE (eta)).  >= 0.

    darke : float
        dark current, in electrons per second.  >= 0.

    cic : float
        clock-induced charge, in electrons.  >= 0.

    rn : float
        read noise, in electrons.  >= 0.

    X : float
        cosmic ray hits/meter**2/sec.  >= 0.

    a : float
        pixel area in meters**2/pixel.  >= 0.

    Lij : int
        for a target pixel, the number of pixels including itself which can
        cause that target pixel to be made useless if hit with a cosmic ray.
        >= 1.

    alpha0 : float
        fraction of the per-pixel full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc : int
        number of electrons to fill the per-pixel full well.  >= 1.

    alpha1 : float
        fraction of the EM gain full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc_em : int
        number of electrons to fill the per-pixel EM gain full well.  >= 1.

    Nmin : int
        minimum number of allowed exposures.  >= 1.

    Nmax : int
        maximum number of allowed exposures.  >= 1.  Must be > Nmin.

    t : float
        fixed time length for all frames, in seconds/frame.  >= 0.

    gmax : float
        maximum permitted gain.  > 1.  (Minimum gain is 1.)

    overhead : float
        Overhead per frame, in seconds, that is not spent observing.  Used to
        compute wall-clock time in optimization figure of merit.  >= 0.

    opt_choice : {0, 1}, optional
        0 to try the first optimization first and then the second if the first
        fails, 1 to go directly to the second optimization (i.e., to maximize
        the SNR without trying to minimize the total integrated exposure time).
        Defaults to 0.

    n : float, optional
        number of standard deviations the signal after gain is below the max
        fwc_em. Defaults to 4.

    Nem : int, optional
        number of gain multiplying elements.  Defaults to 604, which is the
        right hardware number for CGI cameras and very unlikely to change.

    tol : float, optional
        tolerance level used used by optimizations.  >= 0.  Recommended to be
        1e-30 for good results for any given input.  (This has been tested and
        verified over the relevant parameter space.)  Defaults to 1e-30.

    delta_constr:  float, optional
        constraint bounds relaxed via delta_constr so that optimization has
        success if constraints are satisfied within this fraction (to
        accommodate floating-point error).  >=0. For example,
        delta_constr = 0.01 means that the SNR that results from
        the optimizer should be 0.99*target_snr or bigger in order to have a
        successful return for the optimizer. Defaults to 1e-4,
        which has been tested and confirmed as a good choice for
        consistency between the first and second optimization schemes.

    num_pixels : float, optional
        The number of pixels over which to calculate the SNR.  >= 0.  Defaults
        to 1.

    Returns
    -------
    gain : float
        Optimal gain setting for detector

    exptime : float
        Exposure time for each frame in 'n_frames' to reach 'snr_out'
        (same as t)

    n_frames : float
        Total number of exposures to reach 'snr_out'

    snr_out : float
        Expected SNR when using above exposure settings

    optflag : int
        0 or 1: 0 if the first optimization succeeded, 1 if the first failed
        but the second succeeded.

    """

    # Check inputs
    check.real_nonnegative_scalar(target_snr, 'target_snr', TypeError)
    check.real_nonnegative_scalar(fluxe, 'fluxe', TypeError)
    check.real_nonnegative_scalar(fluxe_bright, 'fluxe_bright', TypeError)
    if fluxe_bright < fluxe:
        raise ValueError('fluxe_bright must be greater than or equal to fluxe')
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
    if Nmax <= Nmin:
        raise ValueError('Nmax must be > Nmin')
    check.real_positive_scalar(t, 't', TypeError)
    check.real_positive_scalar(gmax, 'gmax', TypeError)
    if gmax <= 1:
        raise ValueError('gmax must be > 1')
    check.real_nonnegative_scalar(overhead, 'overhead', TypeError)
    check.nonnegative_scalar_integer(opt_choice, 'opt_choice', TypeError)
    if ((opt_choice != 0) and (opt_choice != 1)):
        raise ValueError('opt_choice must be 0 or 1')
    check.real_nonnegative_scalar(n, 'n', TypeError)
    check.positive_scalar_integer(Nem, 'Nem', TypeError)
    check.real_nonnegative_scalar(tol, "tol", TypeError)
    check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)
    check.real_nonnegative_scalar(num_pixels, 'num_pixels', TypeError)

    # Check that there are no unsatisfiable constraints
    # minimal case for rail constraint (parallel CIC negligible):
    if (fluxe_bright*t + darke*t) + \
        n*np.sqrt(fluxe_bright*t + darke*t) > alpha0*fwc:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')
    _gmin = 1

    # in case alpha1*fwc_em is input as smaller than alpha0*fwc:
    #minimal case for em_rail constraint, serial cic ~ total cic:
    if _gmin*(fluxe_bright*t + darke*t + cic) + n*_ENF(_gmin, Nem)*_gmin*\
        np.sqrt(fluxe_bright*t + darke*t + cic) > alpha1*fwc_em:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')

    # [g, N] = v
    # no need for "rail" constraint: it doesn't constrain at all for fixed t
    bounds = scipy.optimize.Bounds(lb=np.array([_gmin, Nmin]),
                                ub=np.array([gmax, Nmax]))

    em_rail = lambda v: v[0]*(fluxe_bright*t + darke*t + cic) + (
                        n*_ENF(v[0], Nem)*v[0]*np.sqrt(fluxe_bright*t +
                        darke*t + cic)
                        )
    # constraint 1:  prevention of EM FWC saturation
    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1*fwc_em)

    nowsnr = lambda v: _SNR_CR_resel(v[0], t, v[1], fluxe, darke, cic, rn, X,
                        a, Lij, sign=1, Nem=Nem, num_pixels=num_pixels)
    # constraint 2:  SNR target
    nconst2 = scipy.optimize.NonlinearConstraint(nowsnr, target_snr, np.inf)

    _tmp_opt_choice = 0 #used to go to 2nd optimization if first fails

    if opt_choice == 0:
        FOM = lambda v: (t + overhead)*v[1] # no gain in wall-clock time

        res1 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([_gmin, Nmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2),
                                    )

        #same thing, but starting point of gmax
        res2 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([gmax, Nmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2),
                                    )

        g1 = res1.x[0]
        N1 = int(np.ceil(res1.x[1]))
        g2 = res2.x[0]
        N2 = int(np.ceil(res2.x[1]))
        snr1 = nowsnr(np.array([g1, N1]))
        snr2 = nowsnr(np.array([g2, N2]))

        #constraints met
        res1_cond = (
        (em_rail(np.array([g1])) <= (1+delta_constr)*(alpha1*fwc_em)) and
        (snr1 >= (1-delta_constr)*target_snr)
                    )

        res2_cond = (
        (em_rail(np.array([g2])) <= (1+delta_constr)*(alpha1*fwc_em)) and
        (snr2 >= (1-delta_constr)*target_snr)
                    )

        if res1_cond and res2_cond:
            if N1*t <= N2*t:
                return (g1, t, N1, snr1, 0)
            if N1*t > N2*t:
                return (g2, t, N2, snr2, 0)
        elif res1_cond and not res2_cond:
            return (g1, t, N1, snr1, 0)
        elif res2_cond and not res1_cond:
            return (g2, t, N2, snr2, 0)

        _tmp_opt_choice = 1

    if (opt_choice == 1) or (_tmp_opt_choice == 1):

        def _SNR_CR2(v, t, fluxe, darke, cic, rn, X, a, Lij, sign, Nem,
                    num_pixels):
            g, N = v
            return _SNR_CR_resel(g, t, N, fluxe, darke, cic, rn, X, a, Lij,
                                sign, Nem, num_pixels)

        res3 = scipy.optimize.minimize(fun=_SNR_CR2,
                                   x0=np.array([_gmin, Nmin]),
                                   args=(t, fluxe, darke, cic, rn, X, a, Lij,
                                        -1, Nem, num_pixels),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1),
                                   )


        #same thing, but starting point of gmax
        res4 = scipy.optimize.minimize(fun=_SNR_CR2,
                                   x0=np.array([gmax, Nmin]),
                                   args=(t, fluxe, darke, cic, rn, X, a, Lij,
                                        -1, Nem, num_pixels),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1),
                                   )

        g3 = res3.x[0]
        N3 = int(np.ceil(res3.x[1]))
        g4 = res4.x[0]
        N4 = int(np.ceil(res4.x[1]))
        snr3 = nowsnr(np.array([g3, N3]))
        snr4 = nowsnr(np.array([g4, N4]))

        #constraints met
        res3_cond = (
        (em_rail(np.array([g3])) <= (1+delta_constr)*(alpha1*fwc_em))
                    )

        res4_cond = (
        (em_rail(np.array([g4])) <= (1+delta_constr)*(alpha1*fwc_em))
                    )

        if res3_cond and res4_cond:
            if snr3 >= snr4:
                return (g3, t, N3, snr3, 1)
            if snr3 < snr4:
                return (g4, t, N4, snr4, 1)
        elif res3_cond and not res4_cond:
            return (g3, t, N3, snr3, 1)
        elif res4_cond and not res3_cond:
            return (g4, t, N4, snr4, 1)

    raise EXCAMOptimizeException('Both optimizations failed, cannot produce ' +
                                 'camera settings')

def calc_gain_fixed_N(target_snr, fluxe, fluxe_bright, darke, cic, rn, X, a,
                      Lij, alpha0, fwc, alpha1, fwc_em, N, tmin, tmax,
                      gmax, overhead, opt_choice=0, n=4, Nem=604, tol=1e-30,
                      delta_constr=1e-4, num_pixels=1):
    """
    Run 1-2 optimizations to find the best EXCAM settings.
    If you don't have a target SNR and simply want to
    maximize the SNR, you may specify opt_choice=1 to skip to the 2nd
    optimizer.

    For a given number of frames (N), this function runs an optimization to
    find the combination of gain
    and exposure time which provides an SNR at or better than
    a target SNR with the smallest wall-clock time.  If there is no feasible
    combination which does so, it runs a second optimization to find the best
    SNR it can get given the constraints.

    Despite the long list of inputs, almost all of them are fixed properties of
    the EXCAM detector or the CGI system as a whole.  The only 3 that will
    usually be changed are the target SNR (target_snr), incoming flux
    (fluxe, in units of electrons), and the incoming peak flux (fluxe_bright,
    in units of electrons).  Both of these depend on the astrophysical target
    and the use case of CGI at that time.

    Inputs that violate the specifications given below will raise a TypeError
    or a ValueError.

    If both optimizations are somehow infeasible, or if compound constraints
    (e.g. electrons generation rate vs. full-well level) that depend on
    fluxe_bright are violated, an EXCAMOptimizeException will be raised.

    Parameters
    ----------
    target_snr : float
        Desired SNR in 'num_pixels' pixels, where 'num_pixels' is the input
        specifying the number of pixels.  This can be useful if considering a
        spatial resolution element ('resel'). >= 0.

    fluxe : float
        observation target's average flux, in electrons per second
        (i.e. photon flux (phi) * QE (eta)), in a pixel.  >= 0.

    fluxe_bright : float
        flux of the peak pixel, in electrons per second
        (i.e. photon flux (phi) * QE (eta)).  >= 0.

    darke : float
        dark current, in electrons per second.  >= 0.

    cic : float
        clock-induced charge, in electrons.  >= 0.

    rn : float
        read noise, in electrons.  >= 0.

    X : float
        cosmic ray hits/meter**2/sec.  >= 0.

    a : float
        pixel area in meters**2/pixel.  >= 0.

    Lij : int
        for a target pixel, the number of pixels including itself which can
        cause that target pixel to be made useless if hit with a cosmic ray.
        >= 1.

    alpha0 : float
        fraction of the per-pixel full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc : int
        number of electrons to fill the per-pixel full well.  >= 1.

    alpha1 : float
        fraction of the EM gain full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc_em : int
        number of electrons to fill the per-pixel EM gain full well.  >= 1.

    N : int
        the fixed number of frames.  >= 1.

    tmin : float
        minimum exposure time length, in seconds/frame.  >= 0.

    tmax : float
        maximum exposure time length, in seconds/frame.  >= 0. Must be > tmin.

    gmax : float
        maximum permitted gain.  > 1.  (Minimum gain is 1.)

    overhead : float
        Overhead per frame, in seconds, that is not spent observing.  Used to
        compute wall-clock time in optimization figure of merit.  >= 0.

    opt_choice : {0, 1}, optional
        0 to try the first optimization first and then the second if the first
        fails, 1 to go directly to the second optimization (i.e., to maximize
        the SNR without trying to minimize the total integrated exposure time).
        Defaults to 0.

    n : float, optional
        number of standard deviations the signal after gain is below the max
        fwc_em. Defaults to 4.

    Nem : int, optional
        number of gain multiplying elements.  Defaults to 604, which is the
        right hardware number for CGI cameras and very unlikely to change.

    tol : float, optional
        tolerance level used used by optimizations.  >= 0.  Recommended to be
        1e-30 for good results for any given input.  (This has been tested and
        verified over the relevant parameter space.)  Defaults to 1e-30.

    delta_constr:  float, optional
        constraint bounds relaxed via delta_constr so that optimization has
        success if constraints are satisfied within this fraction (to
        accommodate floating-point error).  >=0. For example,
        delta_constr = 0.01 means that the SNR that results from
        the optimizer should be 0.99*target_snr or bigger in order to have a
        successful return for the optimizer. Defaults to 1e-4,
        which has been tested and confirmed as a good choice for
        consistency between the first and second optimization schemes.

    num_pixels : float, optional
        The number of pixels over which to calculate the SNR.  >= 0.  Defaults
        to 1.

    Returns
    -------
    gain : float
        Optimal gain setting for detector

    exptime : float
        Exposure time for each frame in 'n_frames' to reach 'snr_out'

    n_frames : float
        Total number of exposures to reach 'snr_out'.  (Same as N.)

    snr_out : float
        Expected SNR when using above exposure settings

    optflag : int
        0 or 1: 0 if the first optimization succeeded, 1 if the first failed
        but the second succeeded.

    """

    # Check inputs
    check.real_nonnegative_scalar(target_snr, 'target_snr', TypeError)
    check.real_nonnegative_scalar(fluxe, 'fluxe', TypeError)
    check.real_nonnegative_scalar(fluxe_bright, 'fluxe_bright', TypeError)
    if fluxe_bright < fluxe:
        raise ValueError('fluxe_bright must be greater than or equal to fluxe')
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
    check.positive_scalar_integer(N, 'N', TypeError)
    check.real_nonnegative_scalar(tmin, 'tmin', TypeError)
    check.real_positive_scalar(tmax, 'tmax', TypeError)
    if tmax <= tmin:
        raise ValueError('tmax must be > tmin')
    check.real_positive_scalar(gmax, 'gmax', TypeError)
    if gmax <= 1:
        raise ValueError('gmax must be > 1')
    check.real_nonnegative_scalar(overhead, 'overhead', TypeError)
    check.nonnegative_scalar_integer(opt_choice, 'opt_choice', TypeError)
    if ((opt_choice != 0) and (opt_choice != 1)):
        raise ValueError('opt_choice must be 0 or 1')
    check.real_nonnegative_scalar(n, 'n', TypeError)
    check.positive_scalar_integer(Nem, 'Nem', TypeError)
    check.real_nonnegative_scalar(tol, "tol", TypeError)
    check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)
    check.real_nonnegative_scalar(num_pixels, 'num_pixels', TypeError)

    # Check that there are no unsatisfiable constraints
    # minimal case for rail constraint (parallel CIC negligible):
    if (fluxe_bright*tmin + darke*tmin) + \
        n*np.sqrt(fluxe_bright*tmin + darke*tmin) > alpha0*fwc:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')

    _gmin = 1

    # in case alpha1*fwc_em is input as smaller than alpha0*fwc:
    #minimal case for em_rail constraint, serial cic ~ total cic:
    if _gmin*(fluxe_bright*tmin + darke*tmin + cic) + \
        n*_ENF(_gmin, Nem)*_gmin*\
        np.sqrt(fluxe_bright*tmin + darke*tmin + cic) > alpha1*fwc_em:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')

    # [g, tfr] = v
    bounds = scipy.optimize.Bounds(lb=np.array([_gmin, tmin]),
                                ub=np.array([gmax, tmax]))

    em_rail = lambda v: v[0]*(fluxe_bright*v[1] + darke*v[1] + cic) + (
                        n*_ENF(v[0], Nem)*v[0]*np.sqrt(fluxe_bright*v[1] +
                        darke*v[1] + cic)
                        )
    # constraint 1:  prevention of EM FWC saturation
    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1*fwc_em)

    nowsnr = lambda v: _SNR_CR_resel(v[0], v[1], N, fluxe, darke, cic, rn, X,
                        a, Lij, sign=1, Nem=Nem, num_pixels=num_pixels)
    # constraint 2:  target SNR
    nconst2 = scipy.optimize.NonlinearConstraint(nowsnr,
                                        target_snr, np.inf)

    rail = lambda v: fluxe_bright*v[1] + darke*v[1] + (
                    n*np.sqrt(fluxe_bright*v[1] + darke*v[1])
                    )
    # constraint 3:  prevention of image FWC saturation
    nconst3 = scipy.optimize.NonlinearConstraint(rail, 0, alpha0*fwc)

    _tmp_opt_choice = 0 #used to go to 2nd optimization if first fails


    if opt_choice == 0:
        FOM = lambda v: (v[1] + overhead)*N # no gain in wall-clock time

        res1 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([_gmin, tmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3),
                                    )

        #same thing, but starting point of gmax
        res2 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([gmax, tmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3),
                                    )

        g1 = res1.x[0]
        tfr1 = res1.x[1]
        g2 = res2.x[0]
        tfr2 = res2.x[1]
        snr1 = nowsnr(np.array([g1, tfr1]))
        snr2 = nowsnr(np.array([g2, tfr2]))

        #constraints met
        res1_cond = (
        (rail(np.array([g1, tfr1])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g1, tfr1])) <= (1+delta_constr)*(alpha1*fwc_em)) and
        (snr1 >= (1-delta_constr)*target_snr)
                    )

        res2_cond = (
        (rail(np.array([g2, tfr2])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g2, tfr2])) <= (1+delta_constr)*(alpha1*fwc_em)) and
        (snr2 >= (1-delta_constr)*target_snr)
                    )

        if res1_cond and res2_cond:
            if N*tfr1 <= N*tfr2:
                return (g1, tfr1, N, snr1, 0)
            if N*tfr1 > N*tfr2:
                return (g2, tfr2, N, snr2, 0)
        elif res1_cond and not res2_cond:
            return (g1, tfr1, N, snr1, 0)
        elif res2_cond and not res1_cond:
            return (g2, tfr2, N, snr2, 0)

        _tmp_opt_choice = 1

    if (opt_choice == 1) or (_tmp_opt_choice == 1):

        def _SNR_CR1(v, N, fluxe, darke, cic, rn, X, a, Lij, sign, Nem,
                    num_pixels):
            g, tfr = v
            return _SNR_CR_resel(g, tfr, N, fluxe, darke, cic, rn, X, a, Lij,
                        sign, Nem, num_pixels)

        res3 = scipy.optimize.minimize(fun=_SNR_CR1,
                                   x0=np.array([_gmin, tmin]),
                                   args=(N, fluxe, darke, cic, rn, X, a, Lij,
                                    -1, Nem, num_pixels),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3),
                                   )


        #same thing, but starting point of gmax
        res4 = scipy.optimize.minimize(fun=_SNR_CR1,
                                   x0=np.array([gmax, tmin]),
                                   args=(N, fluxe, darke, cic, rn, X, a, Lij,
                                    -1, Nem, num_pixels),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3),
                                   )

        g3 = res3.x[0]
        tfr3 = res3.x[1]
        g4 = res4.x[0]
        tfr4 = res4.x[1]
        snr3 = nowsnr(np.array([g3, tfr3]))
        snr4 = nowsnr(np.array([g4, tfr4]))

        #constraints met
        res3_cond = (
        (rail(np.array([g3, tfr3])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g3, tfr3])) <= (1+delta_constr)*(alpha1*fwc_em))
                    )

        res4_cond = (
        (rail(np.array([g4, tfr4])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g4, tfr4])) <= (1+delta_constr)*(alpha1*fwc_em))
                    )

        if res3_cond and res4_cond:
            if snr3 >= snr4:
                return (g3, tfr3, N, snr3, 1)
            if snr3 < snr4:
                return (g4, tfr4, N, snr4, 1)
        elif res3_cond and not res4_cond:
            return (g3, tfr3, N, snr3, 1)
        elif res4_cond and not res3_cond:
            return (g4, tfr4, N, snr4, 1)

    raise EXCAMOptimizeException('Both optimizations failed, cannot produce ' +
                                 'camera settings')

def calc_gain_fixed_g(target_snr, fluxe, fluxe_bright, darke, cic, rn, X, a,
                      Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax, tmin, tmax,
                      g, overhead, opt_choice=0, n=4, Nem=604, tol=1e-30,
                      delta_constr=1e-4, num_pixels=1):
    """
    Run 1-2 optimizations to find the best EXCAM settings.
    If you don't have a target SNR and simply want to
    maximize the SNR, you may specify opt_choice=1 to skip to the 2nd
    optimizer.

    For a given EM gain (g), this function runs an optimization to find the
    exposure time and number of frames which provides an SNR at or better than
    a target SNR with the smallest wall-clock time.  If there is no feasible
    combination which does so, it runs a second optimization to find the best
    SNR it can get given the constraints.

    Despite the long list of inputs, almost all of them are fixed properties of
    the EXCAM detector or the CGI system as a whole.  The only 3 that will
    usually be changed are the target SNR (target_snr), incoming flux
    (fluxe, in units of electrons), and the incoming peak flux (fluxe_bright,
    in units of electrons).  Both of these depend on the astrophysical target
    and the use case of CGI at that time.

    Inputs that violate the specifications given below will raise a TypeError
    or a ValueError.

    If both optimizations are somehow infeasible, or if compound constraints
    (e.g. electrons generation rate vs. full-well level) that depend on
    fluxe_bright are violated, an EXCAMOptimizeException will be raised.

    Parameters
    ----------
    target_snr : float
        Desired SNR in 'num_pixels' pixels, where 'num_pixels' is the input
        specifying the number of pixels.  This can be useful if considering a
        spatial resolution element ('resel'). >= 0.

    fluxe : float
        observation target's average flux, in electrons per second
        (i.e. photon flux (phi) * QE (eta)), in a pixel.  >= 0.

    fluxe_bright : float
        flux of the peak pixel, in electrons per second
        (i.e. photon flux (phi) * QE (eta)).  >= 0.

    darke : float
        dark current, in electrons per second.  >= 0.

    cic : float
        clock-induced charge, in electrons.  >= 0.

    rn : float
        read noise, in electrons.  >= 0.

    X : float
        cosmic ray hits/meter**2/sec.  >= 0.

    a : float
        pixel area in meters**2/pixel.  >= 0.

    Lij : int
        for a target pixel, the number of pixels including itself which can
        cause that target pixel to be made useless if hit with a cosmic ray.
        >= 1.

    alpha0 : float
        fraction of the per-pixel full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc : int
        number of electrons to fill the per-pixel full well.  >= 1.

    alpha1 : float
        fraction of the EM gain full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc_em : int
        number of electrons to fill the per-pixel EM gain full well.  >= 1.

    Nmin : int
        minimum number of allowed exposures.  >= 1.

    Nmax : int
        maximum number of allowed exposures.  >= 1.  Must be > Nmin.

    tmin : float
        minimum exposure time length, in seconds/frame.  >= 0.

    tmax : float
        maximum exposure time length, in seconds/frame.  >= 0. Must be > tmin.

    g : float
        fixed gain.  >= 1.  (Minimum gain is 1.)

    overhead : float
        Overhead per frame, in seconds, that is not spent observing.  Used to
        compute wall-clock time in optimization figure of merit.  >= 0.

    opt_choice : {0, 1}, optional
        0 to try the first optimization first and then the second if the first
        fails, 1 to go directly to the second optimization (i.e., to maximize
        the SNR without trying to minimize the total integrated exposure time).
        Defaults to 0.

    n : float, optional
        number of standard deviations the signal after gain is below the max
        fwc_em. Defaults to 4.

    Nem : int, optional
        number of gain multiplying elements.  Defaults to 604, which is the
        right hardware number for CGI cameras and very unlikely to change.

    tol : float, optional
        tolerance level used used by optimizations.  >= 0.  Recommended to be
        1e-30 for good results for any given input.  (This has been tested and
        verified over the relevant parameter space.)  Defaults to 1e-30.

    delta_constr:  float, optional
        constraint bounds relaxed via delta_constr so that optimization has
        success if constraints are satisfied within this fraction (to
        accommodate floating-point error).  >=0. For example,
        delta_constr = 0.01 means that the SNR that results from
        the optimizer should be 0.99*target_snr or bigger in order to have a
        successful return for the optimizer. Defaults to 1e-4,
        which has been tested and confirmed as a good choice for
        consistency between the first and second optimization schemes.

    num_pixels : float, optional
        The number of pixels over which to calculate the SNR.  >= 0.  Defaults
        to 1.

    Returns
    -------
    gain : float
        Fixed gain setting for detector (same as g)

    exptime : float
        Exposure time for each frame in 'n_frames' to reach 'snr_out'

    n_frames : float
        Total number of exposures to reach 'snr_out'

    snr_out : float
        Expected SNR when using above exposure settings

    optflag : int
        0 or 1: 0 if the first optimization succeeded, 1 if the first failed
        but the second succeeded.

    """

    # Check inputs
    check.real_nonnegative_scalar(target_snr, 'target_snr', TypeError)
    check.real_nonnegative_scalar(fluxe, 'fluxe', TypeError)
    check.real_nonnegative_scalar(fluxe_bright, 'fluxe_bright', TypeError)
    if fluxe_bright < fluxe:
        raise ValueError('fluxe_bright must be greater than or equal to fluxe')
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
    if Nmax <= Nmin:
        raise ValueError('Nmax must be > Nmin')
    check.real_nonnegative_scalar(tmin, 'tmin', TypeError)
    check.real_positive_scalar(tmax, 'tmax', TypeError)
    if tmax <= tmin:
        raise ValueError('tmax must be > tmin')
    check.real_positive_scalar(g, 'g', TypeError)
    if g < 1:
        raise ValueError('g must be >= 1')
    check.real_nonnegative_scalar(overhead, 'overhead', TypeError)
    check.nonnegative_scalar_integer(opt_choice, 'opt_choice', TypeError)
    if ((opt_choice != 0) and (opt_choice != 1)):
        raise ValueError('opt_choice must be 0 or 1')
    check.real_nonnegative_scalar(n, 'n', TypeError)
    check.positive_scalar_integer(Nem, 'Nem', TypeError)
    check.real_nonnegative_scalar(tol, "tol", TypeError)
    check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)
    check.real_nonnegative_scalar(num_pixels, 'num_pixels', TypeError)

    # Check that there are no unsatisfiable constraints
    # minimal case for rail constraint (parallel CIC negligible):
    if (fluxe_bright*tmin + darke*tmin) + \
        n*np.sqrt(fluxe_bright*tmin + darke*tmin) > alpha0*fwc:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')

    # in case alpha1*fwc_em is input as smaller than alpha0*fwc:
    #minimal case for em_rail constraint, serial cic ~ total cic:
    if g*(fluxe_bright*tmin + darke*tmin + cic) + n*_ENF(g, Nem)*g*\
        np.sqrt(fluxe_bright*tmin + darke*tmin + cic) > alpha1*fwc_em:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')

    # [tfr, N] = v
    bounds = scipy.optimize.Bounds(lb=np.array([tmin, Nmin]),
                                ub=np.array([tmax, Nmax]))

    em_rail = lambda v: g*(fluxe_bright*v[0] + darke*v[0] + cic) + (
                        n*_ENF(g, Nem)*g*np.sqrt(fluxe_bright*v[0] +
                        darke*v[0] + cic)
                        )
    # constraint 1:  prevention of EM FWC saturation
    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1*fwc_em)

    nowsnr = lambda v: _SNR_CR_resel(g, v[0], v[1], fluxe, darke, cic, rn, X,
                        a, Lij, sign=1, Nem=Nem, num_pixels=num_pixels)
    # constraint 2:  target SNR
    nconst2 = scipy.optimize.NonlinearConstraint(nowsnr,
                                        target_snr, np.inf)

    rail = lambda v: fluxe_bright*v[0] + darke*v[0] + (
                    n*np.sqrt(fluxe_bright*v[0] + darke*v[0])
                    )
    # constraint 3:  prevention of image FWC saturation
    nconst3 = scipy.optimize.NonlinearConstraint(rail, 0, alpha0*fwc)

    _tmp_opt_choice = 0 #used to go to 2nd optimization if first fails


    if opt_choice == 0:
        FOM = lambda v: (v[0] + overhead)*v[1] # no gain in wall-clock time

        res1 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([tmin, Nmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3),
                                    )

        #same thing, but starting point of tmax
        res2 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([tmax, Nmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3),
                                    )

        tfr1 = res1.x[0]
        N1 = int(np.ceil(res1.x[1]))
        tfr2 = res2.x[0]
        N2 = int(np.ceil(res2.x[1]))
        snr1 = nowsnr(np.array([tfr1, N1]))
        snr2 = nowsnr(np.array([tfr2, N2]))

        #constraints met
        res1_cond = (
        (rail(np.array([tfr1])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([tfr1])) <= (1+delta_constr)*(alpha1*fwc_em)) and
        (snr1 >= (1-delta_constr)*target_snr)
                    )

        res2_cond = (
        (rail(np.array([tfr2])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([tfr2])) <= (1+delta_constr)*(alpha1*fwc_em)) and
        (snr2 >= (1-delta_constr)*target_snr)
                    )

        if res1_cond and res2_cond:
            if N1*tfr1 <= N2*tfr2:
                return (g, tfr1, N1, snr1, 0)
            if N1*tfr1 > N2*tfr2:
                return (g, tfr2, N2, snr2, 0)
        elif res1_cond and not res2_cond:
            return (g, tfr1, N1, snr1, 0)
        elif res2_cond and not res1_cond:
            return (g, tfr2, N2, snr2, 0)

        _tmp_opt_choice = 1

    if (opt_choice == 1) or (_tmp_opt_choice == 1):

        def _SNR_CR1(v, g, fluxe, darke, cic, rn, X, a, Lij, sign, Nem,
                    num_pixels):
            tfr, N = v
            return _SNR_CR_resel(g, tfr, N, fluxe, darke, cic, rn, X, a, Lij,
                                sign, Nem, num_pixels)

        res3 = scipy.optimize.minimize(fun=_SNR_CR1,
                                   x0=np.array([tmin, Nmin]),
                                   args=(g, fluxe, darke, cic, rn, X, a, Lij,
                                    -1, Nem, num_pixels),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3),
                                   )


        #same thing, but starting point of tmax
        res4 = scipy.optimize.minimize(fun=_SNR_CR1,
                                   x0=np.array([tmax, Nmin]),
                                   args=(g, fluxe, darke, cic, rn, X, a, Lij,
                                    -1, Nem, num_pixels),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3),
                                   )

        tfr3 = res3.x[0]
        N3 = int(np.ceil(res3.x[1]))
        tfr4 = res4.x[0]
        N4 = int(np.ceil(res4.x[1]))
        snr3 = nowsnr(np.array([tfr3, N3]))
        snr4 = nowsnr(np.array([tfr4, N4]))

        #constraints met
        res3_cond = (
        (rail(np.array([tfr3])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([tfr3])) <= (1+delta_constr)*(alpha1*fwc_em))
                    )

        res4_cond = (
        (rail(np.array([tfr4])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([tfr4])) <= (1+delta_constr)*(alpha1*fwc_em))
                    )

        if res3_cond and res4_cond:
            if snr3 >= snr4:
                return (g, tfr3, N3, snr3, 1)
            if snr3 < snr4:
                return (g, tfr4, N4, snr4, 1)
        elif res3_cond and not res4_cond:
            return (g, tfr3, N3, snr3, 1)
        elif res4_cond and not res3_cond:
            return (g, tfr4, N4, snr4, 1)

    raise EXCAMOptimizeException('Both optimizations failed, cannot produce ' +
                                 'camera settings')

def calc_gain_fixed_Ntime(t_tot, fluxe, fluxe_bright, darke, cic, rn, X, a,
                      Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax, tmin, tmax,
                      gmax, overhead, n=4, Nem=604, tol=1e-30,
                      delta_constr=1e-4, hard_limit=True, num_pixels=1):
    """
    Runs 1 optimization to find the best EXCAM settings.  (There is no input
    for target SNR for this function since the usual first optimization of
    minimizing the total integration time is irrelevant since that time is
    fixed.  And therefore the input opt_choice and the return
    optflag found in other functions in this file are also irrelevant
    parameters that have been removed as well.)

    For a given total integration time (t_tot, the number of frames times
    the exposure time per frame + overhead), this function runs an
    optimization to find the combination of gain, exposure time, and number of
    frames which maximizes the SNR.

    Despite the long list of inputs, almost all of them are fixed properties of
    the EXCAM detector or the CGI system as a whole.  The only 2 that will
    usually be changed are the incoming flux
    (fluxe, in units of electrons) and the incoming peak flux (fluxe_bright,
    in units of electrons).  Both of these depend on the astrophysical target
    and the use case of CGI at that time.

    Inputs that violate the specifications given below will raise a TypeError
    or a ValueError.

    If the optimization is somehow infeasible, or if compound constraints
    (e.g. electrons generation rate vs. full-well level) that depend on
    fluxe_bright are violated, an EXCAMOptimizeException will be raised.

    Parameters
    ----------
    t_tot : float
        The fixed total integration time (i.e., the number of frames times
    the exposure time per frame plus the overhead per frame).  >= 0.

    fluxe : float
        observation target's average flux, in electrons per second
        (i.e. photon flux (phi) * QE (eta)), in a pixel.  >= 0.

    fluxe_bright : float
        flux of the peak pixel, in electrons per second
        (i.e. photon flux (phi) * QE (eta)).  >= 0.

    darke : float
        dark current, in electrons per second.  >= 0.

    cic : float
        clock-induced charge, in electrons.  >= 0.

    rn : float
        read noise, in electrons.  >= 0.

    X : float
        cosmic ray hits/meter**2/sec.  >= 0.

    a : float
        pixel area in meters**2/pixel.  >= 0.

    Lij : int
        for a target pixel, the number of pixels including itself which can
        cause that target pixel to be made useless if hit with a cosmic ray.
        >= 1.

    alpha0 : float
        fraction of the per-pixel full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc : int
        number of electrons to fill the per-pixel full well.  >= 1.

    alpha1 : float
        fraction of the EM gain full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc_em : int
        number of electrons to fill the per-pixel EM gain full well.  >= 1.

    Nmin : int
        minimum number of allowed exposures.  >= 1.

    Nmax : int
        maximum number of allowed exposures.  >= 1.  Must be > Nmin.

    tmin : float
        minimum exposure time length, in seconds/frame.  >= 0.

    tmax : float
        maximum exposure time length, in seconds/frame.  >= 0. Must be > tmin.

    gmax : float
        maximum permitted gain.  > 1.  (Minimum gain is 1.)

    overhead : float
        Overhead per frame, in seconds, that is not spent observing.  Used to
        compute wall-clock time in optimization figure of merit.  >= 0.

    n : float, optional
        number of standard deviations the signal after gain is below the max
        fwc_em. Defaults to 4.

    Nem : int, optional
         number of gain multiplying elements.  Defaults to 604, which is the
         right hardware number for CGI cameras and very unlikely to change.

    tol : float, optional
        tolerance level used used by optimizations.  >= 0.  Recommended to be
        1e-30 for good results for any given input.  (This has been tested and
        verified over the relevant parameter space.)  Defaults to 1e-30.

    delta_constr:  float, optional
        constraint bounds relaxed via delta_constr so that optimization has
        success if constraints are satisfied within this fraction (to
        accommodate floating-point error).  >=0. For example,
        delta_constr = 0.01 means that the SNR that results from
        the optimizer should be 0.99*target_snr or bigger in order to have a
        successful return for the optimizer. Defaults to 1e-4,
        which has been tested and confirmed as a good choice for
        consistency between the first and second optimization schemes.

    hard_limit: boolean, optional
        True if a hard limit needed for t_tot.  The number of frames (N)
        has to be treated as a float in the optimization calculation, and the
        frame time (t) is optimized under that assumption, but the optimized
        output N will be rounded up to the next integer.  If the user wants the
        optimized N*t to equal the input t_tot exactly, hard_limit should be
        True, and the output t is equal to t_tot/N - overhead.  In this case,
        the optimized SNR output may be a bit smaller than
        what it could have been with the optimized float value of N.  If
        the user is okay with a little variation between N*t and t_tot, then
        hard_limit should be False, and then
        the optimized SNR will be as big as possible since N is rounded up and
        t is the value from the optimization, and N*(t + overhead) may be
        slightly bigger than t_tot.  Defaults to True.

    num_pixels : float, optional
        The number of pixels over which to calculate the SNR.  >= 0.  Defaults
        to 1.

    Returns
    -------
    gain : float
        Optimal gain setting for detector

    exptime : float
        Exposure time for each frame in 'n_frames' to reach 'snr_out'

    n_frames : float
        Total number of exposures to reach 'snr_out'

    snr_out : float
        Expected SNR when using above exposure settings

    t_tot_out : float
        Actual total integration time used, (exp_time + overhead)*n_frames.  If
        hard_limit is True, this should equal the input t_tot.
    """

    # Check inputs
    check.real_nonnegative_scalar(t_tot, 't_tot', TypeError)
    check.real_nonnegative_scalar(fluxe, 'fluxe', TypeError)
    check.real_nonnegative_scalar(fluxe_bright, 'fluxe_bright', TypeError)
    if fluxe_bright < fluxe:
        raise ValueError('fluxe_bright must be greater than or equal to fluxe')
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
    if Nmax <= Nmin:
        raise ValueError('Nmax must be > Nmin')
    check.real_nonnegative_scalar(tmin, 'tmin', TypeError)
    check.real_positive_scalar(tmax, 'tmax', TypeError)
    if tmax <= tmin:
        raise ValueError('tmax must be > tmin')
    check.real_positive_scalar(gmax, 'gmax', TypeError)
    if gmax <= 1:
        raise ValueError('gmax must be > 1')
    check.real_nonnegative_scalar(overhead, 'overhead', TypeError)
    check.real_nonnegative_scalar(n, 'n', TypeError)
    check.positive_scalar_integer(Nem, 'Nem', TypeError)
    check.real_nonnegative_scalar(tol, "tol", TypeError)
    check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)
    if not isinstance(hard_limit, bool):
        raise TypeError('hard_limit must be boolean.')
    check.real_nonnegative_scalar(num_pixels, 'num_pixels', TypeError)
    if t_tot <= overhead:
        raise ValueError('t_tot must be > overhead')

    # effective t bounds affected by fixed t_tot
    t_lb = np.max(np.array([t_tot/Nmax-overhead, tmin]))
    t_ub = np.min(np.array([t_tot/Nmin-overhead, tmax]))

    # Check that there are no unsatisfiable constraints
    # minimal case for rail constraint (parallel CIC negligible):
    if (fluxe_bright*t_lb + darke*t_lb) + \
        n*np.sqrt((fluxe_bright*t_lb + darke*t_lb)) > alpha0*fwc:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')

    _gmin = 1

    # in case alpha1*fwc_em is input as smaller than alpha0*fwc:
    #minimal case for em_rail constraint, serial cic ~ total cic:
    if _gmin*(fluxe_bright*t_lb + darke*t_lb + cic) + \
        n*_ENF(_gmin, Nem)*_gmin*\
        np.sqrt(fluxe_bright*t_lb + darke*t_lb + cic) > alpha1*fwc_em:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')

    if Nmin*(t_lb + overhead) > t_tot:
        raise EXCAMOptimizeException('t_tot is smaller than Nmin*tmin ' +
                                     'with overhead')

    if Nmax*(t_ub + overhead) < t_tot:
        raise EXCAMOptimizeException('t_tot is bigger than Nmax*tmax ' +
                                     'with overhead')

    # SNR that is in terms of t_tot and not N
    def _SNR_CR_ttot(v, t_tot, fluxe, darke, cic, rn, X, a, Lij, sign, Nem,
                     num_pixels, overhead):
        g, tfr = v
        # resel_factor = num_pixels**0.5*(np.exp(-X*a*Lij*tfr*
        #                                         (num_pixels**0.5-1)))**0.5
        # num = np.sqrt((t_tot/(tfr+overhead))*np.exp(-X*a*Lij*tfr))*fluxe*tfr
        # den = _SNR_CR_den(g, Nem, rn, fluxe, tfr, darke, cic)
        # if den == 0:
        #     return 0 # in the limit den terms go to zero, num terms go faster
        # return sign*num*resel_factor/den
        N = t_tot/(tfr + overhead)
        if num_pixels == 1:
            return _SNR_CR(g, tfr, N, fluxe, darke, cic, rn, X, a, Lij, sign,
                            Nem)
        else:
            return (_SNR_CR(g, tfr, N, fluxe, darke, cic, rn, X,
                    (4*num_pixels/np.pi)**0.5*a, Lij, sign, Nem)
                    * num_pixels**0.5)

    # Eq. 40 in etc_snr_v3b.pdf in the doc folder
    # define this function for use in fsolve below
    # I include cic as an argument so that I can easily reproduce the
    # constraint Eq. 39 from the PDF
    def emrail0(g, cic, t0):
        return g*(fluxe_bright*t0 + darke*t0 + cic) + \
            n*_ENF(g, Nem)*g*np.sqrt(fluxe_bright*t0 + darke*t0 +
            cic) - alpha1*fwc_em

    def _max_snr_g(t0):
        """Function useful for calc_gain_fixed_Ntime().  It determines the
        maximum SNR from the allowed values of g whenever N and t are both
        fixed.  It compares SNR values
        evaluated at _gmin, any g values for which there are local extrema in
        the SNR, and the max value of g determined by the emrail constraint."""
        # below would have no solution only in
        # non-realistic cases (such as 0 noise); otherwise, at most, just one
        # solution since all g dependence causes monotonic increase
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gsol, _, flag, _ = scipy.optimize.fsolve(func=emrail0, x0=1,
                args=(cic, t0), full_output=True)
        if flag == 1:
            # effective upper bound of g:
            gsol0 = np.min(gsol) # should just be one entry in this array
            # If gsol>gmax, then EM well saturation won't happen for the
            # current inputs. Restrict bound to gmax:
            gsol0 = min(gsol0, gmax)
            # gsol can't be smaller than 1 (and if it's smaller than 1,
            # then all possible gain values saturate the EM well for the
            # current inputs).  gsol restricted to be at least _gmin, and
            # the emrail saturation constraint should
            # catch this later.
            gsol0 = max(gsol0, _gmin)
        else: # then upper bound is simply gmax
            gsol0 = gmax

        # local extrema found from setting derivative of SNR wrt g to 0
        # (and g=0 solution eliminated from this expression)
        # SNR expression used to get this: Eq. 30 of etc_snr_v3b.pdf from
        # doc folder (and this is independent of num_pixels)
        def snr_deriv_exp(g):
            return 2*(fluxe*t0 + darke*t0 + cic)*g*(1-g+Nem)-g**(1/Nem)*Nem*(
                (fluxe*t0 + darke*t0 + cic)*g+2*rn**2)

        with warnings.catch_warnings():
            # if no local extrema, fsolve gives a Runtime Warning
            warnings.simplefilter('ignore')
            g_vals, _, flag1, _ = scipy.optimize.fsolve(func=snr_deriv_exp,
                                x0=1, full_output=True)

        # include the endpoints of g as well for determining max snr
        # we do the above method instead of using scipy.optimize.minimize to
        # avoid potential issues with nonlinear constraints
        if flag1 != 1: #no solution found, so no local max
            g_vals = np.array([gsol0, _gmin, gmax])
        else:
            g_vals = np.append(g_vals, (gsol0, _gmin, gmax))
        # g_vals should obeys gmax and _gmin limits
        # gsol at the least can be _gmin but not less than it; but apply just
        # in case fsolve goes wonky
        g_vals1 = np.where(g_vals <= _gmin, _gmin, g_vals)
        g_vals2 = np.where(g_vals1 >= gsol0, gsol0, g_vals1)

        snr0 = - np.inf
        for i in g_vals2:
            snr = _SNR_CR_ttot([i, t0], t_tot, fluxe, darke, cic, rn, X, a,
                               Lij, sign=1, Nem=Nem, num_pixels=num_pixels,
                               overhead=overhead)
            if snr > snr0:
                snr0 = snr
                g0 = i

        return g0, snr0

    if t_lb == t_ub:
        # t and N fixed in this case:
        t0 = t_lb
        N0 = t_tot/(t0 + overhead)

        g0, snr0 = _max_snr_g(t0)

        res0_cond = (
        #Eqs. 39 and 40 from etc_snr_v3b.pdf in the doc folder, as usual:
        # first constraint below was checked before this if statement
        # second one should be satisfied by fsolve, but in case solver goes
        # wonky, check it
        ((emrail0(1, 0, t0) + alpha1*fwc_em)/(alpha0*fwc)
                                                   <= 1 + delta_constr) and
        ((emrail0(g0, cic, t0) + alpha1*fwc_em)/(alpha1*fwc_em)
                                                   <= 1+delta_constr) and
        (N0/Nmin >= 1) and
        (N0/Nmax <= 1) # N0, g0, t0: should be satisfied by construction
                    )
        if res0_cond:
            return (g0, t0, N0, snr0, (t0 + overhead)*N0)
        else:
            # would only ever happen if fsolve goes wonky
            raise EXCAMOptimizeException('Optimization failed, ' +
                                'cannot produce camera settings')

    #g, tfr = v
    lb = np.array([_gmin, t_lb])
    ub = np.array([gmax, t_ub])
    bounds = scipy.optimize.Bounds(lb=lb, ub=ub)

    num_frames = lambda v: t_tot/(v[1] + overhead)
    nconst4 = scipy.optimize.NonlinearConstraint(num_frames, Nmin, Nmax)

    em_rail = lambda v: v[0]*(fluxe_bright*v[1] + darke*v[1] + cic) + (
                        n*_ENF(v[0], Nem)*v[0]*np.sqrt(fluxe_bright*v[1] +
                        darke*v[1] + cic)
                        )

    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1*fwc_em)

    nowsnr = lambda v: _SNR_CR_ttot(v, t_tot, fluxe, darke, cic, rn, X, a, Lij,
                                    sign=1, Nem=Nem, num_pixels=num_pixels,
                                    overhead=overhead,
    )


    rail = lambda v: fluxe_bright*v[1] + darke*v[1] + (
                    n*np.sqrt(fluxe_bright*v[1] + darke*v[1])
                    )

    nconst3 = scipy.optimize.NonlinearConstraint(rail, 0, alpha0*fwc)

    res3 = scipy.optimize.minimize(fun=_SNR_CR_ttot,
                                x0=np.array([_gmin, t_lb]),
                                args=(t_tot, fluxe, darke, cic, rn, X, a,
                                        Lij, -1, Nem, num_pixels, overhead),
                                bounds=bounds,
                                tol=tol,
                                constraints=(nconst1, nconst3, nconst4)
                                )

    #same thing, but starting point of gmax
    res4 = scipy.optimize.minimize(fun=_SNR_CR_ttot,
                                x0=np.array([gmax, t_lb]),
                                args=(t_tot, fluxe, darke, cic, rn, X, a,
                                        Lij, -1, Nem, num_pixels, overhead),
                                bounds=bounds,
                                tol=tol,
                                constraints=(nconst1, nconst3, nconst4)
                                )

    g3 = res3.x[0]
    g4 = res4.x[0]

    #constraints met initially, before making N integer (we include check on N
    #since it is involved in a nonlinear constraint rather than merely a bound)
    res3_cond_init = (
    (rail(np.array([g3, res3.x[1]]))/(alpha0*fwc) <= 1+delta_constr) and
    (em_rail(np.array([g3, res3.x[1]]))/(alpha1*fwc_em) <= 1+delta_constr) and
    (num_frames(np.array([g3, res3.x[1]]))/Nmin >= 1-delta_constr) and
    (num_frames(np.array([g3, res3.x[1]]))/Nmax <= 1+delta_constr) and
    #treat Nmin and Nmax as hard limits, just as t and g have
    # hard limits in "bounds"
    # but allow for up to a whole integer off (within a relative delta_constr)
    (num_frames(np.array([g3, res3.x[1]])) > Nmin - 1) and
    (num_frames(np.array([g3, res3.x[1]])) < Nmax + 1)
                )

    res4_cond_init = (
    (rail(np.array([g4, res4.x[1]]))/(alpha0*fwc) <= 1+delta_constr) and
    (em_rail(np.array([g4, res4.x[1]]))/(alpha1*fwc_em) <= 1+delta_constr) and
    (num_frames(np.array([g4, res4.x[1]]))/Nmin >= 1-delta_constr) and
    (num_frames(np.array([g4, res4.x[1]]))/Nmax <= 1+delta_constr) and
    #treat Nmin and Nmax as hard limits, just as t and g have
    # hard limits in "bounds"
    # but allow for up to a whole integer off (within a relative delta_constr)
    (num_frames(np.array([g4, res4.x[1]])) > Nmin - 1) and
    (num_frames(np.array([g4, res4.x[1]])) < Nmax + 1)
                )

    if not res3_cond_init and not res4_cond_init:
        raise EXCAMOptimizeException('Optimization failed, cannot produce ' +
                                 'camera settings')

    #we typically want ceil here b/c our goal is to maximize SNR; N drives
    # SNR more than t for most common times and noise params, but not always.
    # For hard_limit=False, t doesn't
    #change, so increasing N is always better for SNR, as long as it doesn't
    # go above Nmax
    N3_ceil = np.ceil(num_frames(np.array([g3, res3.x[1]])))
    N3_floor = np.floor(num_frames(np.array([g3, res3.x[1]])))
    N4_ceil = np.ceil(num_frames(np.array([g4, res4.x[1]])))
    N4_floor = np.floor(num_frames(np.array([g4, res4.x[1]])))

    if not hard_limit:
        N3 = min(Nmax, N3_ceil)
        tfr3 = res3.x[1]
        N4 = min(Nmax, N4_ceil)
        tfr4 = res4.x[1]
        snr3 = nowsnr(np.array([g3, tfr3]))
        snr4 = nowsnr(np.array([g4, tfr4]))
        # no change in t or g, so we use init res conditions from above
        if (res3_cond_init and res4_cond_init):
            if snr3 >= snr4:
                return (g3, tfr3, N3, snr3, (tfr3 + overhead)*N3)
            if snr3 < snr4:
                return (g4, tfr4, N4, snr4, (tfr4 + overhead)*N4)
        elif res3_cond_init and not res4_cond_init:
            return (g3, tfr3, N3, snr3, (tfr3 + overhead)*N3)
        elif res4_cond_init and not res3_cond_init:
            return (g4, tfr4, N4, snr4, (tfr4 + overhead)*N4)

    if hard_limit:

        # max time from rail constraint, from Eq. 39 of etc_snr_v3b.pdf in
        # doc folder (but including effect of delta_constr; for use below)
        rail_t = (2*alpha0*fwc*(1+delta_constr) + n*(n -
                np.sqrt(4*alpha0*fwc*(1+delta_constr) + n**2)))/(2*
                (fluxe_bright+darke))

        # N3 cases
        if N3_ceil == N3_floor: #then N3 is integer; no adjustment needed
            res3_cond = True #initialize
            N3 = N3_ceil
            tfr3 = res3.x[1]
            snr3 = nowsnr(np.array([g3, tfr3]))

        if N3_ceil != N3_floor:
            res3_cond = True #initialize
            if N3_ceil > Nmax:
                if t_tot/Nmax - overhead > tmax:
                    # then N*t cannot equal t_tot w/o violating tmax, Nmax
                    res3_cond = False
                else: #t_tot/Nmax within bounds
                    N3 = Nmax
                    #time already checked at beginning: doesn't violate rail
                    tfr3 = t_tot/N3 - overhead
                    # optimal g would shift slightly b/c of shift in t, so:
                    g3, snr3 = _max_snr_g(tfr3)
            elif N3_ceil <= Nmax and N3_floor >= Nmin:
                N3_up = N3_ceil
                N3_low = N3_floor
                if t_tot/N3_up - overhead < tmin and \
                   t_tot/N3_low - overhead > tmax:
                    # then N*t cannot equal t_tot w/o violating tmin, tmax
                    res3_cond = False
                elif t_tot/N3_up - overhead >= tmin and \
                     t_tot/N3_low - overhead > tmax:
                    if t_tot/N3_up - overhead > rail_t:
                        res3_cond = False
                    else:
                        N3 = N3_up
                        tfr3 = t_tot/N3_up - overhead
                        g3, snr3 = _max_snr_g(tfr3)
                elif t_tot/N3_up - overhead < tmin and \
                     t_tot/N3_low - overhead <= tmax:
                    if t_tot/N3_low - overhead > rail_t:
                        res3_cond = False
                    else:
                        N3 = N3_low
                        tfr3 = t_tot/N3_low - overhead
                        g3, snr3 = _max_snr_g(tfr3)
                elif t_tot/N3_up - overhead >= tmin and \
                     t_tot/N3_low - overhead <= tmax:
                    N3_up_cond = True
                    N3_low_cond = True
                    if t_tot/N3_up - overhead > rail_t:
                        N3_up_cond = False
                    else:
                        g3_up, snr3_up = _max_snr_g(t_tot/N3_up - overhead)
                    if t_tot/N3_low - overhead > rail_t:
                        N3_low_cond = False
                    else:
                        g3_low, snr3_low = _max_snr_g(t_tot/N3_low - overhead)
                    if not N3_up_cond and not N3_low_cond:
                        res3_cond = False
                    elif not N3_up_cond and N3_low_cond:
                        N3 = N3_low
                        tfr3 = t_tot/N3_low - overhead
                        snr3 = snr3_low
                    elif N3_up_cond and not N3_low_cond:
                        N3 = N3_up
                        tfr3 = t_tot/N3_up - overhead
                        snr3 = snr3_up
                    elif N3_up_cond and N3_low_cond:
                        if snr3_up >= snr3_low:
                            N3 = N3_up
                            tfr3 = t_tot/N3_up - overhead
                            g3 = g3_up
                            snr3 = snr3_up
                        if snr3_up < snr3_low:
                            N3 = N3_low
                            tfr3 = t_tot/N3_low - overhead
                            g3 = g3_low
                            snr3 = snr3_low
            elif N3_floor < Nmin:
                if t_tot/Nmin - overhead < tmin:
                    # then N*t cannot equal t_tot w/o violating tmin, Nmin
                    res3_cond = False
                else: #t_tot/Nmin within bounds
                    if t_tot/Nmin - overhead > rail_t:
                        res3_cond = False
                    else:
                        N3 = Nmin
                        tfr3 = t_tot/N3 - overhead
                        # optimal g would shift slightly b/c of shift in t, so:
                        g3, snr3 = _max_snr_g(tfr3)

        # N4 cases
        if N4_ceil == N4_floor: #then N4 is integer; no adjustment needed
            res4_cond = True #initialize
            N4 = N4_ceil
            tfr4 = res4.x[1]
            snr4 = nowsnr(np.array([g4, tfr4]))

        if N4_ceil != N4_floor:
            res4_cond = True #initialize
            if N4_ceil > Nmax:
                if t_tot/Nmax - overhead > tmax:
                    # then N*t cannot equal t_tot w/o violating tmax, Nmax
                    res4_cond = False
                else: #t_tot/Nmax within bounds
                    N4 = Nmax
                    #time already checked at beginning: doesn't violate rail
                    tfr4 = t_tot/N4 - overhead
                    # optimal g would shift slightly b/c of shift in t, so:
                    g4, snr4 = _max_snr_g(tfr4)
            elif N4_ceil <= Nmax and N4_floor >= Nmin:
                N4_up = N4_ceil
                N4_low = N4_floor
                if t_tot/N4_up - overhead < tmin and \
                   t_tot/N4_low - overhead > tmax:
                    # then N*t cannot equal t_tot w/o violating tmin, tmax
                    res4_cond = False
                elif t_tot/N4_up - overhead >= tmin and \
                     t_tot/N4_low - overhead > tmax:
                    if t_tot/N4_up - overhead > rail_t:
                        res4_cond = False
                    else:
                        N4 = N4_up
                        tfr4 = t_tot/N4_up - overhead
                        g4, snr4 = _max_snr_g(tfr4)
                elif t_tot/N4_up - overhead < tmin and \
                     t_tot/N4_low - overhead <= tmax:
                    if t_tot/N4_low - overhead > rail_t:
                        res4_cond = False
                    else:
                        N4 = N4_low
                        tfr4 = t_tot/N4_low - overhead
                        g4, snr4 = _max_snr_g(tfr4)
                elif t_tot/N4_up - overhead >= tmin and \
                     t_tot/N4_low - overhead <= tmax:
                    N4_up_cond = True
                    N4_low_cond = True
                    if t_tot/N4_up - overhead > rail_t:
                        N4_up_cond = False
                    else:
                        g4_up, snr4_up = _max_snr_g(t_tot/N4_up - overhead)
                    if t_tot/N4_low - overhead > rail_t:
                        N4_low_cond = False
                    else:
                        g4_low, snr4_low = _max_snr_g(t_tot/N4_low - overhead)
                    if not N4_up_cond and not N4_low_cond:
                        res4_cond = False
                    elif not N4_up_cond and N4_low_cond:
                        N4 = N4_low
                        tfr4 = t_tot/N4_low - overhead
                        snr4 = snr4_low
                    elif N4_up_cond and not N4_low_cond:
                        N4 = N4_up
                        tfr4 = t_tot/N4_up - overhead
                        snr4 = snr4_up
                    elif N4_up_cond and N4_low_cond:
                        if snr4_up >= snr4_low:
                            N4 = N4_up
                            tfr4 = t_tot/N4_up - overhead
                            g4 = g4_up
                            snr4 = snr4_up
                        if snr4_up < snr4_low:
                            N4 = N4_low
                            tfr4 = t_tot/N4_low - overhead
                            g4 = g4_low
                            snr4 = snr4_low
            elif N4_floor < Nmin:
                if t_tot/Nmin - overhead < tmin:
                    # then N*t cannot equal t_tot w/o violating tmin, Nmin
                    res4_cond = False
                else: #t_tot/Nmin within bounds
                    if t_tot/Nmin - overhead > rail_t:
                        res4_cond = False
                    else:
                        N4 = Nmin
                        tfr4 = t_tot/N4 - overhead
                        # optimal g would shift slightly b/c of shift in t, so:
                        g4, snr4 = _max_snr_g(tfr4)

        #constraints met (since integer N was fully specified in above, we now
        # give the N constraints below no wiggle room)
        res3_cond_final = (
        res3_cond and
        (rail(np.array([g3, tfr3]))/(alpha0*fwc) <= 1+delta_constr) and
        (em_rail(np.array([g3, tfr3]))/(alpha1*fwc_em) <= 1+delta_constr) and
        (N3/Nmin >= 1) and
        (N3/Nmax <= 1)
                    )

        #constraints met (since integer N was fully specified in above, we now
        # give the N constraints below no wiggle room)
        res4_cond_final = (
        res4_cond and
        (rail(np.array([g4, tfr4]))/(alpha0*fwc) <= 1+delta_constr) and
        (em_rail(np.array([g4, tfr4]))/(alpha1*fwc_em) <= 1+delta_constr) and
        (N4/Nmin >= 1) and
        (N4/Nmax <= 1)
                    )

        if (res3_cond_final and res4_cond_final):
            if snr3 >= snr4:
                return (g3, tfr3, N3, snr3, (tfr3 + overhead)*N3)
            if snr3 < snr4:
                return (g4, tfr4, N4, snr4, (tfr4 + overhead)*N4)
        elif res3_cond_final and not res4_cond_final:
            return (g3, tfr3, N3, snr3, (tfr3 + overhead)*N3)
        elif res4_cond_final and not res3_cond_final:
            return (g4, tfr4, N4, snr4, (tfr4 + overhead)*N4)

        raise EXCAMOptimizeException('Optimization failed, cannot produce ' +
                                    'camera settings for ' +
                                    'hard_limit=True.')


def _mean(g, L, T, N):
    """
    Computes the mean for the photon-counting SNR.  (Only used in
    _SNR_CR_pc.)  The calculation assumes the mean number of electrons per
    pixel per frame (in e-) is relatively small (for a 3rd-order
    Taylor expansion), as it should be for photon counting, and it assumes
    2 iterations of photometric corrections as is done in PhotonCounting.  See
    paper in doc folder of eetc for more details.

    Parameters
    ----------
    g : float
        camera gain, unitless.

    L : float
        mean number of electrons per frame per pixel, in electrons.

    T : float
        threshold for photon counting, in electrons.  Typically,
        T = 5*(read noise).

    N : float
        number of frames, unitless.

    Returns
    -------
    float
        mean value

    """
    # adjusted things below from the paper in doc folder: moved np.e**L from
    # Const and abosrbed it into
    # eThresh to cancel with np.e**(-L) there, which eliminated overflow issues
    # with large L values that occurred during some iterations in optimization
    Const = 6/(6 + L*(6 + L*(3 + L)))

    eThresh = (Const*(np.e**(-T/g)*L*(2*g**2*(6 + L*(3 + L)) +
            2*g*L*(3 + L)*T + L**2*T**2))/(12*g**2))

    exp1 = N*eThresh

    exp2 = N*eThresh*(1 + (-1 + N)*eThresh)

    # When N is huge, or if fluxe is too big (which shouldn't be the case for
    # pc), this can have convergence issues or even turn out
    # complex due to machine precision limitation.  In these cases, good
    # enough to just use L for _mean().
    try:
        exp3 = float((eThresh*(-((-1 + N)*(-1 + eThresh)**2*
            (1 + (-2 + N)*eThresh*(3 + (-3 + N)*eThresh))) -
            (1 - eThresh)**N*
            hyper([2, 2, 2, 1 - N], [1, 1, 1], 1 + 1/(-1 + eThresh))
            ))/(-1 + eThresh))

        return (np.e**(T/g)*exp1/N +
                np.e**(2*T/g)*(g-T)*exp2/(2*g*N**2) +
                np.e**(3*T/g)*(4*g**2-8*g*T+5*T**2)*exp3/(12*g**2*N**3))
    except:
        return L

def _var(g, L, T, N):
    """
    Computes the variance for the photon-counting SNR.  (Only used in
    _SNR_CR_pc.)  The calculation assumes the mean number of electrons per
    pixel per frame (in e-) is relatively small (for a 3rd-order
    Taylor expansion), as it should be for photon counting, and it assumes
    2 iterations of photometric corrections as is done in PhotonCounting.  See
    paper in doc folder of eetc for more details.

    Parameters
    ----------
    g : float
        camera gain, unitless.

    L : float
        mean number of electrons per frame per pixel, in electrons.

    T : float
        threshold for photon counting, in electrons.  Typically,
        T = 5*(read noise).

    N : float
        number of frames, unitless.

    Returns
    -------
    float
        variance value

    """
    # adjusted things below from the paper in doc folder: moved np.e**L from
    # Const and abosrbed it into
    # eThresh to cancel with np.e**(-L) there, which eliminated overflow issues
    # with large L values that occurred during some iterations in optimization
    Const = 6/(6 + L*(6 + L*(3 + L)))

    eThresh = (Const*(np.e**(-T/g)*L*(2*g**2*(6 + L*(3 + L)) +
            2*g*L*(3 + L)*T + L**2*T**2))/(12*g**2))

    std_dev = np.sqrt(eThresh * (1-eThresh))

    return N*(std_dev)**2*(((np.e**((T/g)))/N) +
        2*((np.e**((2*T)/g)*(g - T))/(2*g*N**2))*(N*eThresh) +
        3*(((np.e**(((3*T)/g)))*(4*g**2 - 8*g*T + 5*T**2))/(
        12*g**2*N**3))*(N*eThresh)**2)**2

def _SNR_CR_pc(g, tfr, N, fluxe, darke, cic, T, X, a, Lij, sign):
    """
    Computes the photon-counting SNR per pixel given camera settings and noise
    properties, including cosmic ray effects.  This also assumes that
    photon-counted dark frames are subtracted and that the number of frames
    used for dark calibration is much more than the number of photon-counted
    dark frames.

    Internal function to feed the optimizer only.  Does not enforce any of the
    bounds you might expect, as the optimizer will eventually enforce them, so
    use for non-optimizer applications at your own risk.

    Parameters
    ----------
    g : float
        camera gain, unitless.

    tfr : float
        exposure time per frame, in seconds.

    N : float
        number of frames, unitless.

    fluxe : float
        flux, in electrons per second (i.e. photon flux (phi) * QE (eta)).

    darke : float
        dark current, in electrons per second.

    cic : float
        clock-induced charge, in electrons.

    T : float
        threshold for photon counting, in electrons.  Typically,
        T = 5*(read noise).

    X : float
        cosmic ray hits/meter**2/sec.

    a : float
        pixel area in meters**2/pixel.

    Lij : int
        for a target pixel, the number of pixels including itself which can
        cause that target pixel to be made useless if the other pixel is hit.

    sign : {1, -1}
        1 or -1, to return a positive or negative value (for minimization or
        maximization, respectively)

    Returns
    -------
    float
        SNR value (single floating-point number)

    """
    # lambda; serial cic treated as avg charge present before gain,
    # so included here
    L = fluxe*tfr + darke*tfr + cic
    L_dk = darke*tfr + cic
    # didn't originally account for cosmics in SNR formulation, so do that here
    N_CR = N*np.e**(-X*a*tfr*Lij)
    # just as in the analog case, the number of dark frames taken assumed to be
    # huge compared to number of brights; in that case, _mean() for the darks
    # approaches L_dk (and we don't know the exact number of dark frames to
    # plug in here anyways), and _var() for them approaches 0:
    numerator = _mean(g, L, T, N_CR) - L_dk
    denominator = np.sqrt(_var(g, L, T, N_CR))
    # note:  The SNR in general monotonically increases with N EXCEPT for
    # values of N between 1 and 2. This range of N is not realistic for photon
    # counting anyways, but it means that optimization in the pc functions
    # could get confused on either side of the local minimum in the SNR.
    # Optimizer functions should start the optimization search at Nmin and Nmax
    # and pick the better result (which is what is done).
    if denominator == 0:
        return 0
    else:
        snr = numerator/denominator
        return sign*snr

def _SNR_CR_pc_resel(g, tfr, N, fluxe, darke, cic, T, X, a, Lij, sign,
                    num_pixels):
    """
    Computes the photon-counting SNR per spatial resolution element ('resel')
    comprised of 'num_pixels' pixels, given camera settings and noise
    properties, including cosmic ray effects.  This also assumes that
    photon-counted dark frames are subtracted and that the number of frames
    used for dark calibration is much more than the number of photon-counted
    dark frames.  SNR per resel is modeled in a certain way as detailed in
    snr_resel.pdf in the doc folder of eetc.

    Internal function to feed the optimizer only.  Does not enforce any of the
    bounds you might expect, as the optimizer will eventually enforce them, so
    use for non-optimizer applications at your own risk.

    Parameters
    ----------
    g : float
        camera gain, unitless.

    tfr : float
        exposure time per frame, in seconds.

    N : float
        number of frames, unitless.

    fluxe : float
        flux, in electrons per second (i.e. photon flux (phi) * QE (eta)).

    darke : float
        dark current, in electrons per second.

    cic : float
        clock-induced charge, in electrons.

    T : float
        threshold for photon counting, in electrons.  Typically,
        T = 5*(read noise).

    X : float
        cosmic ray hits/meter**2/sec.

    a : float
        pixel area in meters**2/pixel.

    Lij : int
        for a target pixel, the number of pixels including itself which can
        cause that target pixel to be made useless if the other pixel is hit.

    sign : {1, -1}
        1 or -1, to return a positive or negative value (for minimization or
        maximization, respectively)

    num_pixels : float
        number of pixels considered.

    Returns
    -------
    float
        SNR value (single floating-point number)

    """
    # see snr_resel.pdf in doc folder for details
    if num_pixels == 1:
        return _SNR_CR_pc(g, tfr, N, fluxe, darke, cic, T, X, a, Lij, sign)
    else:
        return (_SNR_CR_pc(g, tfr, N, fluxe, darke, cic, T, X,
            (4*num_pixels/np.pi)**0.5*a, Lij, sign)*num_pixels**0.5)

def calc_pc(target_snr, fluxe, fluxe_bright, darke, cic, rn, X, a,
                      Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax,
                      tmin, tmax, gmax, overhead, pc_ecount_max, T_factor=5,
                      opt_choice=0, n=4, Nem=604, tol=1e-30, delta_constr=1e-4,
                      num_pixels=1):
    """
    Runs 1-2 optimization to find the best EXCAM settings for photon counting.
    Used for continuous-frame observation.  It raises an exception if inputs
    are not conducive to photon counting.  If you don't have a target SNR and
    simply want to maximize the SNR, you may specify opt_choice=1 to skip to
    the 2nd optimizer.

    This function runs an optimization to find the combination of gain,
    exposure time, and number of frames which provides an SNR at or better than
    a target SNR with the smallest wall-clock time.  If there is no feasible
    combination which does so, it runs a second optimization to find the best
    SNR it can get given the constraints.

    Despite the long list of inputs, almost all of them are fixed properties of
    the EXCAM detector or the CGI system as a whole.  The only 3 that will
    usually be changed are the target SNR (target_snr), incoming flux
    (fluxe, in units of electrons), and the incoming peak flux (fluxe_bright,
    in units of electrons).  Both of these depend on the astrophysical target
    and the use case of CGI at that time.

    Inputs that violate the specifications given below will raise a TypeError
    or a ValueError.

    If both optimizations are somehow infeasible, or if compound constraints
    (e.g. electrons generation rate vs. full-well level) that depend on
    fluxe_bright are violated, an EXCAMOptimizeException will be raised.

    Parameters
    ----------
    target_snr : float
        Desired SNR in 'num_pixels' pixels, where 'num_pixels' is the input
        specifying the number of pixels.  This can be useful if considering a
        spatial resolution element ('resel'). >= 0.

    fluxe : float
        observation target's average flux, in electrons per second
        (i.e. photon flux (phi) * QE (eta)), in a pixel.  >= 0.

    fluxe_bright : float
        flux of the peak pixel, in electrons per second
        (i.e. photon flux (phi) * QE (eta)).  >= 0.

    darke : float
        dark current, in electrons per second.  >= 0.

    cic : float
        clock-induced charge, in electrons.  >= 0.

    rn : float
        read noise, in electrons.  >= 0.

    X : float
        cosmic ray hits/meter**2/sec.  >= 0.

    a : float
        pixel area in meters**2/pixel.  >= 0.

    Lij : int
        for a target pixel, the number of pixels including itself which can
        cause that target pixel to be made useless if hit with a cosmic ray.
        >= 1.

    alpha0 : float
        fraction of the per-pixel full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc : int
        number of electrons to fill the per-pixel full well.  >= 1.

    alpha1 : float
        fraction of the EM gain full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc_em : int
        number of electrons to fill the per-pixel EM gain full well.  >= 1.

    Nmin : int
        minimum number of allowed exposures.  >= 1.

    Nmax : int
        maximum number of allowed exposures.  >= 1.  Must be > Nmin.

    tmin : float
        minimum exposure time length, in seconds/frame.  >= 0.

    tmax : float
        maximum exposure time length, in seconds/frame.  >= 0. Must be > tmin.

    gmax : float
        maximum permitted gain.  > 1.  (Minimum gain is 1.)

    overhead : float
        Overhead per frame, in seconds, that is not spent observing.  Used to
        compute wall-clock time in optimization figure of merit.  >= 0.

    pc_ecount_max : float
        maximum photo-electron flux allowed in a pixel for photon counting.
        Units of electrons/pixel/frame.

    T_factor : float, optional
        number of read noise standard deviations at which to set the
        photon-counting threshold.  The default is the suggested value of 5.

    opt_choice : {0, 1}, optional
        0 to try the first optimization first and then the second if the first
        fails, 1 to go directly to the second optimization (i.e., to maximize
        the SNR without trying to minimize the total integrated exposure time).
        Defaults to 0.

    n : float, optional
        number of standard deviations the signal after gain is below the max
        fwc_em. Defaults to 4.

    Nem : int, optional
         number of gain multiplying elements.  Defaults to 604, which is the
         right hardware number for CGI cameras and very unlikely to change.

    tol : float, optional
        tolerance level used used by optimizations.  >= 0.  Recommended to be
        1e-30 for good results for any given input.  (This has been tested and
        verified over the relevant parameter space.)  Defaults to 1e-30.

    delta_constr:  float, optional
        constraint bounds relaxed via delta_constr so that optimization has
        success if constraints are satisfied within this fraction (to
        accommodate floating-point error).  >=0. For example,
        delta_constr = 0.01 means that the SNR that results from
        the optimizer should be 0.99*target_snr or bigger in order to have a
        successful return for the optimizer. Defaults to 1e-4,
        which has been tested and confirmed as a good choice for
        consistency between the first and second optimization schemes.

    num_pixels : float, optional
        The number of pixels over which to calculate the SNR.  >= 0.  Defaults
        to 1.

    Returns
    -------
    gain : float
        Optimal gain setting for detector

    exptime : float
        Exposure time for each frame in 'n_frames' to reach 'snr_out'

    n_frames : float
        Total number of exposures to reach 'snr_out'

    snr_out : float
        Expected SNR when using above exposure settings

    optflag : int
        0 or 1: 0 if the first optimization succeeded, 1 if the first failed
        but the second succeeded.
    """
    # Check inputs
    check.real_nonnegative_scalar(target_snr, 'target_snr', TypeError)
    check.real_nonnegative_scalar(fluxe, 'fluxe', TypeError)
    check.real_nonnegative_scalar(fluxe_bright, 'fluxe_bright', TypeError)
    if fluxe_bright < fluxe:
        raise ValueError('fluxe_bright must be greater than or equal to fluxe')
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
    if Nmax <= Nmin:
        raise ValueError('Nmax must be > Nmin')
    check.real_nonnegative_scalar(tmin, 'tmin', TypeError)
    check.real_positive_scalar(tmax, 'tmax', TypeError)
    if tmax <= tmin:
        raise ValueError('tmax must be > tmin')
    check.real_positive_scalar(gmax, 'gmax', TypeError)
    if gmax <= 1:
        raise ValueError('gmax must be > 1')
    check.real_nonnegative_scalar(overhead, 'overhead', TypeError)
    check.real_positive_scalar(pc_ecount_max, 'pc_ecount_max', TypeError)
    check.real_positive_scalar(T_factor, 'T_factor', TypeError)
    check.positive_scalar_integer(Nem, 'Nem', TypeError)
    if ((opt_choice != 0) and (opt_choice != 1)):
        raise ValueError('opt_choice must be 0 or 1')
    check.real_nonnegative_scalar(n, 'n', TypeError)
    check.real_nonnegative_scalar(tol, "tol", TypeError)
    check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)
    check.real_nonnegative_scalar(num_pixels, 'num_pixels', TypeError)

    _gmin = 1
    if fluxe > 0:
        t_pcmax = (pc_ecount_max)/(fluxe)
    elif fluxe == 0:
        t_pcmax = np.inf
    t_ub = np.min(np.array([tmax, t_pcmax]))
    # threshold, T; need not be 5*cic since we are subtracting pc darks
    T = T_factor*rn
    g_lb = np.max(np.array([_gmin, T]))
    # Check that there are no unsatisfiable constraints; parallel cic ~ 0
    if (fluxe_bright*tmin + darke*tmin) + \
        n*np.sqrt((fluxe_bright*tmin + darke*tmin)) > alpha0*fwc:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')
    if g_lb*(fluxe_bright*tmin + darke*tmin + cic) + n*_ENF(g_lb, Nem)*g_lb*\
        np.sqrt(fluxe_bright*tmin + darke*tmin + cic) > alpha1*fwc_em:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')
    if g_lb >= gmax:
        raise EXCAMOptimizeException("No finite-width window of viable EM " +
        'gain for photon counting')
    if tmin >= t_ub:
        raise EXCAMOptimizeException('No finite-width window of viable frame '+
        'time for photon counting')

    # [g, tfr, N] = v
    bounds = scipy.optimize.Bounds(lb=np.array([g_lb, tmin, Nmin]),
                                ub=np.array([gmax, t_ub, Nmax]))

    em_rail = lambda v: v[0]*(fluxe_bright*v[1] + darke*v[1] + cic) + (
                        n*_ENF(v[0], Nem)*v[0]*np.sqrt(fluxe_bright*v[1] +
                        darke*v[1] + cic)
                        )
    # constraint 1:  prevention of EM FWC saturation
    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1*fwc_em)

    nowsnr = lambda v: _SNR_CR_pc_resel(v[0], v[1], v[2], fluxe, darke, cic,
                        T, X, a, Lij, sign=1, num_pixels=num_pixels)

    # constraint 2:  SNR target
    nconst2 = scipy.optimize.NonlinearConstraint(nowsnr,
                                        target_snr, np.inf)

    rail = lambda v: fluxe_bright*v[1] + darke*v[1] + (
                    n*np.sqrt(fluxe_bright*v[1] + darke*v[1])
                    )
    # constraint 3:  prevention of image FWC saturation
    nconst3 = scipy.optimize.NonlinearConstraint(rail, 0, alpha0*fwc)

    _tmp_opt_choice = 0 #used to go to 2nd optimization if first fails

    if opt_choice == 0:
        FOM = lambda v: (v[1] + overhead)*v[2] # no gain in wall-clock time

        res1 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([g_lb, tmin, Nmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3)
                                    )

        #same thing, but starting point of gmax and Nmax
        res2 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([gmax, tmin, Nmax]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3)
                                    )

        g1 = res1.x[0]
        tfr1 = res1.x[1]
        N1 = int(np.ceil(res1.x[2]))
        g2 = res2.x[0]
        tfr2 = res2.x[1]
        N2 = int(np.ceil(res2.x[2]))
        snr1 = nowsnr(np.array([g1, tfr1, N1]))
        snr2 = nowsnr(np.array([g2, tfr2, N2]))

        #constraints met
        res1_cond = (
        (rail(np.array([g1, tfr1])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g1, tfr1])) <= (1+delta_constr)*(alpha1*fwc_em)) and
        (snr1 >= (1-delta_constr)*target_snr)
                    )

        res2_cond = (
        (rail(np.array([g2, tfr2])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g2, tfr2])) <= (1+delta_constr)*(alpha1*fwc_em)) and
        (snr2 >= (1-delta_constr)*target_snr)
                    )

        if res1_cond and res2_cond:
            if N1*tfr1 <= N2*tfr2:
                return (g1, tfr1, N1, snr1, 0)
            if N1*tfr1 > N2*tfr2:
                return (g2, tfr2, N2, snr2, 0)
        elif res1_cond and not res2_cond:
            return (g1, tfr1, N1, snr1, 0)
        elif res2_cond and not res1_cond:
            return (g2, tfr2, N2, snr2, 0)

        _tmp_opt_choice = 1

    if (opt_choice == 1) or (_tmp_opt_choice == 1):

        def _SNR_CR_pc1(v, fluxe, darke, cic, T, X, a, Lij, sign, num_pixels):
            g, tfr, N = v
            return _SNR_CR_pc_resel(g, tfr, N, fluxe, darke, cic, T, X, a, Lij,
                    sign, num_pixels)

        res3 = scipy.optimize.minimize(fun=_SNR_CR_pc1,
                                   x0=np.array([g_lb, tmin, Nmin]),
                                   args=(fluxe, darke, cic, T, X, a, Lij, -1,
                                        num_pixels),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3)
                                   )


        #same thing, but starting point of gmax and Nmax
        res4 = scipy.optimize.minimize(fun=_SNR_CR_pc1,
                                   x0=np.array([gmax, tmin, Nmax]),
                                   args=(fluxe, darke, cic, rn, X, a, Lij, -1,
                                        num_pixels),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3),
                                   )

        g3 = res3.x[0]
        tfr3 = res3.x[1]
        N3 = int(np.ceil(res3.x[2]))
        g4 = res4.x[0]
        tfr4 = res4.x[1]
        N4 = int(np.ceil(res4.x[2]))
        snr3 = nowsnr(np.array([g3, tfr3, N3]))
        snr4 = nowsnr(np.array([g4, tfr4, N4]))

        #constraints met
        res3_cond = (
        (rail(np.array([g3, tfr3])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g3, tfr3])) <= (1+delta_constr)*(alpha1*fwc_em))
                    )

        res4_cond = (
        (rail(np.array([g4, tfr4])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g4, tfr4])) <= (1+delta_constr)*(alpha1*fwc_em))
                    )

        if res3_cond and res4_cond:
            if snr3 >= snr4:
                return (g3, tfr3, N3, snr3, 1)
            if snr3 < snr4:
                return (g4, tfr4, N4, snr4, 1)
        elif res3_cond and not res4_cond:
            return (g3, tfr3, N3, snr3, 1)
        elif res4_cond and not res3_cond:
            return (g4, tfr4, N4, snr4, 1)

    raise EXCAMOptimizeException('Both optimizations failed, cannot produce ' +
                                 'camera settings')

def calc_pc_fixed_N(target_snr, fluxe, fluxe_bright, darke, cic, rn, X, a,
                      Lij, alpha0, fwc, alpha1, fwc_em, N, tmin, tmax, gmax,
                      overhead, pc_ecount_max, T_factor=5,
                      opt_choice=0, n=4, Nem=604, tol=1e-30, delta_constr=1e-4,
                      num_pixels=1):
    """
    Runs 1-2 optimization to find the best EXCAM settings for photon counting.
    Used for continuous-frame observation.  It raises an exception if inputs
    are not conducive to photon counting.  If you don't have a target SNR and
    simply want to maximize the SNR, you may specify opt_choice=1 to skip to
    the 2nd optimizer.

    For a given number of frames (N), this function runs an optimization to
    find the combination of gain
    and exposure time which provides an SNR at or better than
    a target SNR with the smallest wall-clock time.  If there is no feasible
    combination which does so, it runs a second optimization to find the best
    SNR it can get given the constraints.

    Despite the long list of inputs, almost all of them are fixed properties of
    the EXCAM detector or the CGI system as a whole.  The only 3 that will
    usually be changed are the target SNR (target_snr), incoming flux
    (fluxe, in units of electrons), and the incoming peak flux (fluxe_bright,
    in units of electrons).  Both of these depend on the astrophysical target
    and the use case of CGI at that time.

    Inputs that violate the specifications given below will raise a TypeError
    or a ValueError.

    If both optimizations are somehow infeasible, or if compound constraints
    (e.g. electrons generation rate vs. full-well level) that depend on
    fluxe_bright are violated, an EXCAMOptimizeException will be raised.

    Parameters
    ----------
    target_snr : float
        Desired SNR in 'num_pixels' pixels, where 'num_pixels' is the input
        specifying the number of pixels.  This can be useful if considering a
        spatial resolution element ('resel'). >= 0.

    fluxe : float
        observation target's average flux, in electrons per second
        (i.e. photon flux (phi) * QE (eta)), in a pixel.  >= 0.

    fluxe_bright : float
        flux of the peak pixel, in electrons per second
        (i.e. photon flux (phi) * QE (eta)).  >= 0.

    darke : float
        dark current, in electrons per second.  >= 0.

    cic : float
        clock-induced charge, in electrons.  >= 0.

    rn : float
        read noise, in electrons.  >= 0.

    X : float
        cosmic ray hits/meter**2/sec.  >= 0.

    a : float
        pixel area in meters**2/pixel.  >= 0.

    Lij : int
        for a target pixel, the number of pixels including itself which can
        cause that target pixel to be made useless if hit with a cosmic ray.
        >= 1.

    alpha0 : float
        fraction of the per-pixel full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc : int
        number of electrons to fill the per-pixel full well.  >= 1.

    alpha1 : float
        fraction of the EM gain full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc_em : int
        number of electrons to fill the per-pixel EM gain full well.  >= 1.

    N : int
        the fixed number of frames.  >= 1.

    tmin : float
        minimum exposure time length, in seconds/frame.  >= 0.

    tmax : float
        maximum exposure time length, in seconds/frame.  >= 0. Must be > tmin.

    gmax : float
        maximum permitted gain.  > 1.  (Minimum gain is 1.)

    overhead : float
        Overhead per frame, in seconds, that is not spent observing.  Used to
        compute wall-clock time in optimization figure of merit.  >= 0.

    pc_ecount_max : float
        maximum photo-electron flux allowed in a pixel for photon counting.
        Units of electrons/pixel/frame.

    T_factor : float, optional
        number of read noise standard deviations at which to set the
        photon-counting threshold.  The default is the suggested value of 5.

    opt_choice : {0, 1}, optional
        0 to try the first optimization first and then the second if the first
        fails, 1 to go directly to the second optimization (i.e., to maximize
        the SNR without trying to minimize the total integrated exposure time).
        Defaults to 0.

    n : float, optional
        number of standard deviations the signal after gain is below the max
        fwc_em. Defaults to 4.

    Nem : int, optional
         number of gain multiplying elements.  Defaults to 604, which is the
         right hardware number for CGI cameras and very unlikely to change.

    tol : float, optional
        tolerance level used used by optimizations.  >= 0.  Recommended to be
        1e-30 for good results for any given input.  (This has been tested and
        verified over the relevant parameter space.)  Defaults to 1e-30.

    delta_constr:  float, optional
        constraint bounds relaxed via delta_constr so that optimization has
        success if constraints are satisfied within this fraction (to
        accommodate floating-point error).  >=0. For example,
        delta_constr = 0.01 means that the SNR that results from
        the optimizer should be 0.99*target_snr or bigger in order to have a
        successful return for the optimizer. Defaults to 1e-4,
        which has been tested and confirmed as a good choice for
        consistency between the first and second optimization schemes.

    num_pixels : float, optional
        The number of pixels over which to calculate the SNR.  >= 0.  Defaults
        to 1.

    Returns
    -------
    gain : float
        Optimal gain setting for detector

    exptime : float
        Exposure time for each frame in 'n_frames' to reach 'snr_out'

    n_frames : float
        Total number of exposures to reach 'snr_out'

    snr_out : float
        Expected SNR when using above exposure settings

    optflag : int
        0 or 1: 0 if the first optimization succeeded, 1 if the first failed
        but the second succeeded.
    """
    # Check inputs
    check.real_nonnegative_scalar(target_snr, 'target_snr', TypeError)
    check.real_nonnegative_scalar(fluxe, 'fluxe', TypeError)
    check.real_nonnegative_scalar(fluxe_bright, 'fluxe_bright', TypeError)
    if fluxe_bright < fluxe:
        raise ValueError('fluxe_bright must be greater than or equal to fluxe')
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
    check.positive_scalar_integer(N, 'N', TypeError)
    check.real_nonnegative_scalar(tmin, 'tmin', TypeError)
    check.real_positive_scalar(tmax, 'tmax', TypeError)
    if tmax <= tmin:
        raise ValueError('tmax must be > tmin')
    check.real_positive_scalar(gmax, 'gmax', TypeError)
    if gmax <= 1:
        raise ValueError('gmax must be > 1')
    check.real_nonnegative_scalar(overhead, 'overhead', TypeError)
    check.real_positive_scalar(pc_ecount_max, 'pc_ecount_max', TypeError)
    check.real_positive_scalar(T_factor, 'T_factor', TypeError)
    check.positive_scalar_integer(Nem, 'Nem', TypeError)
    if ((opt_choice != 0) and (opt_choice != 1)):
        raise ValueError('opt_choice must be 0 or 1')
    check.real_nonnegative_scalar(n, 'n', TypeError)
    check.real_nonnegative_scalar(tol, "tol", TypeError)
    check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)
    check.real_nonnegative_scalar(num_pixels, 'num_pixels', TypeError)

    _gmin = 1
    if fluxe > 0:
        t_pcmax = (pc_ecount_max)/(fluxe)
    elif fluxe == 0:
        t_pcmax = np.inf
    t_ub = np.min(np.array([tmax, t_pcmax]))
    # threshold, T; need not be 5*cic since we are subtracting pc darks
    T = T_factor*rn
    g_lb = np.max(np.array([_gmin, T]))
    # Check that there are no unsatisfiable constraints; parallel cic ~ 0
    if (fluxe_bright*tmin + darke*tmin) + \
        n*np.sqrt((fluxe_bright*tmin + darke*tmin)) > alpha0*fwc:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')
    if g_lb*(fluxe_bright*tmin + darke*tmin + cic) + n*_ENF(g_lb, Nem)*g_lb*\
        np.sqrt(fluxe_bright*tmin + darke*tmin + cic) > alpha1*fwc_em:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')
    if g_lb >= gmax:
        raise EXCAMOptimizeException("No finite-width window of viable EM " +
        'gain for photon counting')
    if tmin >= t_ub:
        raise EXCAMOptimizeException('No finite-width window of viable frame '+
        'time for photon counting')

    # [g, tfr] = v
    bounds = scipy.optimize.Bounds(lb=np.array([g_lb, tmin]),
                                ub=np.array([gmax, t_ub]))

    em_rail = lambda v: v[0]*(fluxe_bright*v[1] + darke*v[1] + cic) + (
                        n*_ENF(v[0], Nem)*v[0]*np.sqrt(fluxe_bright*v[1] +
                        darke*v[1] + cic)
                        )
    # constraint 1:  prevention of EM FWC saturation
    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1*fwc_em)

    nowsnr = lambda v: _SNR_CR_pc_resel(v[0], v[1], N, fluxe, darke, cic, T, X,
                        a, Lij, sign=1, num_pixels=num_pixels)
    # constraint 2:  SNR target
    nconst2 = scipy.optimize.NonlinearConstraint(nowsnr,
                                        target_snr, np.inf)

    rail = lambda v: fluxe_bright*v[1] + darke*v[1] + (
                    n*np.sqrt(fluxe_bright*v[1] + darke*v[1])
                    )
    # constraint 3:  prevention of image FWC saturation
    nconst3 = scipy.optimize.NonlinearConstraint(rail, 0, alpha0*fwc)

    _tmp_opt_choice = 0 #used to go to 2nd optimization if first fails

    if opt_choice == 0:
        FOM = lambda v: (v[1] + overhead)*N # no gain in wall-clock time

        res1 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([g_lb, tmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3)
                                    )

        #same thing, but starting point of gmax and tmax
        res2 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([gmax, tmax]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3)
                                    )

        g1 = res1.x[0]
        tfr1 = res1.x[1]
        g2 = res2.x[0]
        tfr2 = res2.x[1]
        snr1 = nowsnr(np.array([g1, tfr1]))
        snr2 = nowsnr(np.array([g2, tfr2]))

        #constraints met
        res1_cond = (
        (rail(np.array([g1, tfr1])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g1, tfr1])) <= (1+delta_constr)*(alpha1*fwc_em)) and
        (snr1 >= (1-delta_constr)*target_snr)
                    )

        res2_cond = (
        (rail(np.array([g2, tfr2])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g2, tfr2])) <= (1+delta_constr)*(alpha1*fwc_em)) and
        (snr2 >= (1-delta_constr)*target_snr)
                    )

        if res1_cond and res2_cond:
            if N*tfr1 <= N*tfr2:
                return (g1, tfr1, N, snr1, 0)
            if N*tfr1 > N*tfr2:
                return (g2, tfr2, N, snr2, 0)
        elif res1_cond and not res2_cond:
            return (g1, tfr1, N, snr1, 0)
        elif res2_cond and not res1_cond:
            return (g2, tfr2, N, snr2, 0)

        _tmp_opt_choice = 1

    if (opt_choice == 1) or (_tmp_opt_choice == 1):

        def _SNR_CR_pc1(v, N, fluxe, darke, cic, T, X, a, Lij, sign,
                        num_pixels):
            g, tfr = v
            return _SNR_CR_pc_resel(g, tfr, N, fluxe, darke, cic, T, X, a, Lij,
                                    sign, num_pixels)

        res3 = scipy.optimize.minimize(fun=_SNR_CR_pc1,
                                   x0=np.array([g_lb, tmin]),
                                   args=(N, fluxe, darke, cic, T, X, a, Lij,
                                         -1, num_pixels),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3)
                                   )


        #same thing, but starting point of gmax and tmax
        res4 = scipy.optimize.minimize(fun=_SNR_CR_pc1,
                                   x0=np.array([gmax, tmax]),
                                   args=(N, fluxe, darke, cic, rn, X, a, Lij,
                                         -1, num_pixels),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3),
                                   )

        g3 = res3.x[0]
        tfr3 = res3.x[1]
        g4 = res4.x[0]
        tfr4 = res4.x[1]
        snr3 = nowsnr(np.array([g3, tfr3]))
        snr4 = nowsnr(np.array([g4, tfr4]))

        #constraints met
        res3_cond = (
        (rail(np.array([g3, tfr3])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g3, tfr3])) <= (1+delta_constr)*(alpha1*fwc_em))
                    )

        res4_cond = (
        (rail(np.array([g4, tfr4])) <= (1+delta_constr)*(alpha0*fwc)) and
        (em_rail(np.array([g4, tfr4])) <= (1+delta_constr)*(alpha1*fwc_em))
                    )

        if res3_cond and res4_cond:
            if snr3 >= snr4:
                return (g3, tfr3, N, snr3, 1)
            if snr3 < snr4:
                return (g4, tfr4, N, snr4, 1)
        elif res3_cond and not res4_cond:
            return (g3, tfr3, N, snr3, 1)
        elif res4_cond and not res3_cond:
            return (g4, tfr4, N, snr4, 1)

    raise EXCAMOptimizeException('Both optimizations failed, cannot produce ' +
                                 'camera settings')

def calc_pc_gain_fixed_Ntime(t_tot, fluxe, fluxe_bright, darke, cic, rn, X, a,
                      Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax, tmin, tmax,
                      gmax, overhead, pc_ecount_max,
                      T_factor=5, n=4, Nem=604, tol=1e-30,
                      delta_constr=1e-4, hard_limit=True, num_pixels=1):
    """
    Runs 1 optimization to find the best EXCAM settings for
    photon counting.  (There is no input
    for target SNR for this function since the usual first optimization of
    minimizing the total integration time is irrelevant since that time is
    fixed.  And therefore the input opt_choice and the return
    optflag found in other functions in this file are also irrelevant
    parameters that have been removed as well.)

    For a given total integration time (t_tot, the number of frames times
    the exposure time per frame + overhead), this function runs an
    optimization to find the combination of gain, exposure time, and number of
    frames which maximizes the SNR.

    Despite the long list of inputs, almost all of them are fixed properties of
    the EXCAM detector or the CGI system as a whole.  The only 2 that will
    usually be changed are the incoming flux
    (fluxe, in units of electrons) and the incoming peak flux (fluxe_bright,
    in units of electrons).  Both of these depend on the astrophysical target
    and the use case of CGI at that time.

    Inputs that violate the specifications given below will raise a TypeError
    or a ValueError.

    If the optimization is somehow infeasible, or if compound constraints
    (e.g. electrons generation rate vs. full-well level) that depend on
    fluxe_bright are violated, an EXCAMOptimizeException will be raised.

    Parameters
    ----------
    t_tot : float
        The fixed total integration time (i.e., the number of frames times
    the exposure time per frame + overhead per frame).  >= 0.

    fluxe : float
        observation target's average flux, in electrons per second
        (i.e. photon flux (phi) * QE (eta)), in a pixel.  >= 0.

    fluxe_bright : float
        flux of the peak pixel, in electrons per second
        (i.e. photon flux (phi) * QE (eta)).  >= 0.

    darke : float
        dark current, in electrons per second.  >= 0.

    cic : float
        clock-induced charge, in electrons.  >= 0.

    rn : float
        read noise, in electrons.  >= 0.

    X : float
        cosmic ray hits/meter**2/sec.  >= 0.

    a : float
        pixel area in meters**2/pixel.  >= 0.

    Lij : int
        for a target pixel, the number of pixels including itself which can
        cause that target pixel to be made useless if hit with a cosmic ray.
        >= 1.

    alpha0 : float
        fraction of the per-pixel full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc : int
        number of electrons to fill the per-pixel full well.  >= 1.

    alpha1 : float
        fraction of the EM gain full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc_em : int
        number of electrons to fill the per-pixel EM gain full well.  >= 1.

    Nmin : int
        minimum number of allowed exposures.  >= 1.

    Nmax : int
        maximum number of allowed exposures.  >= 1.  Must be > Nmin.

    tmin : float
        minimum exposure time length, in seconds/frame.  >= 0.

    tmax : float
        maximum exposure time length, in seconds/frame.  >= 0. Must be > tmin.

    gmax : float
        maximum permitted gain.  > 1.  (Minimum gain is 1.)

    overhead : float
        Overhead per frame, in seconds, that is not spent observing.  Used to
        compute wall-clock time in optimization figure of merit.  >= 0.

    pc_ecount_max : float
        maximum photo-electron flux allowed in a pixel for photon counting.
        Units of electrons/pixel/frame.

    T_factor : float, optional
        number of read noise standard deviations at which to set the
        photon-counting threshold.  The default is the suggested value of 5.

    n : float, optional
        number of standard deviations the signal after gain is below the max
        fwc_em. Defaults to 4.

    Nem : int, optional
         number of gain multiplying elements.  Defaults to 604, which is the
         right hardware number for CGI cameras and very unlikely to change.

    tol : float, optional
        tolerance level used used by optimizations.  >= 0.  Recommended to be
        1e-30 for good results for any given input.  (This has been tested and
        verified over the relevant parameter space.)  Defaults to 1e-30.

    delta_constr:  float, optional
        constraint bounds relaxed via delta_constr so that optimization has
        success if constraints are satisfied within this fraction (to
        accommodate floating-point error).  >=0. For example,
        delta_constr = 0.01 means that the SNR that results from
        the optimizer should be 0.99*target_snr or bigger in order to have a
        successful return for the optimizer. Defaults to 1e-4,
        which has been tested and confirmed as a good choice for
        consistency between the first and second optimization schemes.

    hard_limit: boolean, optional
        True if a hard limit needed for t_tot.  The number of frames (N)
        has to be treated as a float in the optimization calculation, and the
        frame time (t) is optimized under that assumption, but the optimized
        output N will be rounded up to the next integer.  If the user wants the
        optimized N*t to equal the input t_tot exactly, hard_limit should be
        True, and the output t is equal to t_tot/N - overhead.  In this case,
        the optimized SNR output may be a bit smaller than
        what it could have been with the optimized float value of N.  If
        the user is okay with a little variation between N*t and t_tot, then
        hard_limit should be False, and then
        the optimized SNR will be as big as possible since N is rounded up and
        t is the value from the optimization, and N*(t + overhead) may be
        slightly bigger than t_tot.  Defaults to True.

    num_pixels : float, optional
        The number of pixels over which to calculate the SNR.  >= 0.  Defaults
        to 1.

    Returns
    -------
    gain : float
        Optimal gain setting for detector

    exptime : float
        Exposure time for each frame in 'n_frames' to reach 'snr_out'

    n_frames : float
        Total number of exposures to reach 'snr_out'

    snr_out : float
        Expected SNR when using above exposure settings

    t_tot_out : float
        Actual total integration time used, (exp_time + overhead)*n_frames.  If
        hard_limit is True, this should equal the input t_tot.
    """

    # Check inputs
    check.real_nonnegative_scalar(t_tot, 't_tot', TypeError)
    check.real_nonnegative_scalar(fluxe, 'fluxe', TypeError)
    check.real_nonnegative_scalar(fluxe_bright, 'fluxe_bright', TypeError)
    if fluxe_bright < fluxe:
        raise ValueError('fluxe_bright must be greater than or equal to fluxe')
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
    if Nmax <= Nmin:
        raise ValueError('Nmax must be > Nmin')
    check.real_nonnegative_scalar(tmin, 'tmin', TypeError)
    check.real_positive_scalar(tmax, 'tmax', TypeError)
    if tmax <= tmin:
        raise ValueError('tmax must be > tmin')
    check.real_positive_scalar(gmax, 'gmax', TypeError)
    if gmax <= 1:
        raise ValueError('gmax must be > 1')
    check.real_nonnegative_scalar(overhead, 'overhead', TypeError)
    check.real_positive_scalar(pc_ecount_max, 'pc_ecount_max', TypeError)
    check.real_positive_scalar(T_factor, 'T_factor', TypeError)
    check.real_nonnegative_scalar(n, 'n', TypeError)
    check.positive_scalar_integer(Nem, 'Nem', TypeError)
    check.real_nonnegative_scalar(tol, "tol", TypeError)
    check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)
    if not isinstance(hard_limit, bool):
        raise TypeError('hard_limit must be boolean.')
    check.real_nonnegative_scalar(num_pixels, 'num_pixels', TypeError)
    if t_tot <= overhead:
        raise ValueError('t_tot must be > overhead')

    # effective t bounds affected by fixed t_tot
    t_lb = np.max(np.array([t_tot/Nmax - overhead, tmin]))
    t_ub = np.min(np.array([t_tot/Nmin - overhead, tmax]))

    # Check that there are no unsatisfiable constraints
    # minimal case for rail constraint (parallel CIC negligible):
    if (fluxe_bright*t_lb + darke*t_lb) + \
        n*np.sqrt((fluxe_bright*t_lb + darke*t_lb)) > alpha0*fwc:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')

    _gmin = 1
    if fluxe > 0:
        t_pcmax = (pc_ecount_max)/(fluxe)
    elif fluxe == 0:
        t_pcmax = np.inf
    # generic upper bound, regardless of choice of N
    t_ub_gen = np.min(np.array([tmax, t_pcmax]))
    t_UB = np.min(np.array([t_ub, t_pcmax]))
    # threshold, T; need not be 5*cic since we are subtracting pc darks
    T = T_factor*rn
    g_lb = np.max(np.array([_gmin, T]))
    # in case alpha1*fwc_em is input as smaller than alpha0*fwc:
    #minimal case for em_rail constraint, serial cic ~ total cic:
    if g_lb*(fluxe_bright*t_lb + darke*t_lb + cic) + \
        n*_ENF(g_lb, Nem)*g_lb*\
        np.sqrt(fluxe_bright*t_lb + darke*t_lb + cic) > alpha1*fwc_em:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')

    if Nmin*(t_lb + overhead) > t_tot:
        raise EXCAMOptimizeException('t_tot is smaller than Nmin*tmin ' +
                                     'with overhead')

    if Nmax*(t_UB + overhead) < t_tot:
        raise EXCAMOptimizeException('t_tot is bigger than Nmax*tmax, with ' +
                                     'overhead, or the exposure time ' +
                                     'allowed for pc_ecount_max')

    if g_lb >= gmax:
        raise EXCAMOptimizeException("No finite-width window of viable EM " +
        'gain for photon counting')
    if tmin >= t_UB:
        raise EXCAMOptimizeException('No finite-width window of viable frame '+
        'time for photon counting')


    # SNR that is in terms of t_tot and not N
    def _SNR_CR_ttot(v, t_tot, fluxe, darke, cic, T, X, a, Lij, sign,
                     num_pixels, overhead):
        g, tfr = v
        # resel_factor = num_pixels**0.5*(np.exp(-X*a*Lij*tfr*
        #                                         (num_pixels**0.5-1)))**0.5
        # num = np.sqrt((t_tot/(tfr+overhead))*np.exp(-X*a*Lij*tfr))*fluxe*tfr
        # den = _SNR_CR_den(g, Nem, rn, fluxe, tfr, darke, cic)
        # if den == 0:
        #     return 0 # in the limit den terms go to zero, num terms go faster
        # return sign*num*resel_factor/den
        N = t_tot/(tfr + overhead)
        if num_pixels == 1:
            return _SNR_CR_pc(g, tfr, N, fluxe, darke, cic, T, X, a, Lij, sign)

        else:
            return (_SNR_CR_pc(g, tfr, N, fluxe, darke, cic, T, X,
                    (4*num_pixels/np.pi)**0.5*a, Lij, sign)
                    * num_pixels**0.5)

    # Eq. 40 in etc_snr_v3b.pdf in the doc folder
    # define this function for use in fsolve below
    # I include cic as an argument so that I can easily reproduce the
    # constraint Eq. 39 from the PDF
    def emrail0(g, cic, t0):
        return g*(fluxe_bright*t0 + darke*t0 + cic) + \
            n*_ENF(g, Nem)*g*np.sqrt(fluxe_bright*t0 + darke*t0 +
            cic) - alpha1*fwc_em

    def _max_snr_g(t0):
        """Function useful for calc_pc_gain_fixed_Ntime().  It determines the
        maximum SNR from the allowed values of g whenever N and t are both
        fixed.  It compares SNR values
        evaluated at g_lb, any g values for which there are local extrema in
        the SNR, and the max value of g determined by the emrail constraint."""
        # below would have no solution only in
        # non-realistic cases (such as 0 noise); otherwise, at most, just one
        # solution since all g dependence causes monotonic increase
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gsol, _, flag, _ = scipy.optimize.fsolve(func=emrail0, x0=1,
                args=(cic, t0), full_output=True)

        if flag == 1:
            # effective upper bound of g:
            gsol0 = np.min(gsol) # should just be one entry in this array
            # If gsol>gmax, then EM well saturation won't happen for the
            # current inputs. Restrict bound to gmax:
            gsol0 = min(gsol0, gmax)
            # gsol can't be smaller than 1 (and if it's smaller than 1,
            # then all possible gain values saturate the EM well for the
            # current inputs).  gsol restricted to be at least _gmin, and
            # the emrail saturation constraint should
            # catch this later.
            gsol0 = max(gsol0, g_lb)
        else: # then upper bound is simply gmax
            gsol0 = gmax

        # find any local maxima that may exist
        def _SNR_CR_ttot1(g, tfr, t_tot, fluxe, darke, cic, T, X, a, Lij, sign,
                          num_pixels, overhead):
            return _SNR_CR_ttot([g, tfr], t_tot, fluxe, darke, cic, T, X, a,
                                Lij, sign, num_pixels, overhead)

        lower = np.array([g_lb])
        upper = np.array([gmax])
        bound = scipy.optimize.Bounds(lb=lower, ub=upper)
        sres = scipy.optimize.minimize(fun=_SNR_CR_ttot1,
                            x0=np.array([g_lb]),
                            args=(t0, t_tot, fluxe, darke, cic, T, X, a,
                                    Lij, -1, num_pixels, overhead),
                            bounds=bound,
                            tol=tol
                            )
        g0 = sres.x[0]
        g_vals = np.array([g0, gsol0, g_lb, gmax])
        # include the endpoints of g as well for determining max snr

        # g_vals should obeys gmax and g_lb limits
        # gsol at the least can be g_lb but not less than it; but apply just
        # in case optimizer goes wonky
        g_vals1 = np.where(g_vals <= g_lb, g_lb, g_vals)
        g_vals2 = np.where(g_vals1 >= gsol0, gsol0, g_vals1)

        snr0 = - np.inf
        for i in g_vals2:
            snr = _SNR_CR_ttot([i, t0], t_tot, fluxe, darke, cic, T, X, a,
                               Lij, sign=1, num_pixels=num_pixels,
                               overhead=overhead)
            if snr > snr0:
                snr0 = snr
                g0 = i

        return g0, snr0

    if t_lb == t_UB:
        # t and N fixed in this case:
        t0 = t_lb
        N0 = t_tot/(t0 + overhead)

        g0, snr0 = _max_snr_g(t0)

        res0_cond = (
        #Eqs. 39 and 40 from etc_snr_v3b.pdf in the doc folder, as usual:
        # first constraint below was checked before this if statement
        # second one should be satisfied by fsolve, but in case solver goes
        # wonky, check it
        ((emrail0(1, 0, t0) + alpha1*fwc_em)/(alpha0*fwc)
                                                 <= 1 + delta_constr) and
        ((emrail0(g0, cic, t0) + alpha1*fwc_em)/(alpha1*fwc_em)
                                                 <= 1 + delta_constr) and
        (N0/Nmin >= 1) and
        (N0/Nmax <= 1) # N0, g0, t0: should be satisfied by construction
                    )
        if res0_cond:
            return (g0, t0, N0, snr0, (t0 + overhead)*N0)
        else:
            # would only ever happen if fsolve goes wonky
            raise EXCAMOptimizeException('Optimization failed, ' +
                                'cannot produce camera settings')

    #g, tfr = v
    lb = np.array([g_lb, t_lb])
    ub = np.array([gmax, t_UB])
    bounds = scipy.optimize.Bounds(lb=lb, ub=ub)

    num_frames = lambda v: t_tot/(v[1] + overhead)
    nconst4 = scipy.optimize.NonlinearConstraint(num_frames, Nmin, Nmax)

    em_rail = lambda v: v[0]*(fluxe_bright*v[1] + darke*v[1] + cic) + (
                        n*_ENF(v[0], Nem)*v[0]*np.sqrt(fluxe_bright*v[1] +
                        darke*v[1] + cic)
                        )

    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1*fwc_em)

    nowsnr = lambda v: _SNR_CR_ttot(v, t_tot, fluxe, darke, cic, T, X, a, Lij,
                                    sign=1, num_pixels=num_pixels,
                                    overhead=overhead)


    rail = lambda v: fluxe_bright*v[1] + darke*v[1] + (
                    n*np.sqrt(fluxe_bright*v[1] + darke*v[1])
                    )

    nconst3 = scipy.optimize.NonlinearConstraint(rail, 0, alpha0*fwc)

    res3 = scipy.optimize.minimize(fun=_SNR_CR_ttot,
                                x0=np.array([g_lb, t_lb]),
                                args=(t_tot, fluxe, darke, cic, T, X, a,
                                        Lij, -1, num_pixels, overhead),
                                bounds=bounds,
                                tol=tol,
                                constraints=(nconst1, nconst3, nconst4)
                                )

    #same thing, but starting point of gmax and t_UB
    res4 = scipy.optimize.minimize(fun=_SNR_CR_ttot,
                                x0=np.array([gmax, t_UB]),
                                args=(t_tot, fluxe, darke, cic, T, X, a,
                                        Lij, -1, num_pixels, overhead),
                                bounds=bounds,
                                tol=tol,
                                constraints=(nconst1, nconst3, nconst4)
                                )

    g3 = res3.x[0]
    g4 = res4.x[0]

    #constraints met initially, before making N integer (we include check on N
    #since it is involved in a nonlinear constraint rather than merely a bound)
    res3_cond_init = (
    (rail(np.array([g3, res3.x[1]]))/(alpha0*fwc) <= 1+delta_constr) and
    (em_rail(np.array([g3, res3.x[1]]))/(alpha1*fwc_em) <= 1+delta_constr) and
    (num_frames(np.array([g3, res3.x[1]]))/Nmin >= 1-delta_constr) and
    (num_frames(np.array([g3, res3.x[1]]))/Nmax <= 1+delta_constr) and
    #treat Nmin and Nmax as hard limits, just as t and g have
    # hard limits in "bounds"
    # but allow for up to a whole integer off (within a relative delta_constr)
    (num_frames(np.array([g3, res3.x[1]])) > Nmin - 1) and
    (num_frames(np.array([g3, res3.x[1]])) < Nmax + 1)
                )

    res4_cond_init = (
    (rail(np.array([g4, res4.x[1]]))/(alpha0*fwc) <= 1+delta_constr) and
    (em_rail(np.array([g4, res4.x[1]]))/(alpha1*fwc_em) <= 1+delta_constr) and
    (num_frames(np.array([g4, res4.x[1]]))/Nmin >= 1-delta_constr) and
    (num_frames(np.array([g4, res4.x[1]]))/Nmax <= 1+delta_constr) and
    #treat Nmin and Nmax as hard limits, just as t and g have
    # hard limits in "bounds"
    # but allow for up to a whole integer off (within a relative delta_constr)
    (num_frames(np.array([g4, res4.x[1]])) > Nmin - 1) and
    (num_frames(np.array([g4, res4.x[1]])) < Nmax + 1)
                )

    if not res3_cond_init and not res4_cond_init:
        raise EXCAMOptimizeException('Optimization failed, cannot produce ' +
                                 'camera settings')

    #we typically want ceil here b/c our goal is to maximize SNR; N drives
    # SNR more than t for most common times and noise params, but not always.
    # For hard_limit=False, t doesn't
    #change, so increasing N is always better for SNR, as long as it doesn't
    # go above Nmax
    N3_ceil = np.ceil(num_frames(np.array([g3, res3.x[1]])))
    N3_floor = np.floor(num_frames(np.array([g3, res3.x[1]])))
    N4_ceil = np.ceil(num_frames(np.array([g4, res4.x[1]])))
    N4_floor = np.floor(num_frames(np.array([g4, res4.x[1]])))

    if not hard_limit:
        N3 = min(Nmax, N3_ceil)
        tfr3 = res3.x[1]
        N4 = min(Nmax, N4_ceil)
        tfr4 = res4.x[1]
        snr3 = nowsnr(np.array([g3, tfr3]))
        snr4 = nowsnr(np.array([g4, tfr4]))
        # no change in t or g, so we use init res conditions from above
        if (res3_cond_init and res4_cond_init):
            if snr3 >= snr4:
                return (g3, tfr3, N3, snr3, (tfr3 + overhead)*N3)
            if snr3 < snr4:
                return (g4, tfr4, N4, snr4, (tfr4 + overhead)*N4)
        elif res3_cond_init and not res4_cond_init:
            return (g3, tfr3, N3, snr3, (tfr3 + overhead)*N3)
        elif res4_cond_init and not res3_cond_init:
            return (g4, tfr4, N4, snr4, (tfr4 + overhead)*N4)

    if hard_limit:

        # max time from rail constraint, from Eq. 39 of etc_snr_v3b.pdf in
        # doc folder (but including effect of delta_constr; for use below)
        rail_t = (2*alpha0*fwc*(1+delta_constr) + n*(n -
                np.sqrt(4*alpha0*fwc*(1+delta_constr) + n**2)))/(2*
                (fluxe_bright+darke))

        # N3 cases
        if N3_ceil == N3_floor: #then N3 is integer; no adjustment needed
            res3_cond = True #initialize
            N3 = N3_ceil
            tfr3 = res3.x[1]
            snr3 = nowsnr(np.array([g3, tfr3]))

        if N3_ceil != N3_floor:
            res3_cond = True #initialize
            if N3_ceil > Nmax:
                if t_tot/Nmax - overhead > t_ub_gen:
                    # then N*t cannot equal t_tot w/o violating t_ub_gen, Nmax
                    res3_cond = False
                else: #t_tot/Nmax within bounds
                    N3 = Nmax
                    #time already checked at beginning: doesn't violate rail
                    tfr3 = t_tot/N3 - overhead
                    # optimal g would shift slightly b/c of shift in t, so:
                    g3, snr3 = _max_snr_g(tfr3)
            elif N3_ceil <= Nmax and N3_floor >= Nmin:
                N3_up = N3_ceil
                N3_low = N3_floor
                if t_tot/N3_up - overhead < tmin and \
                   t_tot/N3_low - overhead > t_ub_gen:
                    # then N*t cannot equal t_tot w/o violating tmin, t_ub_gen
                    res3_cond = False
                elif t_tot/N3_up - overhead >= tmin and \
                     t_tot/N3_low - overhead > t_ub_gen:
                    if t_tot/N3_up - overhead > rail_t:
                        res3_cond = False
                    else:
                        N3 = N3_up
                        tfr3 = t_tot/N3_up - overhead
                        g3, snr3 = _max_snr_g(tfr3)
                elif t_tot/N3_up - overhead < tmin and \
                     t_tot/N3_low - overhead <= t_ub_gen:
                    if t_tot/N3_low - overhead > rail_t:
                        res3_cond = False
                    else:
                        N3 = N3_low
                        tfr3 = t_tot/N3_low - overhead
                        g3, snr3 = _max_snr_g(tfr3)
                elif t_tot/N3_up - overhead >= tmin and \
                     t_tot/N3_low - overhead <= t_ub_gen:
                    N3_up_cond = True
                    N3_low_cond = True
                    if t_tot/N3_up - overhead > rail_t:
                        N3_up_cond = False
                    else:
                        g3_up, snr3_up = _max_snr_g(t_tot/N3_up - overhead)
                    if t_tot/N3_low - overhead > rail_t:
                        N3_low_cond = False
                    else:
                        g3_low, snr3_low = _max_snr_g(t_tot/N3_low - overhead)
                    if not N3_up_cond and not N3_low_cond:
                        res3_cond = False
                    elif not N3_up_cond and N3_low_cond:
                        N3 = N3_low
                        tfr3 = t_tot/N3_low - overhead
                        snr3 = snr3_low
                    elif N3_up_cond and not N3_low_cond:
                        N3 = N3_up
                        tfr3 = t_tot/N3_up - overhead
                        snr3 = snr3_up
                    elif N3_up_cond and N3_low_cond:
                        if snr3_up >= snr3_low:
                            N3 = N3_up
                            tfr3 = t_tot/N3_up - overhead
                            g3 = g3_up
                            snr3 = snr3_up
                        if snr3_up < snr3_low:
                            N3 = N3_low
                            tfr3 = t_tot/N3_low - overhead
                            g3 = g3_low
                            snr3 = snr3_low
            elif N3_floor < Nmin:
                if t_tot/Nmin - overhead < tmin:
                    # then N*t cannot equal t_tot w/o violating tmin, Nmin
                    res3_cond = False
                else: #t_tot/Nmin within bounds
                    if t_tot/Nmin - overhead > rail_t:
                        res3_cond = False
                    else:
                        N3 = Nmin
                        tfr3 = t_tot/N3 - overhead
                        # optimal g would shift slightly b/c of shift in t, so:
                        g3, snr3 = _max_snr_g(tfr3)

        # N4 cases
        if N4_ceil == N4_floor: #then N4 is integer; no adjustment needed
            res4_cond = True #initialize
            N4 = N4_ceil
            tfr4 = res4.x[1]
            snr4 = nowsnr(np.array([g4, tfr4]))

        if N4_ceil != N4_floor:
            res4_cond = True #initialize
            if N4_ceil > Nmax:
                if t_tot/Nmax - overhead > t_ub_gen:
                    # then N*t cannot equal t_tot w/o violating t_ub_gen, Nmax
                    res4_cond = False
                else: #t_tot/Nmax within bounds
                    N4 = Nmax
                    #time already checked at beginning: doesn't violate rail
                    tfr4 = t_tot/N4 - overhead
                    # optimal g would shift slightly b/c of shift in t, so:
                    g4, snr4 = _max_snr_g(tfr4)
            elif N4_ceil <= Nmax and N4_floor >= Nmin:
                N4_up = N4_ceil
                N4_low = N4_floor
                if t_tot/N4_up - overhead < tmin and \
                   t_tot/N4_low - overhead > t_ub_gen:
                    # then N*t cannot equal t_tot w/o violating tmin, t_ub_gen
                    res4_cond = False
                elif t_tot/N4_up - overhead >= tmin and \
                     t_tot/N4_low - overhead > t_ub_gen:
                    if t_tot/N4_up - overhead > rail_t:
                        res4_cond = False
                    else:
                        N4 = N4_up
                        tfr4 = t_tot/N4_up - overhead
                        g4, snr4 = _max_snr_g(tfr4)
                elif t_tot/N4_up - overhead < tmin and \
                     t_tot/N4_low - overhead <= t_ub_gen:
                    if t_tot/N4_low - overhead > rail_t:
                        res4_cond = False
                    else:
                        N4 = N4_low
                        tfr4 = t_tot/N4_low - overhead
                        g4, snr4 = _max_snr_g(tfr4)
                elif t_tot/N4_up - overhead >= tmin and \
                     t_tot/N4_low - overhead <= t_ub_gen:
                    N4_up_cond = True
                    N4_low_cond = True
                    if t_tot/N4_up - overhead > rail_t:
                        N4_up_cond = False
                    else:
                        g4_up, snr4_up = _max_snr_g(t_tot/N4_up - overhead)
                    if t_tot/N4_low - overhead > rail_t:
                        N4_low_cond = False
                    else:
                        g4_low, snr4_low = _max_snr_g(t_tot/N4_low - overhead)
                    if not N4_up_cond and not N4_low_cond:
                        res4_cond = False
                    elif not N4_up_cond and N4_low_cond:
                        N4 = N4_low
                        tfr4 = t_tot/N4_low - overhead
                        snr4 = snr4_low
                    elif N4_up_cond and not N4_low_cond:
                        N4 = N4_up
                        tfr4 = t_tot/N4_up - overhead
                        snr4 = snr4_up
                    elif N4_up_cond and N4_low_cond:
                        if snr4_up >= snr4_low:
                            N4 = N4_up
                            tfr4 = t_tot/N4_up - overhead
                            g4 = g4_up
                            snr4 = snr4_up
                        if snr4_up < snr4_low:
                            N4 = N4_low
                            tfr4 = t_tot/N4_low - overhead
                            g4 = g4_low
                            snr4 = snr4_low
            elif N4_floor < Nmin:
                if t_tot/Nmin - overhead < tmin:
                    # then N*t cannot equal t_tot w/o violating tmin, Nmin
                    res4_cond = False
                else: #t_tot/Nmin within bounds
                    if t_tot/Nmin - overhead > rail_t:
                        res4_cond = False
                    else:
                        N4 = Nmin
                        tfr4 = t_tot/N4 - overhead
                        # optimal g would shift slightly b/c of shift in t, so:
                        g4, snr4 = _max_snr_g(tfr4)

        #constraints met (since integer N was fully specified in above, we now
        # give the N constraints below no wiggle room)
        res3_cond_final = (
        res3_cond and
        (rail(np.array([g3, tfr3]))/(alpha0*fwc) <= 1+delta_constr) and
        (em_rail(np.array([g3, tfr3]))/(alpha1*fwc_em) <= 1+delta_constr) and
        (N3/Nmin >= 1) and
        (N3/Nmax <= 1)
                    )

        #constraints met (since integer N was fully specified in above, we now
        # give the N constraints below no wiggle room)
        res4_cond_final = (
        res4_cond and
        (rail(np.array([g4, tfr4]))/(alpha0*fwc) <= 1+delta_constr) and
        (em_rail(np.array([g4, tfr4]))/(alpha1*fwc_em) <= 1+delta_constr) and
        (N4/Nmin >= 1) and
        (N4/Nmax <= 1)
                    )

        if (res3_cond_final and res4_cond_final):
            if snr3 >= snr4:
                return (g3, tfr3, N3, snr3, (tfr3 + overhead)*N3)
            if snr3 < snr4:
                return (g4, tfr4, N4, snr4, (tfr4 + overhead)*N4)
        elif res3_cond_final and not res4_cond_final:
            return (g3, tfr3, N3, snr3, (tfr3 + overhead)*N3)
        elif res4_cond_final and not res3_cond_final:
            return (g4, tfr4, N4, snr4, (tfr4 + overhead)*N4)

        raise EXCAMOptimizeException('Optimization failed, cannot produce ' +
                                    'camera settings for ' +
                                    'hard_limit=True.')


if __name__ == "__main__":
    # testing only
    ap = argparse.ArgumentParser(prog='python camera.py',
                                 description="Compute EXCAM settings")
    ap.add_argument('--snr', default=7, help="Target SNR.  Default = 7.",
                    type=float)
    ap.add_argument('--flux', default=100, type=float,
                    help="Electrons from photon flux per second. Default=100.")
    args = ap.parse_args()


    out = calc_gain_exptime(target_snr=2, #183.54790086591134, #args.snr,
                            fluxe=1000,#1e-8,
                            fluxe_bright=5500,#1e-6,
                            darke=8.33e-4,  # inputs from excam_properties.yaml
                            cic=0.02,
                            rn=100,
                            X=5e4,
                            a=1.69e-10,
                            Lij=512,
                            alpha0=0.75,
                            fwc=60000,
                            alpha1=0.75,
                            fwc_em=100000,
                            Nmin=1,
                            Nmax=20,#25200,#49,
                            tmin=0.5,
                            tmax=6300.,
                            gmax=5000,
                            overhead=3,
                            opt_choice=0,
                            n=4,
                            Nem=604,
                            tol=1e-30,
                            delta_constr=1e-4,
                            num_pixels=5
                            )
    print('calc_gain_exptime: ', out)

    out2 = calc_gain_fixed_time(target_snr=4.4, #183.54790086591134, #args.snr,
                            fluxe=0.4,
                            fluxe_bright=8,
                            darke=8.33e-4,  # inputs from excam_properties.yaml
                            cic=0.02,
                            rn=200,
                            X=5e4,
                            a=1.69e-10,
                            Lij=512,
                            alpha0=0.75,
                            fwc=60000,
                            alpha1=0.75,
                            fwc_em=100000,
                            Nmin=1,
                            Nmax=2,
                            t=1,
                            gmax=5000,
                            overhead=3,
                            opt_choice=0,
                            n=4,
                            Nem=604,
                            tol=1e-30,
                            delta_constr=1e-4,
                            num_pixels=5
                            )
    print('calc_gain_fixed_time: ', out2)

    out3 = calc_gain_fixed_N(target_snr=2, #183.54790086591134, #args.snr,
                            fluxe=.1,#1e-8,
                            fluxe_bright=1,#1e-6,
                            darke=8.33e-4,  # inputs from excam_properties.yaml
                            cic=0.02,
                            rn=100,
                            X=5e4,
                            a=1.69e-10,
                            Lij=512,
                            alpha0=0.75,
                            fwc=60000,
                            alpha1=0.75,
                            fwc_em=100000,
                            N=20,
                            tmin=0.5,
                            tmax=6300.,
                            gmax=5000,
                            overhead=3,
                            opt_choice=0,
                            n=4,
                            Nem=604,
                            tol=1e-30,
                            delta_constr=1e-4,
                            num_pixels=5
                            )
    print('calc_gain_fixed_N: ', out3)

    out4 = calc_gain_fixed_g(target_snr=2, #183.54790086591134, #args.snr,
                            fluxe=.1,#1e-8,
                            fluxe_bright=1,#1e-6,
                            darke=8.33e-4,  # inputs from excam_properties.yaml
                            cic=0.02,
                            rn=100,
                            X=5e4,
                            a=1.69e-10,
                            Lij=512,
                            alpha0=0.75,
                            fwc=60000,
                            alpha1=0.75,
                            fwc_em=100000,
                            Nmin=1,
                            Nmax=20,#25200,#49,
                            tmin=0.5,
                            tmax=6300.,
                            g=3581.039, #236.978,
                            overhead=3,
                            opt_choice=0,
                            n=4,
                            Nem=604,
                            tol=1e-30,
                            delta_constr=1e-4,
                            num_pixels=5
                            )
    print('calc_fixed_g: ', out4)

    out5 = calc_gain_fixed_Ntime(t_tot=400,
                            fluxe=.01,
                            fluxe_bright=1,
                            darke=8.33e-4,
                            cic=0.02,
                            rn=120,#160,#100,
                            X=5e4,
                            a=1.69e-10,
                            Lij=512,
                            alpha0=0.75,
                            fwc=50000,#60000,#50000, #60000,
                            alpha1=0.75,
                            fwc_em=90000,#53000, #90000,#100000,
                            Nmin=1,#20,
                            Nmax=49,#25200,#30, #int(1e12), #2475,
                            tmin=1,#0.6,#0.264,#1,
                            tmax=120,#120,#6300.,
                            gmax=5000,
                            overhead=3,
                            n=4, #0,
                            Nem=604,
                            tol=1e-30,
                            delta_constr=1e-4, #1e-7,
                            hard_limit=True,
                            num_pixels=1
                            )
    print('calc_gain_fixed_Ntime:  ', out5)

    out6 = calc_pc(target_snr=0.3,
                fluxe=0.002225854406660047,#.000178068,#1e-8,
                fluxe_bright=0.0027056017118759555,#.0027056,#1e-6,
                darke=8.33e-4,  # inputs from excam_properties.yaml
                cic=0.02,
                rn=100,
                X=5e4,
                a=1.69e-10,
                Lij=512,
                alpha0=0.75,
                fwc=60000,
                alpha1=0.75,
                fwc_em=100000,
                Nmin=1,
                Nmax=25200, #8 rolls, 3150 frames per roll for min time of 2s
                tmin=0,
                tmax=6300,
                gmax=5000,
                overhead=3,
                pc_ecount_max=0.1,
                T_factor=5,
                opt_choice=1,
                n=4,
                Nem=604,
                tol=1e-30,
                delta_constr=1e-4,
                num_pixels=1
                )
    print('calc_pc: ', out6)

    out7 = calc_pc_fixed_N(target_snr=0,
                fluxe=0.01,#1e-8,
                fluxe_bright=1,#1e-6,
                darke=8.33e-4,  # inputs from excam_properties.yaml
                cic=0.02,
                rn=100,
                X=5e4,
                a=1.69e-10,
                Lij=512,
                alpha0=0.75,
                fwc=60000,
                alpha1=0.75,
                fwc_em=100000,
                N=1,
                tmin=0,
                tmax=6300,
                gmax=5000,
                overhead=3,
                pc_ecount_max=0.1,
                T_factor=5,
                opt_choice=0,
                n=4,
                Nem=604,
                tol=1e-30,
                delta_constr=1e-4,
                num_pixels=5
                )
    print('calc_pc_fixed_N: ', out7)

    out8 = calc_pc_gain_fixed_Ntime(t_tot=3600,
                            fluxe=.01,
                            fluxe_bright=1,
                            darke=8.33e-4,
                            cic=0.02,
                            rn=100,#160,#100,
                            X=5e4,
                            a=1.69e-10,
                            Lij=512,
                            alpha0=0.75,
                            fwc=60000,#60000,#50000, #60000,
                            alpha1=0.75,
                            fwc_em=100000,#53000, #90000,#100000,
                            Nmin=1,#20,
                            Nmax=25200,#30, #int(1e12), #2475,
                            tmin=1,#0.6,#0.264,#1,
                            tmax=6300,#120,#6300.,
                            gmax=5000,
                            overhead=3,
                            pc_ecount_max=0.1,
                            T_factor=5,
                            n=4, #0,
                            Nem=604,
                            tol=1e-30,
                            delta_constr=1e-4, #1e-7,
                            hard_limit=True,
                            num_pixels=1
                            )
    print('calc_pc_gain_fixed_Ntime: ', out8)
    pass
