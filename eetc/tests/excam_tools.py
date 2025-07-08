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
"""

import argparse
import warnings

import numpy as np
import scipy.optimize

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


def _SNR_CR(g, tfr, N, fluxe, darke, cic, rn, X, a, Lij, sign, Nem=604):
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

    Nem : int, optional
        number of gain multiplying elements.  Defaults to 604, which is the
        right hardware number for CGI cameras and very unlikely to change.

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


def calc_gain_exptime(target_snr, fluxe, fluxe_bright, darke, cic, rn, X, a,
                      Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax, tmin, tmax,
                      gmax, opt_choice=0, n=4, Nem=604, tol=1e-30,
                      delta_constr=1e-4, **kwargs):
    """
    Run 1-2 optimizations to find the best EXCAM settings for the next
    iteration.  If you don't have a target SNR and simply want to maximize the
    SNR, you may specify opt_choice=1 to skip to the 2nd optimizer.

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
        SNR in a pixel. >= 0.

    fluxe : float
        target flux, in electrons per second (i.e. photon flux (phi) * QE
        (eta)), in a pixel.  >= 0.

    fluxe_bright : float
        flux maximum, in electrons per second (i.e. photon flux (phi) * QE
        (eta)), in a pixel.  >= 0.

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
        cause that target pixel to be made useless if the other pixel is hit.
        >= 1.

    alpha0 : float
        fraction of the per-pixel full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc : int
        number of electrons in the per-pixel full well.  >= 1.

    alpha1 : float
        fraction of the EM gain full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc_em : int
        number of electrons in the EM gain full well.  >= 1.

    Nmin : int
        minimum number of allowed exposures.  >= 1.

    Nmax : int
        maximum number of allowed exposures.  >= 1.  Must be >= Nmin.

    tmin : float
        minimum exposure time length, in seconds/frame.  >= 0.

    tmax : float
        maximum exposure time length, in seconds/frame.  >= 0. Must be >= tmin.

    gmax : float
        maximum permitted gain.  >= 1.  (Minimum gain is 1.)

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
        tolerance level used used by optimizations.  >=0.  Recommended to be
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

    Returns
    -------
    gain : float
        Optimal gain setting for detector

    exptime : float
        Exposure time for each frame in 'n_frames' to reach 'snr'

    n_frames : float
        Total number of exposures to reach 'snr'

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
    check.real_nonnegative_scalar(alpha0, 'alpha0', TypeError)
    if alpha0 > 1:
        raise ValueError('alpha0 must be between 0 and 1 (fraction of total)')
    check.positive_scalar_integer(fwc, 'fwc', TypeError)
    check.real_nonnegative_scalar(alpha1, 'alpha1', TypeError)
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
    check.real_positive_scalar(gmax, 'gmax', TypeError)
    if gmax < 1:
        raise ValueError('gmax must be >= 1')
    check.nonnegative_scalar_integer(opt_choice, 'opt_choice', TypeError)
    if ((opt_choice != 0) and (opt_choice != 1)):
        raise ValueError('opt_choice must be 0 or 1')
    check.real_nonnegative_scalar(n, 'n', TypeError)
    check.positive_scalar_integer(Nem, 'Nem', TypeError)
    check.real_nonnegative_scalar(tol, "tol", TypeError)
    check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)

    # Check that there are no unsatisfiable constraints
    # minimal case for rail constraint (parallel CIC negligible):
    if (fluxe_bright*tmin + darke*tmin) + \
        n*np.sqrt(fluxe_bright*tmin + darke*tmin) > alpha0*fwc:
        return (0,0,0,0,3)
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')

    _gmin = 1

    # in case alpha1*fwc_em is input as smaller than alpha0*fwc:
    #minimal case for em_rail constraint, serial cic ~ total cic:
    if _gmin*(fluxe_bright*tmin + darke*tmin + cic) + n*_ENF(_gmin,Nem)*_gmin*\
        np.sqrt(fluxe_bright*tmin + darke*tmin + cic) > alpha1*fwc_em:
        return (0,0,0,0,3)
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')
    Nmax = np.min(np.array([Nmax, 8*6300/2]))
    # [g, tfr, N] = v
    bounds = scipy.optimize.Bounds(lb=np.array([_gmin, tmin, Nmin]),
                                ub=np.array([gmax, tmax, Nmax]))

    em_rail = lambda v: v[0]*(fluxe_bright*v[1] + darke*v[1] + cic) + (
                        n*_ENF(v[0], Nem)*v[0]*np.sqrt(fluxe_bright*v[1] +
                        darke*v[1] + cic)
                        )
    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1*fwc_em)

    nowsnr = lambda v: _SNR_CR(v[0], v[1], v[2], fluxe, darke, cic, rn, X, a,
                        Lij, sign=1)

    nconst2 = scipy.optimize.NonlinearConstraint(nowsnr,
                                        target_snr, np.inf)

    rail = lambda v: fluxe_bright*v[1] + darke*v[1] + (
                    n*np.sqrt(fluxe_bright*v[1] + darke*v[1])
                    )
    nconst3 = scipy.optimize.NonlinearConstraint(rail, 0, alpha0*fwc)

    t_tot = lambda v: v[1]*v[2]

    nconst5 = scipy.optimize.NonlinearConstraint(t_tot, 0, 6300*8)

    _tmp_opt_choice = 0 #used to go to 2nd optimization if first fails


    if opt_choice == 0:
        FOM = lambda v: v[1]*v[2] # no gain in wall-clock time

        res1 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([_gmin, tmin, Nmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3,
                                    nconst5),
                                    )

        #same thing, but starting point of gmax
        res2 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([gmax, tmin, Nmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3,
                                    nconst5),
                                    )

        g1 = res1.x[0]
        tfr1 = res1.x[1]
        # N1 = int(np.ceil(res1.x[2]))
        # if  N1*tfr1/(8*6300) > 1+delta_constr:
        #     N1 = N1 - 1
        N1 = res1.x[2]
        g2 = res2.x[0]
        tfr2 = res2.x[1]
        # N2 = int(np.ceil(res2.x[2]))
        # if N2*tfr2/(8*6300) > 1+delta_constr:
        #     N2 = N2 - 1
        N2 = res2.x[2]
        snr1 = nowsnr(np.array([g1, tfr1, N1]))
        snr2 = nowsnr(np.array([g2, tfr2, N2]))

        #constraints met
        res1_cond = (
         (rail(np.array([g1, tfr1]))/(alpha0*fwc) <= 1+delta_constr) and
         (em_rail(np.array([g1, tfr1]))/(alpha1*fwc_em) <= 1+delta_constr) and
         (snr1/target_snr >= 1-delta_constr) and
        (N1*tfr1/(8*6300) <= 1+delta_constr)
                    )

        res2_cond = (
         (rail(np.array([g2, tfr2]))/(alpha0*fwc) <= 1+delta_constr) and
         (em_rail(np.array([g2, tfr2]))/(alpha1*fwc_em) <= 1+delta_constr) and
         (snr2/target_snr >= 1-delta_constr) and
        (N2*tfr2/(8*6300) <= 1+delta_constr)
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

        def _SNR_CR1(v, fluxe, darke, cic, rn, X, a, Lij, sign, Nem=604):
            g, tfr, N = v
            return _SNR_CR(g, tfr, N, fluxe, darke, cic, rn, X, a, Lij, sign,
            Nem=604)

        res3 = scipy.optimize.minimize(fun=_SNR_CR1,
                                   x0=np.array([_gmin, tmin, Nmin]),
                                   args=(fluxe, darke, cic, rn, X, a, Lij, -1),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3, nconst5),
                                   )


        #same thing, but starting point of gmax
        res4 = scipy.optimize.minimize(fun=_SNR_CR1,
                                   x0=np.array([gmax, tmin, Nmin]),
                                   args=(fluxe, darke, cic, rn, X, a, Lij, -1),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3, nconst5),
                                   )

        g3 = res3.x[0]
        tfr3 = res3.x[1]
        N3 = int(np.ceil(res3.x[2]))
        if N3*tfr3/(8*6300) > 1+delta_constr:
            N3 = N3 - 1
        #N3 = res3.x[2]
        g4 = res4.x[0]
        tfr4 = res4.x[1]
        N4 = int(np.ceil(res4.x[2]))
        if N4*tfr4/(8*6300) > 1+delta_constr:
            N4 = N4 - 1
        #N4 = res4.x[2]
        snr3 = nowsnr(np.array([g3, tfr3, N3]))
        snr4 = nowsnr(np.array([g4, tfr4, N4]))

        #constraints met
        res3_cond = (
         (rail(np.array([g3, tfr3]))/(alpha0*fwc) <= 1+delta_constr) and
         (em_rail(np.array([g3, tfr3]))/(alpha1*fwc_em) <= 1+delta_constr) and
        (N3*tfr3/(8*6300) <= 1+delta_constr)
                    )

        res4_cond = (
         (rail(np.array([g4, tfr4]))/(alpha0*fwc) <= 1+delta_constr) and
         (em_rail(np.array([g4, tfr4]))/(alpha1*fwc_em) <= 1+delta_constr) and
        (N4*tfr4/(8*6300) <= 1+delta_constr)
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
    return (0,0,0,0,3)
    raise EXCAMOptimizeException('Both optimizations failed, cannot produce ' +
                                 'camera settings')

def calc_gain_fixed_time(target_snr, fluxe, fluxe_bright, darke, cic, rn, X, a,
                      Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax, t,
                      gmax, opt_choice=0, n=4, Nem=604, tol=1e-30,
                      delta_constr=1e-4, **kwargs):
    """
    Runs 1-2 optimizations to find the best EXCAM settings for a given
    frame exposure time (t) for the next iteration.  If you don't have a target
    SNR and simply want to maximize the SNR, you may specify opt_choice=1 to
    skip to the 2nd optimizer.

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
        SNR in a pixel. >= 0.

    fluxe : float
        target flux, in electrons per second (i.e. photon flux (phi) * QE
        (eta)), in a pixel.  >= 0.

    fluxe_bright : float
        flux maximum, in electrons per second (i.e. photon flux (phi) * QE
        (eta)), in a pixel.  >= 0.

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
        cause that target pixel to be made useless if the other pixel is hit.
        >= 1.

    alpha0 : float
        fraction of the per-pixel full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc : int
        number of electrons in the per-pixel full well.  >= 1.

    alpha1 : float
        fraction of the EM gain full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.

    fwc_em : int
        number of electrons in the EM gain full well.  >= 1.

    Nmin : int
        minimum number of allowed exposures.  >= 1.

    Nmax : int
        maximum number of allowed exposures.  >= 1.  Must be >= Nmin.

    t : float
        fixed time length for all frames, in seconds/frame.  >= 0.

    gmax : float
        maximum permitted gain.  >= 1.  (Minimum gain is 1.)

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
        tolerance level used used by optimizations.  >=0.  Recommended to be
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

    Returns
    -------
    gain : float
        Optimal gain setting for detector

    exptime : float
        Exposure time for each frame in 'n_frames' to reach 'snr' (same as t)

    n_frames : float
        Total number of exposures to reach 'snr'

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
    check.real_nonnegative_scalar(alpha0, 'alpha0', TypeError)
    if alpha0 > 1:
        raise ValueError('alpha0 must be between 0 and 1 (fraction of total)')
    check.positive_scalar_integer(fwc, 'fwc', TypeError)
    check.real_nonnegative_scalar(alpha1, 'alpha1', TypeError)
    if alpha1 > 1:
        raise ValueError('alpha1 must be between 0 and 1 (fraction of total)')
    check.positive_scalar_integer(fwc_em, 'fwc_em', TypeError)
    check.positive_scalar_integer(Nmin, 'Nmin', TypeError)
    check.positive_scalar_integer(Nmax, 'Nmax', TypeError)
    if Nmax < Nmin:
        raise ValueError('Nmax must be >= Nmin')
    check.real_nonnegative_scalar(t, 't', TypeError)
    check.real_positive_scalar(gmax, 'gmax', TypeError)
    if gmax < 1:
        raise ValueError('gmax must be >= 1')
    check.nonnegative_scalar_integer(opt_choice, 'opt_choice', TypeError)
    if ((opt_choice != 0) and (opt_choice != 1)):
        raise ValueError('opt_choice must be 0 or 1')
    check.real_nonnegative_scalar(n, 'n', TypeError)
    check.positive_scalar_integer(Nem, 'Nem', TypeError)
    check.real_nonnegative_scalar(tol, "tol", TypeError)
    check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)

    # Check that there are no unsatisfiable constraints
    # minimal case for rail constraint (parallel CIC negligible):
    if (fluxe_bright*t + darke*t) + \
        n*np.sqrt(fluxe_bright*t + darke*t) > alpha0*fwc:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')
    _gmin = 1

    # in case alpha1*fwc_em is input as smaller than alpha0*fwc:
    #minimal case for em_rail constraint, serial cic ~ total cic:
    if _gmin*(fluxe_bright*t + darke*t + cic) + n*_ENF(_gmin,Nem)*_gmin*\
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
    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1*fwc_em)

    nowsnr = lambda v: _SNR_CR(v[0], t, v[1], fluxe, darke, cic, rn, X, a, Lij,
                        sign=1)
    nconst2 = scipy.optimize.NonlinearConstraint(nowsnr, target_snr, np.inf)

    _tmp_opt_choice = 0 #used to go to 2nd optimization if first fails

    if opt_choice == 0:
        FOM = lambda v: t*v[1] # no gain in wall-clock time

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
         (em_rail(np.array([g1]))/(alpha1*fwc_em) <= 1+delta_constr) and
         (snr1/target_snr >= 1-delta_constr)
                    )

        res2_cond = (
         (em_rail(np.array([g2]))/(alpha1*fwc_em) <= 1+delta_constr) and
         (snr2/target_snr >= 1-delta_constr)
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

        def _SNR_CR2(v, t, fluxe, darke, cic, rn, X, a, Lij, sign, Nem=604):
            g, N = v
            return _SNR_CR(g, t, N, fluxe, darke, cic, rn, X, a, Lij, sign,
            Nem=604)

        res3 = scipy.optimize.minimize(fun=_SNR_CR2,
                                   x0=np.array([_gmin, Nmin]),
                                   args=(t ,fluxe, darke, cic, rn, X, a, Lij,
                                        -1),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1),
                                   )


        #same thing, but starting point of gmax
        res4 = scipy.optimize.minimize(fun=_SNR_CR2,
                                   x0=np.array([gmax, Nmin]),
                                   args=(t, fluxe, darke, cic, rn, X, a, Lij,
                                        -1),
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
         (em_rail(np.array([g3]))/(alpha1*fwc_em) <= 1+delta_constr)
                    )

        res4_cond = (
         (em_rail(np.array([g4]))/(alpha1*fwc_em) <= 1+delta_constr)
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

def _e_coinloss(L):
    """
    Computes the coincidence loss factor for photon counting.  (Only used in
    _SNR_CR_pc.)

    Parameters
    ----------
    L : float
        mean number of electrons per frame per pixel (in e-).

    Returns
    -------
    float
        coincidence loss factor

    """
    return (1 - np.exp(-L)) / L

def var(g, L, T, N):
    """
    Computes the variance for the photon-counting SNR.  (Only used in
    _SNR_CR_pc.)  The calculation assumes the mean number of electrons per
    pixel per frame (in e-) is relatively small (for a 3rd-order
    Taylor expansion), as it should be for photon counting, and it assumes
    2 iterations of photometric corrections as is done in PhotonCounting.

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
    return (np.e**(-5*L + T/g)*(-1 + np.e**L)*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)*
        (16*np.e**(4*(L + T/g))*g**8*(6 + L*(3 + L))**4*(4*g**2 - 8*g*T + 5*T**2)**2 - 240*np.e**(3*(L + T/g))*g**6*(6 + L*(3 + L))**3*N*(4*g**2 - 8*g*T + 5*T**2)**2*
            (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2) + 336*np.e**(2*(L + T/g))*g**4*(6 + L*(3 + L))**2*N**2*(4*g**2 - 8*g*T + 5*T**2)**2*
            (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**2 - 108*np.e**(L + T/g)*g**2*(6 + L*(3 + L))*N**3*(4*g**2 - 8*g*T + 5*T**2)**2*
            (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**3 + 9*N**4*(4*g**2 - 8*g*T + 5*T**2)**2*
            (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**4 - 36*np.e**L*N**4*(4*g**2 - 8*g*T + 5*T**2)*
            (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**3*(12*g**4*(6 + L*(3 + L)) - 12*g**3*(10 + L*(3 + L))*T - 2*g**2*(-30 + L*(9 + L))*T**2 +
            2*g*L*(15 + L)*T**3 + 5*L**2*T**4) - 672*np.e**(3*L + (2*T)/g)*g**4*(6 + L*(3 + L))**2*N**2*(4*g**2 - 8*g*T + 5*T**2)*
            (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)*(10*g**4*(6 + L*(3 + L)) - 2*g**3*(54 + 5*L*(3 + L))*T - 2*g**2*(-30 + L*(9 + L))*T**2 +
            2*g*L*(15 + L)*T**3 + 5*L**2*T**4) + 48*np.e**(4*L + (3*T)/g)*g**6*(6 + L*(3 + L))**3*N*(4*g**2 - 8*g*T + 5*T**2)*
            (44*g**4*(6 + L*(3 + L)) - 4*g**3*(126 + 11*L*(3 + L))*T - 10*g**2*(-30 + L*(9 + L))*T**2 + 10*g*L*(15 + L)*T**3 + 25*L**2*T**4) +
        36*np.e**(2*L + T/g)*g**2*(6 + L*(3 + L))*N**3*(4*g**2 - 8*g*T + 5*T**2)*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**2*
            (100*g**4*(6 + L*(3 + L)) - 4*g**3*(258 + 25*L*(3 + L))*T - 18*g**2*(-30 + L*(9 + L))*T**2 + 18*g*L*(15 + L)*T**3 + 45*L**2*T**4) -
        36*np.e**(3*L)*N**4*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)*(12*g**4*(6 + L*(3 + L)) - 12*g**3*(10 + L*(3 + L))*T -
            2*g**2*(-30 + L*(9 + L))*T**2 + 2*g*L*(15 + L)*T**3 + 5*L**2*T**4)*(48*g**6*(6 + L*(3 + L))**2 - 288*g**5*(6 + L*(3 + L))*T -
            4*g**4*(-180 + L*(180 + L*(123 + L*(48 + 5*L))))*T**2 - 8*g**3*L*(-90 + L*(3 + L)*(-3 + 2*L))*T**3 + 12*g**2*L**2*(25 + L*(7 + L))*T**4 +
            12*g*L**3*(5 + L)*T**5 + 5*L**4*T**6) + 9*np.e**(4*L)*N**4*(48*g**6*(6 + L*(3 + L))**2 - 288*g**5*(6 + L*(3 + L))*T -
            4*g**4*(-180 + L*(180 + L*(123 + L*(48 + 5*L))))*T**2 - 8*g**3*L*(-90 + L*(3 + L)*(-3 + 2*L))*T**3 + 12*g**2*L**2*(25 + L*(7 + L))*T**4 +
            12*g*L**3*(5 + L)*T**5 + 5*L**4*T**6)**2 + 18*np.e**(2*L)*N**4*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**2*
            (480*g**8*(6 + L*(3 + L))**2 - 192*g**7*(6 + L*(3 + L))*(48 + 5*L*(3 + L))*T + 32*g**6*(2232 + L*(1044 + L*(420 + L*(39 + 11*L))))*T**2 +
            288*g**5*(-150 + L*(45 + L*(29 + L*(15 + L))))*T**3 + 12*g**4*(900 + L*(-2340 + L*(-459 + L*(-108 + 19*L))))*T**4 -
            24*g**3*L*(-450 + L*(255 + L*(69 + 16*L)))*T**5 - 12*g**2*L**2*(-375 + L*(15 + 4*L))*T**6 + 60*g*L**3*(15 + L)*T**7 + 75*L**4*T**8) +
        48*np.e**(4*L + (2*T)/g)*g**4*(6 + L*(3 + L))**2*N**2*(716*g**8*(6 + L*(3 + L))**2 - 8*g**7*(6 + L*(3 + L))*(1914 + 179*L*(3 + L))*T +
            12*g**6*(11076 + L*(4692 + L*(1757 + L*(82 + 37*L))))*T**2 + 112*g**5*(-810 + L*(243 + L*(147 + 5*L*(15 + L))))*T**3 +
            28*g**4*(900 + L*(-2160 + L*(-387 + L*(-87 + 16*L))))*T**4 - 84*g**3*L*(-300 + L*(160 + L*(41 + 9*L)))*T**5 -
            28*g**2*L**2*(-375 + L*(15 + 4*L))*T**6 + 140*g*L**3*(15 + L)*T**7 + 175*L**4*T**8) - 36*np.e**(3*L + T/g)*g**2*(6 + L*(3 + L))*N**3*
            (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)*(1200*g**8*(6 + L*(3 + L))**2 - 2400*g**7*(6 + L*(3 + L))*(10 + L*(3 + L))*T +
            32*g**6*(6084 + L*(2736 + L*(1071 + L*(81 + 26*L))))*T**2 + 32*g**5*(-3870 + L*(1161 + L*(729 + 25*L*(15 + L))))*T**3 +
            12*g**4*(2700 + L*(-6780 + L*(-1281 + L*(-296 + 53*L))))*T**4 - 8*g**3*L*(-4050 + L*(2235 + L*(591 + 134*L)))*T**5 -
            36*g**2*L**2*(-375 + L*(15 + 4*L))*T**6 + 180*g*L**3*(15 + L)*T**7 + 225*L**4*T**8) + 36*np.e**(4*L + T/g)*g**2*(6 + L*(3 + L))*N**3*
            (1248*g**10*(6 + L*(3 + L))**3 - 96*g**9*(6 + L*(3 + L))**2*(228 + 13*L*(3 + L))*T -
            16*g**8*(6 + L*(3 + L))*(-9828 + L*(-828 + L*(417 + L*(537 + 52*L))))*T**2 +
            16*g**7*(-33480 + L*(22788 + L*(26406 + L*(16011 + L*(4011 + L*(657 + 23*L))))))*T**3 +
            24*g**6*(5400 + L*(-22860 + L*(-6294 + L*(393 + L*(1843 + 13*L*(37 + 5*L))))))*T**4 -
            24*g**5*L*(-8100 + L*(8580 + L*(4809 + L*(1613 + L*(109 + L)))))*T**5 - 4*g**4*L**2*(-32400 + L*(4950 + L*(5307 + L*(1971 + 166*L))))*T**6 +
            4*g**3*L**3*(12150 + L*(1950 - L*(51 + 95*L)))*T**7 + 6*g**2*L**4*(1800 + L*(405 + 37*L))*T**8 + 30*g*L**5*(45 + 7*L)*T**9 + 75*L**6*T**10)))/(4608*g**14*(6 + L*(3 + L))**5*N**5)


def _SNR_CR_pc(g, tfr, N, fluxe, darke, cic, T, X, a, Lij, sign):
    """
    Computes the photon-counting SNR given camera settings and noise
    properties, including cosmic ray effects.  This also assumes that
    photon-counted dark frames are subtracted.

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
    # numerator = L - L_dk:
    numerator = fluxe*tfr
    # lambda; serial cic treated as avg charge present before gain,
    # so included here
    L = fluxe*tfr + darke*tfr + cic
    L_dk = darke*tfr + cic
    # didn't originally account for cosmics in SNR formulation, so do that here
    N_CR = N*np.e**(-X*a*tfr*Lij)
    denominator = np.sqrt(var(g, L, T, N_CR)*_e_coinloss(L)**2 +
    var(g, L_dk, T, N_CR)*_e_coinloss(L_dk)**2)
    #denominator = np.sqrt(var(g, L, T, N_CR)*_e_coinloss(L)**2)
    snr = numerator/denominator
    return sign*snr

def calc_pc(target_snr, fluxe, fluxe_bright, darke, cic, rn, X, a,
                      Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax,
                      tmin, tmax, gmax, pc_ecount_max, fault, fluxe_bim=None,
                      opt_choice=0, n=4, Nem=604, tol=1e-30, delta_constr=1e-4,
                       **kwargs):
    """
    XXX change doc string and add to etc_snr_v3b pdf? Include in doc string that function assumes subtraction of photon-counted darks and that the number of frames < number used for dark calibration
    XXX add max roll time (6300s) and fault (~250000?) to excam_config.yaml, adjust ut
    XXX tmin set to 0.001 in excam_config
    Finds the best EXCAM settings for photon counting
    for a given average photon flux and peak photon flux for the next
    iteration.  It raises an exception if inputs are not conducive to photon
    counting.  It then finds

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
        SNR in a pixel. >= 0.
    fluxe : float
        target flux, in electrons per second (i.e. photon flux (phi) * QE
        (eta)), in a pixel.  >= 0.
    fluxe_bright : float
        flux maximum, in electrons per second (i.e. photon flux (phi) * QE
        (eta)), in a pixel.  >= 0.
    fluxe_bim:  float
        flux maximum for overall image, in electrons per second (i.e.,
        photon flux (phi) * QE (eta)), in a pixel.  >=0.
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
        cause that target pixel to be made useless if the other pixel is hit.
        >= 1.
    alpha0 : float
        fraction of the per-pixel full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.
    fwc : int
        number of electrons in the per-pixel full well.  >= 1.
    alpha1 : float
        fraction of the EM gain full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.
    fwc_em : int
        number of electrons in the EM gain full well.  >= 1.
    Nmin : int
        minimum number of allowed exposures.  >= 1.
    Nmax : int
        maximum number of allowed exposures.  >= 1.  Must be >= Nmin.
    tmin : float
        minimum exposure time length, in seconds/frame.  >= 0.
    tmax : float
        maximum exposure time length, in seconds/frame.  >= 0. Must be >= tmin.
    gmax : float
        maximum permitted gain.  >= 1.  (Minimum gain is 1.)
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
        tolerance level used used by optimizations.  >=0.  Recommended to be
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
    hard limit: boolean, optional
        True if a hard limit needed for t_tot.  Defaults to True.
    fault: int, optional
        Fault protection value, in electrons.  Defaults to 250,000 electrons.
    Returns
    -------
    gain : float
        Optimal gain setting for detector
    exptime : float
        Exposure time for each frame in 'n_frames' to reach 'snr'
    n_frames : float
        Total number of exposures to reach 'snr'
    snr_out : float
        Expected SNR when using above exposure settings
    optflag : int
        0 or 1: 0 if the first optimization succeeded, 1 if the first failed
        but the second succeeded.
    """
    if fluxe_bim == None:
        fluxe_bim = fluxe_bright
    # Check inputs
    check.real_nonnegative_scalar(target_snr, 'target_snr', TypeError)
    check.real_nonnegative_scalar(fluxe, 'fluxe', TypeError)
    check.real_nonnegative_scalar(fluxe_bright, 'fluxe_bright', TypeError)
    if fluxe_bright < fluxe:
        raise ValueError('fluxe_bright must be greater than or equal to fluxe')
    check.real_nonnegative_scalar(fluxe_bim, 'fluxe_bim', TypeError)
    if fluxe_bim < fluxe_bright:
        raise ValueError('fluxe_bim must be greater than or equal to '
        'fluxe_bright')
    check.real_nonnegative_scalar(darke, 'darke', TypeError)
    check.real_nonnegative_scalar(cic, 'cic', TypeError)
    check.real_nonnegative_scalar(rn, 'rn', TypeError)
    check.real_nonnegative_scalar(X, 'X', TypeError)
    check.real_nonnegative_scalar(a, 'a', TypeError)
    check.positive_scalar_integer(Lij, 'Lij', TypeError)
    check.real_scalar(alpha0, 'alpha0', TypeError)
    if alpha0 < 0 or alpha0 > 1:
        raise ValueError('alpha0 must be between 0 and 1 (fraction of total)')
    check.positive_scalar_integer(fwc, 'fwc', TypeError)
    check.real_scalar(alpha1, 'alpha1', TypeError)
    if alpha1 < 0 or alpha1 > 1:
        raise ValueError('alpha1 must be between 0 and 1 (fraction of total)')
    check.positive_scalar_integer(fwc_em, 'fwc_em', TypeError)
    check.positive_scalar_integer(fault, 'fault', TypeError)
    if fault < fwc_em:
        raise ValueError('fault protection value must be bigger than or equal '
        'to fwc_em')
    check.positive_scalar_integer(Nmin, 'Nmin', TypeError)
    check.positive_scalar_integer(Nmax, 'Nmax', TypeError)
    if Nmax < Nmin:
        raise ValueError('Nmax must be >= Nmin')
    check.real_nonnegative_scalar(tmin, 'tmin', TypeError)
    check.real_nonnegative_scalar(tmax, 'tmax', TypeError)
    if tmax < tmin:
        raise ValueError('tmax must be >= tmin')
    check.real_scalar(gmax, 'gmax', TypeError)
    if gmax < 1:
        raise ValueError('gmax must be >= 1')
    check.positive_scalar_integer(Nem, 'Nem', TypeError)
    check.nonnegative_scalar_integer(opt_choice, 'opt_choice', TypeError)
    if ((opt_choice != 0) and (opt_choice != 1)):
        raise ValueError('opt_choice must be 0 or 1')
    check.real_nonnegative_scalar(n, 'n', TypeError)
    check.real_nonnegative_scalar(tol, "tol", TypeError)
    check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)

    _gmin = 1
    #t_lb = np.min(np.array([tmin, 3*cic]))
    t_lb = tmin

    # 1.75 hr = 6300 s (total possible roll time)
    #t_pcmax = (pc_ecount_max-cic)/(fluxe_bright+darke)
    #t_pcmax = (pc_ecount_max-cic)/(fluxe+darke)
    t_pcmax = (pc_ecount_max)/(fluxe)
    #if t_pcmax <= 0:
    #    raise EXCAMOptimizeException('cic greater than pc_ecount_max')

    t_ub = np.min(np.array([tmax,6300,t_pcmax]))
    # 8 rolls per day, 2 s minimum per frame
    N_ub = np.min(np.array([Nmax, 8*6300/2]))

    # threshold, T, need not be 5*cic since we are subtracting pc darks
    T = 5*rn
    g_lb = np.max(np.array([_gmin, T]))
    # Check that there are no unsatisfiable constraints; parallel cic ~ 0
    if (fluxe_bright*t_lb + darke*t_lb) + \
        n*np.sqrt((fluxe_bright*t_lb + darke*t_lb)) > alpha0*fwc:
        return (0,0,0,0,3)
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')
    if g_lb*(fluxe_bright*t_lb + darke*t_lb + cic) + n*_ENF(g_lb,Nem)*g_lb*\
        np.sqrt(fluxe_bright*t_lb + darke*t_lb + cic) > alpha1*fwc_em:
        return (0,0,0,0,3)
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')
    if g_lb >= gmax:
        return (0,0,0,0,3)
        raise EXCAMOptimizeException("No finite-width window of viable EM" +
        'gain for photon counting')
    if t_lb >= t_ub:
        return (0,0,0,0,3)
        raise EXCAMOptimizeException('No finite-width window of viable frame' +
        'time for photon counting')
    if Nmin >= N_ub:
        return (0,0,0,0,3)
        raise EXCAMOptimizeException('No finite-width window of vaible' +
        'number of frames for photon counting')

    # XXX need target snr, 2-level opt scheme in continuous frame mode
    # XXX change fwc_em to 50k in excam_config; and what about fwc?
    # XXX account for cosmics by changing Nfr to include correction factors?
    # XXX min exposure time should be readout time (.264s)? More like 1 s.
    # XXX SNR comparisons b/w pc and non-pc? with excam_tools_time.py?
    # XXX if N input instead of t_tot, can do raise exception if N*2s > 6300; if tfr > 2, then take that into account for Nfr upper bound
    # XXX and need to change code to have a fixed t_tot and then make chosen N less than num used for calibrating darks (through excam_config?) If so, need to add into code constraint that each frame must be at least 2 secs

    # [g, tfr, N] = v
    bounds = scipy.optimize.Bounds(lb=np.array([g_lb, t_lb, Nmin]),
                                ub=np.array([gmax, t_ub, N_ub]))

    em_rail = lambda v: v[0]*(fluxe_bright*v[1] + darke*v[1] + cic) + (
                        n*_ENF(v[0], Nem)*v[0]*np.sqrt(fluxe_bright*v[1] +
                        darke*v[1] + cic)
                        )
    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1*fwc_em)

    nowsnr = lambda v: _SNR_CR_pc(v[0], v[1], v[2], fluxe, darke, cic, T, X,
                        a, Lij, sign=1)

    nconst2 = scipy.optimize.NonlinearConstraint(nowsnr,
                                        target_snr, np.inf)

    rail = lambda v: fluxe_bright*v[1] + darke*v[1] + (
                    n*np.sqrt(fluxe_bright*v[1] + darke*v[1])
                    )
    nconst3 = scipy.optimize.NonlinearConstraint(rail, 0, alpha0*fwc)

    bim_rail = lambda v: v[0]*(fluxe_bim*v[1] + darke*v[1] + cic)

    nconst4 = scipy.optimize.NonlinearConstraint(bim_rail, 0, fault)

    t_tot = lambda v: v[1]*v[2]

    nconst5 = scipy.optimize.NonlinearConstraint(t_tot, 0, 6300*8)

    _tmp_opt_choice = 0 #used to go to 2nd optimization if first fails

    if opt_choice == 0:
        FOM = lambda v: v[1]*v[2] # no gain in wall-clock time

        res1 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([g_lb, t_lb, Nmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3,
                                    nconst4, nconst5),
                                    )

        #same thing, but starting point of gmax
        res2 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([gmax, t_lb, Nmin]),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst2, nconst3,
                                    nconst4, nconst5),
                                    )

        g1 = res1.x[0]
        tfr1 = res1.x[1]
        N1 = int(np.ceil(res1.x[2]))
        if N1*tfr1/(8*6300) > 1:  # +delta_constr:
            N1 = N1 - 1
        N1 = res1.x[2]
        g2 = res2.x[0]
        tfr2 = res2.x[1]
        N2 = int(np.ceil(res2.x[2]))
        if N2*tfr2/(8*6300) > 1:   #+delta_constr:
            N2 = N2 - 1
        N2 = res2.x[2]
        snr1 = nowsnr(np.array([g1, tfr1, N1]))
        snr2 = nowsnr(np.array([g2, tfr2, N2]))

        #constraints met
        res1_cond = (
         (rail(np.array([g1, tfr1]))/(alpha0*fwc) <= 1+delta_constr) and
         (em_rail(np.array([g1, tfr1]))/(alpha1*fwc_em) <= 1+delta_constr) and
         (bim_rail(np.array([g1, tfr1])) < fault) and
         (N1*tfr1/(8*6300) <= 1) and
         (snr1/target_snr >= 1-delta_constr)
                    )

        res2_cond = (
         (rail(np.array([g2, tfr2]))/(alpha0*fwc) <= 1+delta_constr) and
         (em_rail(np.array([g2, tfr2]))/(alpha1*fwc_em) <= 1+delta_constr) and
         (bim_rail(np.array([g2, tfr2])) < fault) and
         (N2*tfr2/(8*6300) <= 1) and
         (snr2/target_snr >= 1-delta_constr)
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

        def _SNR_CR_pc1(v, fluxe, darke, cic, T, X, a, Lij, sign):
            g, tfr, N = v
            return _SNR_CR_pc(g, tfr, N, fluxe, darke, cic, T, X, a, Lij, sign)

        res3 = scipy.optimize.minimize(fun=_SNR_CR_pc1,
                                   x0=np.array([g_lb, t_lb, Nmin]),
                                   args=(fluxe, darke, cic, T, X, a, Lij, -1),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3, nconst4,
                                   nconst5),
                                   )


        #same thing, but starting point of gmax
        res4 = scipy.optimize.minimize(fun=_SNR_CR_pc1,
                                   x0=np.array([gmax, t_lb, Nmin]),
                                   args=(fluxe, darke, cic, rn, X, a, Lij, -1),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3, nconst4,
                                   nconst5),
                                   )

        g3 = res3.x[0]
        tfr3 = res3.x[1]
        #N3 = res3.x[2]
        N3 = int(np.ceil(res3.x[2]))
        if N3*tfr3/(8*6300) > 1: # +delta_constr:
            N3 = N3 - 1
        g4 = res4.x[0]
        tfr4 = res4.x[1]
        N4 = int(np.ceil(res4.x[2]))
        if N4*tfr4/(8*6300) > 1: #+delta_constr:
            N4 = N4 - 1
        #N4 = res4.x[2]
        snr3 = nowsnr(np.array([g3, tfr3, N3]))
        snr4 = nowsnr(np.array([g4, tfr4, N4]))

        #constraints met
        res3_cond = (
         (rail(np.array([g3, tfr3]))/(alpha0*fwc) <= 1+delta_constr) and
         (em_rail(np.array([g3, tfr3]))/(alpha1*fwc_em) <= 1+delta_constr) and
         (bim_rail(np.array([g3, tfr3])) < fault) and
         (N3*tfr3/(8*6300) <= 1)
                    )

        res4_cond = (
         (rail(np.array([g4, tfr4]))/(alpha0*fwc) <= 1+delta_constr) and
         (em_rail(np.array([g4, tfr4]))/(alpha1*fwc_em) <= 1+delta_constr) and
         (bim_rail(np.array([g4, tfr4])) < fault) and
         (N4*tfr4/(8*6300) <= 1)
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

    return (0,0,0,0,3)
    raise EXCAMOptimizeException('Both optimizations failed, cannot produce ' +
                                 'camera settings')


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
                            Nmax=25200,#49,
                            tmin=1,
                            tmax=6300.,
                            gmax=5000,
                            opt_choice=0,
                            n=4,
                            Nem=604,
                            tol=1e-30,
                            delta_constr=1e-4
                            )
    print(out)

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
                            Nmax=10,
                            t = 10,
                            gmax=5000,
                            opt_choice=0,
                            n=4,
                            Nem=604,
                            tol=1e-30,
                            delta_constr=1e-4
                            )
    print(out2)

    out3 = calc_pc(target_snr = 2,
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
                Nmin = 1,
                Nmax = 25200, #8 rolls, 3150 frames per roll for min time of 2s
                tmin = 1, #1,
                tmax = 1.0000999999999662,#6300,
                gmax=5000,
                pc_ecount_max = 0.1,
                fault = 250000,
                fluxe_bim = None,
                opt_choice=0,
                n=4,
                Nem=604,
                tol=1e-30,
                delta_constr=1e-4
                )
    print(out3)