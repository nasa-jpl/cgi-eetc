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
    #minimal case for rail constraint:
    if (fluxe_bright*tmin + darke*tmin + cic) + \
        (n*_SNR_CR_den(1, Nem, rn, fluxe_bright, tmin,
        darke, cic)) > alpha0*fwc:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')

    _gmin = 1

    # in case alpha1*fwc_em is input as smaller than alpha0*fwc:
    #minimal case for em_rail constraint:
    if _gmin*(fluxe_bright*tmin + darke*tmin + cic) + \
        (n*_gmin*_SNR_CR_den(_gmin, Nem, rn, fluxe_bright, tmin,
        darke, cic)) > alpha1*fwc_em:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')

    # [g, tfr, N] = v
    bounds = scipy.optimize.Bounds(lb=np.array([_gmin, tmin, Nmin]),
                                ub=np.array([gmax, tmax, Nmax]))

    em_rail = lambda v: v[0]*(fluxe_bright*v[1] + darke*v[1] + cic) + (
                        n*v[0]*_SNR_CR_den(v[0], Nem, rn, fluxe_bright, v[1],
                                        darke, cic)
                        )
    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1*fwc_em)

    nowsnr = lambda v: _SNR_CR(v[0], v[1], v[2], fluxe, darke, cic, rn, X, a,
                        Lij, sign=1)

    nconst2 = scipy.optimize.NonlinearConstraint(nowsnr,
                                        target_snr, np.inf)

    rail = lambda v: fluxe_bright*v[1] + darke*v[1] + cic + (
                    n*_SNR_CR_den(1, Nem, rn, fluxe_bright, v[1], darke, cic)
                    )
    nconst3 = scipy.optimize.NonlinearConstraint(rail, 0, alpha0*fwc)

    _tmp_opt_choice = 0 #used to go to 2nd optimization if first fails


    if opt_choice == 0:
        FOM = lambda v: v[1]*v[2] # no gain in wall-clock time

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

        #return (g1, tfr1, N1, snr1, 11+res1.status)

        #constraints met
        res1_cond = (
         (rail(np.array([g1, tfr1]))/(alpha0*fwc) <= 1+delta_constr) and
         (em_rail(np.array([g1, tfr1]))/(alpha1*fwc_em) <= 1+delta_constr) and
         (snr1/target_snr >= 1-delta_constr)
                    )

        res2_cond = (
         (rail(np.array([g2, tfr2]))/(alpha0*fwc) <= 1+delta_constr) and
         (em_rail(np.array([g2, tfr2]))/(alpha1*fwc_em) <= 1+delta_constr) and
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


        def _SNR_CR1(v, fluxe, darke, cic, rn, X, a, Lij, sign, Nem=604):
            g, tfr, N = v
            return _SNR_CR(g, tfr, N, fluxe, darke, cic, rn, X, a, Lij, sign,
            Nem=604)

        res3 = scipy.optimize.minimize(fun=_SNR_CR1,
                                   x0=np.array([_gmin, tmin, Nmin]),
                                   args=(fluxe, darke, cic, rn, X, a, Lij, -1),
                                   bounds=bounds,
                                   tol=tol,
                                   constraints=(nconst1, nconst3),
                                   )


        #same thing, but starting point of gmax
        res4 = scipy.optimize.minimize(fun=_SNR_CR1,
                                   x0=np.array([gmax, tmin, Nmin]),
                                   args=(fluxe, darke, cic, rn, X, a, Lij, -1),
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

        if not res4.success:
            # res42 = scipy.optimize.minimize(fun=_SNR_CR1,
            #                     x0=np.array([(g4+_gmin)/2, (tfr4+tmax)/2, (N4+Nmax)/2]),
            #                     args=(fluxe, darke, cic, rn, X, a, Lij, -1),
            #                     bounds=bounds,
            #                     tol=tol,
            #                     constraints=(nconst1, nconst3),
            #                     )
            g42 = res4.x[0]
            tfr42 = res4.x[1]
            N42 = int(np.ceil(res4.x[2]))
            _x = False
            count = 0
            while _x ==False and count<=10:
                res42 = scipy.optimize.minimize(fun=_SNR_CR1,
                                #x0=np.array([np.max(np.array([g42*.999,_gmin])), np.min(np.array([tfr42*1.001,tmax])), np.min(np.array([N42*1.001,Nmax]))]),
                                #x0=np.array([np.max(np.array([g42-1,_gmin])), np.min(np.array([tfr42+1,tmax])), np.min(np.array([N42+1,Nmax]))]),
                                #x0=np.array([g42,tfr42,N42]),
                                x0=np.array([(g42+_gmin)/2,(tfr42+tmax)/2,(N42+Nmax)/2]),
                                args=(fluxe, darke, cic, rn, X, a, Lij, -1),
                                bounds=bounds,
                                tol=tol,
                                constraints=(nconst1, nconst3),
                                )
                count += 1
                _x = res42.success
                g42 = res42.x[0]
                tfr42 = res42.x[1]
                N42 = int(np.ceil(res42.x[2]))
                snr42 = nowsnr(np.array([g42, tfr42, N42]))

            g42 = res42.x[0]
            tfr42 = res42.x[1]
            N42 = int(np.ceil(res42.x[2]))
            snr42 = nowsnr(np.array([g42, tfr42, N42]))
            res42_cond = (
                (rail(np.array([g42, tfr42]))/(alpha0*fwc) <= 1+delta_constr) and
                (em_rail(np.array([g42, tfr42]))/(alpha1*fwc_em) <= 1+delta_constr)
                        )

            if res42_cond and res42.success:
                if np.isclose([g4,tfr4,N4,snr4],[g42,tfr42,N42,snr42],rtol=delta_constr).all()==True:
                    return (g42, tfr42, N42, snr42, 21)
                else:
                    if snr42 > snr4:
                        return (g42, tfr42, N42, snr42, 22)
                    if snr42 < snr4:
                        return (g4, tfr4, N4, snr4, 23)
            if res42_cond and not res42.success:
                if np.isclose([g4,tfr4,N4,snr4],[g42,tfr42,N42,snr42],rtol=delta_constr).all()==True:
                    return (g42, tfr42, N42, snr42, 24)
                else:
                    if snr42 > snr4:
                        return (g42, tfr42, N42, snr42, 25)
                    if snr42 < snr4:
                        return (g4, tfr4, N4, snr4, 26)
            if not res42_cond:
                return (g42, tfr42, N42, snr42, 27)
        #return (g4, tfr4, N4, snr4, 11+res4.status)

        #constraints met
        res3_cond = (
         (rail(np.array([g3, tfr3]))/(alpha0*fwc) <= 1+delta_constr) and
         (em_rail(np.array([g3, tfr3]))/(alpha1*fwc_em) <= 1+delta_constr)
                    )

        res4_cond = (
         (rail(np.array([g4, tfr4]))/(alpha0*fwc) <= 1+delta_constr) and
         (em_rail(np.array([g4, tfr4]))/(alpha1*fwc_em) <= 1+delta_constr)
                    )

        # if res3_cond and res4_cond:
        #     if snr3 >= snr4:
        #         return (g3, tfr3, N3, snr3, 1)
        #     if snr3 < snr4:
        #         return (g4, tfr4, N4, snr4, 1)
        # elif res3_cond and not res4_cond:
        #     return (g3, tfr3, N3, snr3, 1)
        # elif res4_cond and not res3_cond:
        #     return (g4, tfr4, N4, snr4, 1)

    return (0,0,0,0,3)
    #raise EXCAMOptimizeException('Both optimizations failed, cannot produce ' +
    #                             'camera settings')

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
    # rail constraint (for fixed t):
    if (fluxe_bright*t + darke*t + cic) + \
        (n*_SNR_CR_den(1, Nem, rn, fluxe_bright, t,
        darke, cic)) > alpha0*fwc:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')

    _gmin = 1

    # in case alpha1*fwc_em is input as smaller than alpha0*fwc:
    #minimal case for em_rail constraint:
    if _gmin*(fluxe_bright*t + darke*t + cic) + \
        (n*_gmin*_SNR_CR_den(_gmin, Nem, rn, fluxe_bright, t,
        darke, cic)) > alpha1*fwc_em:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')

    # [g, N] = v
    bounds = scipy.optimize.Bounds(lb=np.array([_gmin, Nmin]),
                                ub=np.array([gmax, Nmax]))

    em_rail = lambda v: v[0]*(fluxe_bright*t + darke*t + cic) + (
                        n*v[0]*_SNR_CR_den(v[0], Nem, rn, fluxe_bright, t,
                                        darke, cic)
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

    return (0, 0, 0, 0, 3)
    #raise EXCAMOptimizeException('Both optimizations failed, cannot produce ' +
    #                             'camera settings')

if __name__ == "__main__":
    # testing only
    ap = argparse.ArgumentParser(prog='python camera.py',
                                 description="Compute EXCAM settings")
    ap.add_argument('--snr', default=7, help="Target SNR.  Default = 7.",
                    type=float)
    ap.add_argument('--flux', default=100, type=float,
                    help="Electrons from photon flux per second. Default=100.")
    args = ap.parse_args()

    out = calc_gain_exptime(target_snr=9, #183.54790086591134, #args.snr,
                            fluxe=0.4,
                            fluxe_bright=8,
                            darke=8.33e-4,  # inputs from excam_properties.yaml
                            cic=0.02,
                            rn=160,
                            X=5e4,
                            a=1.69e-10,
                            Lij=512,
                            alpha0=0.75,
                            fwc=60000,
                            alpha1=0.75,
                            fwc_em=100000,
                            Nmin=1,
                            Nmax=49,
                            tmin=0.264,
                            tmax=120.,
                            t = 10,
                            gmax=5000,
                            opt_choice=0,
                            n=4,
                            Nem=604,
                            tol=1e-30,
                            delta_constr=1e-4
                            )
    print(out)
