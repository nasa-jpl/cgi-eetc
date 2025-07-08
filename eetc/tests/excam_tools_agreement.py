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

import numpy as np
import scipy.optimize


import eetc.util.check as check


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


def _SNR_CR(v, fluxe, darke, cic, rn, X, a, Lij, sign, Nem=604):
    """
    Compute the SNR given camera settings and noise properties, including
    cosmic ray effects.

    Internal function to feed the optimizer only.  Does not enforce any of the
    bounds you might expect, as the optimizer will eventually enforce them, so
    use for non-optimizer applications at your own risk.

    Parameters
    ----------
    v : array_like
        three-element vector of (gain, exposure time, number of frames).  These
        are all shoved in one vector as this is what the optimizer expects.

        g = camera gain, unitless
        tfr = exposure time per frame, in seconds.
        N = number of frames, unitless.

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
    g, tfr, N = v

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
    iteration.  If you don't have a target SNR and simply want to maximize the SNR,
    you may specify opt_choice=1 to skip to the 2nd optimizer.

    This function runs an optimization to find the combination of gain,
    exposure time, and number of frames which provides an SNR at or better than
    a target SNR with the smallest wall-clock time.  If there is no feasible
    combination which does so, it runs a second optimization to find the best
    SNR it can get given the constraints.

    Despite the long list of inputs, almost all of them are fixed properties of
    the EXCAM detector or the CGI system as a whole.  The only 2 that will
    usually be changed are the target SNR (target_snr) and incoming flux,
    converted into electrons (fluxe).  Both of these depend on the
    astrophysical target and the use case of CGI at that time.

    Inputs that violate the specifications given below will raise a TypeError
    or a ValueError.

    If both optimizations are somehow infeasible, or if compound constraints
    (e.g. electrons generation rate vs. full-well level) that depend on fluxe
    are violated, an EXCAMOptimizeException will be raised.

    Known issues: if the upper and lower bounds on gain, exposure time, or
    number of frames are identical, so that value is fixed, then the default
    optimizer has problems computing finite derivatives.  The HOWFSC use case
    doesn't fix any of these parameters, so this is not an issue for v1.0, but
    should be fixed in a later revision for robustness.  TODO

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
        constraint bounds used in scipy minimize function made more
        conservative by this fraction to ensure the output of the minimize
        function, within its own error, actually meets the constraints.  >=0.
        For example, delta_constr = 0.01 means that 1.01*target_snr is the
        target snr in that constraint for scipy minimize.  Defaults to 0.01.

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
    if (fluxe*tmin + darke*tmin + cic) > alpha0*fwc:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')
    if (fluxe*tmin + darke*tmin + cic) > alpha1*fwc_em:
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')


    # [g, tfr, N] = v
    bounds = scipy.optimize.Bounds(lb=np.array([1, tmin, Nmin]),
                                ub=np.array([gmax, tmax, Nmax]))

    # bounds on constraint made more conservative than required since minimize
    # function tends to erroneously land on value just slightly outside of
    # bounds; delta_constr used to accomplish this
    em_rail = lambda v: v[0]*(fluxe_bright*v[1] + darke*v[1] + cic) + (
                        n*v[0]*_SNR_CR_den(v[0], Nem, rn, fluxe_bright, v[1],
                                        darke, cic)
                        )
    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1*fwc_em*
                                                (1-delta_constr))

    nowsnr = lambda v: _SNR_CR(v, fluxe, darke, cic, rn, X, a, Lij, sign=1)
    nconst2 = scipy.optimize.NonlinearConstraint(nowsnr,
                                        (1+delta_constr)*target_snr, np.inf)

    rail = lambda v: fluxe_bright*v[1] + darke*v[1] + cic + (
                    n*_SNR_CR_den(1, Nem, rn, fluxe_bright, v[1], darke, cic)
                    )
    nconst3 = scipy.optimize.NonlinearConstraint(rail, 0, alpha0*fwc*
                                                (1-delta_constr))

    _tmp_opt_choice = 0 #used to go to 2nd optimization if first fails


    if opt_choice == 0:
        FOM = lambda v: v[1]*v[2] # no gain in wall-clock time

        res1 = scipy.optimize.minimize(fun=FOM,
                                    x0=np.array([1, tmin, Nmin]),
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
            (rail(np.array([g1,tfr1]))/(alpha0*fwc)<= 1) and
            (em_rail(np.array([g1,tfr1]))/(alpha1*fwc_em)<= 1) and
            (snr1/target_snr >= 1)
                    )

        res2_cond = (
            (rail(np.array([g2,tfr2]))/(alpha0*fwc)<= 1) and
            (em_rail(np.array([g2,tfr2]))/(alpha1*fwc_em)<= 1) and
            (snr2/target_snr >= 1)
                    )

        if (res1_cond==True) and (res2_cond==True):
            if ( N1*tfr1/(N2*tfr2) >=1+delta_constr) or ( N1*tfr1/(N2*tfr2) <=1-delta_constr) or ( snr1/snr2 >=1+delta_constr) or ( snr1/snr2 <=1-delta_constr):
                return (1, 1, 1, 1, 9)
            if N1*tfr1 <= N2*tfr2:
                return (g1, tfr1, N1, snr1, 0)
            if N1*tfr1 > N2*tfr2:
                return (g2, tfr2, N2, snr2, 0)
        elif (res1_cond==True) and (res2_cond==False):
            return (g1, tfr1, N1, snr1, 0)
        elif (res1_cond==False) and (res2_cond==True):
            return (g2, tfr2, N2, snr2, 0)

        #change from excam_tools.py _________________
        if (res1_cond==False) and (snr1/target_snr < 1) and (target_snr !=100):
            print("rail(np.array([g1,tfr1]))/(alpha0*fwc): ",
            rail(np.array([g1,tfr1]))/(alpha0*fwc))
            print("em_rail(np.array([g1,tfr1]))/(alpha1*fwc_em): ",
            em_rail(np.array([g1,tfr1]))/(alpha1*fwc_em))
            print("snr1/target_snr: ", snr1/target_snr)
        if (res2_cond==False) and (snr2/target_snr < 1) and (target_snr !=100):
            print("rail(np.array([g2,tfr2]))/(alpha0*fwc): ",
            rail(np.array([g2,tfr2]))/(alpha0*fwc))
            print("em_rail(np.array([g2,tfr2]))/(alpha1*fwc_em): ",
            em_rail(np.array([g2,tfr2]))/(alpha1*fwc_em))
            print("snr2/target_snr: ", snr2/target_snr)
#_____________________________________________

        _tmp_opt_choice = 1

    if (opt_choice == 1) or (_tmp_opt_choice == 1):
        res3 = scipy.optimize.minimize(fun=_SNR_CR,
                                    x0=np.array([1, tmin, Nmin]),
                                    args=(fluxe, darke, cic, rn, X, a, Lij,-1),
                                    bounds=bounds,
                                    tol=tol,
                                    constraints=(nconst1, nconst3),
                                    )


        #same thing, but starting point of gmax
        res4 = scipy.optimize.minimize(fun=_SNR_CR,
                                    x0=np.array([gmax, tmin, Nmin]),
                                    args=(fluxe, darke, cic, rn, X, a, Lij,-1),
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
            (rail(np.array([g3,tfr3]))/(alpha0*fwc) <= 1) and
            (em_rail(np.array([g3,tfr3]))/(alpha1*fwc_em) <= 1)
                    )

        res4_cond = (
           (rail(np.array([g4,tfr4]))/(alpha0*fwc) <= 1) and
            (em_rail(np.array([g4,tfr4]))/(alpha1*fwc_em) <= 1)
                    )

        if (res3_cond==True) and (res4_cond==True):
            if ( N3*tfr3/(N4*tfr4) >=1+delta_constr) or ( N3*tfr3/(N4*tfr4) <=1-delta_constr) or ( snr3/snr4 >=1+delta_constr) or ( snr3/snr4 <=1-delta_constr):
                return (1, 1, 1, 1, 9)
            if snr3 >= snr4:
                return (g3, tfr3, N3, snr3, 1)
            if snr3 < snr4:
                return (g4, tfr4, N4, snr4, 1)
        elif (res3_cond==True) and (res4_cond==False):
            return (g3, tfr3, N3, snr3, 1)
        elif (res3_cond==False) and (res4_cond==True):
            return (g4, tfr4, N4, snr4, 1)

    #all below up until the exception is used for testing in excam_tools_paramspace.py
    #comment this out and uncomment the exception when not testing
    if res3_cond==False:
        print("rail(np.array([g3,tfr3]))/(alpha0*fwc): ",
        rail(np.array([g3,tfr3]))/(alpha0*fwc))
        print("em_rail(np.array([g3,tfr3]))/(alpha1*fwc_em): ",
        em_rail(np.array([g3,tfr3]))/(alpha1*fwc_em))
    if res4_cond==False:
        print("rail(np.array([g4,tfr4]))/(alpha0*fwc): ",
        rail(np.array([g4,tfr4]))/(alpha0*fwc))
        print("em_rail(np.array([g4,tfr4]))/(alpha1*fwc_em): ",
        em_rail(np.array([g4,tfr4]))/(alpha1*fwc_em))
    print("fluxe: ", fluxe, " fluxe_bright: ",fluxe_bright)
    return (g4, tfr4, N4, snr4, 3)
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

    out = calc_gain_exptime(target_snr=.00022, #args.snr,
                            fluxe=1e-6,    # 0.00737866,  #eetc Issue #42
                            fluxe_bright=10,   #0.0737866,
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
                            gmax=5000,
                            opt_choice=0,
                            n=4,
                            Nem=604,
                            tol=1e-30,
                            delta_constr=1e-9,
                            )
    print(out)
