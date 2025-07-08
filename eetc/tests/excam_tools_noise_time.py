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
from xml.etree.ElementTree import TreeBuilder

import scipy.optimize
import numpy as np

import eetc.util.check as check


class EXCAMOptimizeException(Exception):
    """Thin class for optimizer-specific failures"""
    pass

def _fwcem(g):
    """
    Returns fwc_em.  Dependence was derived from an interpolation.
    """
    if g <=1.06082:  #corresponding to hv = 22 (1.0608 in fit function, but we know it to be 1 hv=22)
        return 100000
    if g > 1.06082:
        return int(np.floor((6917529027641081856000*(3/2)**(337164395852765312/6243314768165359- (11258999068426240*np.log(76451918253118240*np.log(g)))/6243314768165359))/1332894850849759 + 40000))
        #return ((6917529027641081856000*(3/2)**(337164395852765312/6243314768165359- (11258999068426240*np.log(76451918253118240*np.log(g)))/6243314768165359))/1332894850849759 + 40000)


def _slope(g):
    """
    Fitted function for slope as it differs from 0.5 (shot noise).
    """
     #fitSlope=  0.5 +2.^(hv/2.3)/2.^(22/2.3)/750;
     #fitG = 10.^(2.^(hv0/2.5)/2.^(22/2.5)/39);
     #invert the function to get gain(hv):
     #hv = -(210*log(2) + 5*log(log(10))...
     #    - 5*log(76451918253118239*log(g)))/(2*log(2));
    if g <=1:
        return 0.5
    if g > 1:
        return (4398046511104*2**((225179981368524800*np.log(76451918253118240*np.log(g)))/143596239667803257 - 6743287917055306240/143596239667803257))/2498839895518397625 + 1/2


def _ypow(g,status):
    """
    Fitted function for the power to which the factor representing the
    y-intercept of PTC curves should be raised.  Linear interpolation for
    the y-intercepts was good enough for this process.

    status:  {0, 1}
        0 if you don't want to use the y-intercept constraint (assume regular form in the noise), 1 if you want to use it.
    """
    if status == 0:
        return 0.5
    if status == 1:
        if g <=1.495:  #g corresponding to hv=30 (exactly, not from fitted function, but doesn't matter too much either way
            return -0.0972
        if g > 1.495:
            hv =  -(210*np.log(2) + 5*np.log(np.log(10))- 5*np.log(76451918253118239*np.log(g)))/(2*np.log(2))
            b = (hv-30)*(0.1253 + 0.4836)/9 - 0.4836
            #return (b-(_slope(g)-1)*np.log10(8))/(np.log10((_ENF(g,604))**2*g))
            #replace _slope(g) with ((1.07882-.789911)/(719.766-1.495)*
            # (g-1.495)+0.789911), which essentially breaks up _slope(g)
            # into as many steps as there are in g between HV=30 and 39;
            # need to do this b/c I originally assumed linear fit for y-int,
            # and it really isn't linear
            return (b-(((1.07882-.789911)/(719.766-1.495)*(g-1.495)+0.789911)-1)*np.log10(8))/(np.log10((_ENF(g,604))**2*g)) #specific to 604 EMCCD


def _ENF(g, Nem):
    """
    Returns the ENF.
    """
    return np.sqrt(2*(g-1)*g**(-(Nem+1)/Nem) + 1/g)


def _SNR_CR_den(g, Nem, rn, fluxe, tfr, darke, cic, status):
    """
    Returns the denominator of the SNR given camera settings and noise
    properties, including cosmic ray effects (denominator of the
    _SNR_CR function).
    """
    return np.sqrt(rn**2 + ((_ENF(g,Nem))**2*g)**(2*_ypow(g,status))*(g*(fluxe*tfr + darke*tfr + cic))**(2*_slope(g)))


def _SNR_CR(v, t_tot, fluxe, darke, cic, rn, X, a, Lij, sign, status, Nem=604):
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
    g, tfr = v
    #int(np.ceil((t_tot/tfr)))
    #num = np.sqrt(int(np.ceil((t_tot/tfr)))*np.exp(-X*a*Lij*tfr))*fluxe*tfr
    num = np.sqrt((t_tot/tfr)*np.exp(-X*a*Lij*tfr))*fluxe*tfr*g
    den = _SNR_CR_den(g, Nem, rn, fluxe, tfr, darke, cic, status)
    if den == 0:
        return 0 # in the limit den terms go to zero, num terms go faster

    return sign*num/den


def calc_gain_exptime(t_tot, fluxe, fluxe_bright, darke, cic, rn, X, a,
                      Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax, tmin, tmax,
                      gmax, status, opt_choice=0, n=4, Nem=604, tol=1e-30,
                      delta_constr=1e-4, hard_limit=True, **kwargs):
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
    check.real_scalar(alpha0, 'alpha0', TypeError)
    if alpha0 < 0 or alpha0 > 1:
        raise ValueError('alpha0 must be between 0 and 1 (fraction of total)')
    check.positive_scalar_integer(fwc, 'fwc', TypeError)
    check.real_scalar(alpha1, 'alpha1', TypeError)
    if alpha1 < 0 or alpha1 > 1:
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
    check.real_scalar(gmax, 'gmax', TypeError)
    if gmax < 1:
        raise ValueError('gmax must be >= 1')
    check.nonnegative_scalar_integer(status, 'status', TypeError)
    if ((status != 0) and (status != 1)):
        raise ValueError('status must be 0 or 1')
    check.nonnegative_scalar_integer(opt_choice, 'opt_choice', TypeError)
    if ((opt_choice != 0) and (opt_choice != 1)):
        raise ValueError('opt_choice must be 0 or 1')
    check.positive_scalar_integer(n, 'n', TypeError)
    check.real_nonnegative_scalar(Nem, 'Nem', TypeError)
    check.real_nonnegative_scalar(tol, "tol", TypeError)
    check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)


    # [g, tfr] = v
    t_lb = np.max(np.array([t_tot/Nmax,tmin]))
    t_ub = np.min(np.array([t_tot/Nmin,tmax]))

    # Check that there are no unsatisfiable constraints
    if (fluxe_bright*t_lb + darke*t_lb + cic) + \
        (n*_SNR_CR_den(1, Nem, rn, fluxe_bright, t_lb,
        darke, cic, status)) > alpha0*fwc:
        return (0,0,0,0,0,11)
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')
    #if (fluxe_bright*t_lb + darke*t_lb + cic) + \
    #    (n*_SNR_CR_den(1, Nem, rn, fluxe_bright, t_lb,
    #    darke, cic, status)) > alpha1*_fwcem(1):
    #    return (0,0,0,0,0,11)
        raise EXCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for EM full well)')
    if Nmax*t_ub < t_tot:
        return (0,0,0,0,0,11)
        raise EXCAMOptimizeException('t_tot is bigger than Nmax*tmax')

    if t_lb == t_ub:
        t0 = t_lb
        N0 = t_tot/t0
        def emrail0(g):
            return g*(fluxe_bright*t0 + darke*t0 + cic) + \
                n*_SNR_CR_den(g, Nem, rn, fluxe_bright, t0, darke, cic, status) - \
                alpha1*_fwcem(g)
        gsol = scipy.optimize.fsolve(emrail0,1)[0] #there will only be on sol
        g0 = np.min(np.array([gsol,gmax]))
        snr0 = _SNR_CR([g0,t0],t_tot,fluxe,darke,cic,rn,X,a,Lij,1,status)
        res0_cond = (
                    ((emrail0(1)+alpha1*_fwcem(1))/(alpha0*fwc) <= 1+delta_constr) and
                    ((emrail0(g0)+alpha1*_fwcem(g0))/(alpha1*fwc_em) <= 1+delta_constr) and
                    (N0/Nmin >=1) and
                    (N0/Nmax <= 1)
                    #(total_time(np.array([g3,tfr3])) <= 1.1*t_tot) and
                    #(total_time(np.array([g3,tfr3])) >= 0.9*t_tot)
                    )
        if res0_cond:
            return (g0, t0, N0, snr0, t0*N0, 1)

    lb=np.array([1, t_lb])
    ub=np.array([gmax, t_ub])
    bounds = scipy.optimize.Bounds(lb=lb,ub=ub)

    num_frames = lambda v: t_tot/v[1]
    nconst4 = scipy.optimize.NonlinearConstraint(num_frames, Nmin, Nmax)

    em_rail = lambda v: (v[0]*(fluxe_bright*v[1] + darke*v[1] + cic) + (
                        n*_SNR_CR_den(v[0], Nem, rn, fluxe_bright, v[1],
                                        darke, cic, status)))/_fwcem(v[0])

    nconst1 = scipy.optimize.NonlinearConstraint(em_rail, 0, alpha1)

    nowsnr = lambda v: _SNR_CR(v, t_tot, fluxe, darke, cic, rn, X, a, Lij, 1, status)


    rail = lambda v: fluxe_bright*v[1] + darke*v[1] + cic + (
                    n*_SNR_CR_den(1, Nem, rn, fluxe_bright, v[1], darke, cic, status)
                    )

    nconst3 = scipy.optimize.NonlinearConstraint(rail, 0, alpha0*fwc)

    _tmp_opt_choice = 1

    if (opt_choice == 1) or (_tmp_opt_choice == 1):
        res3 = scipy.optimize.minimize(fun=_SNR_CR,
                                    x0=np.array([1, t_lb]),
                                    args=(t_tot, fluxe, darke, cic, rn, X, a, Lij, -1, status),
                                    bounds=bounds,
                                    tol=tol,
                                    #options={'disp':True,'maxiter':10000},
                                    constraints=(nconst1, nconst3, nconst4)
                                    )

        #same thing, but starting point of gmax
        res4 = scipy.optimize.minimize(fun=_SNR_CR,
                                    x0=np.array([gmax, t_lb]),
                                    args=(t_tot, fluxe, darke, cic, rn, X, a, Lij, -1, status),
                                    bounds=bounds,
                                    tol=tol,
                                    #options={'disp':True,'maxiter':10000},
                                    constraints=(nconst1, nconst3, nconst4)
                                    )

        g3 = res3.x[0]
        #N3 = np.round(num_frames(np.array([g3,res3.x[1]])))
        #we want ceil here b/c our goal is to maximize SNR, and N drives SNR
        #more than tfr
        N3 = np.ceil(num_frames(np.array([g3,res3.x[1]])))
        if hard_limit:
            tfr3 = t_tot/N3
        else:
            tfr3 = res3.x[1]

        g4 = res4.x[0]
        #N4 = np.round(num_frames(np.array([g4,res4.x[1]])))
        N4 = np.ceil(num_frames(np.array([g4,res4.x[1]])))
        if hard_limit:
            tfr4 = t_tot/N4
        else:
            tfr4 = res4.x[1]

        snr3 = nowsnr(np.array([g3, tfr3]))
        snr4 = nowsnr(np.array([g4, tfr4]))

        #constraints met%
        res3_cond = (
                    (rail(np.array([g3,tfr3]))/(alpha0*fwc) <= 1+delta_constr) and
                    (em_rail(np.array([g3,tfr3]))/(alpha1) <= 1+delta_constr) and
                    (N3/Nmin >=1) and
                    (N3/Nmax <= 1)
                    #(total_time(np.array([g3,tfr3])) <= 1.1*t_tot) and
                    #(total_time(np.array([g3,tfr3])) >= 0.9*t_tot)
                    )

        res4_cond = (
                    (rail(np.array([g4,tfr4]))/(alpha0*fwc) <= 1+delta_constr) and
                    (em_rail(np.array([g4,tfr4]))/(alpha1) <= 1+delta_constr) and
                    (N4/Nmin >=1) and
                    (N4/Nmax <= 1)
                    #(total_time(np.array([g4,tfr4])) <= 1.1*t_tot) and
                    #(total_time(np.array([g4,tfr4])) >= 0.9*t_tot)
                    )


        if ((res3_cond==True) and (res4_cond==True)):
            if snr3>=snr4:
                return (g3, tfr3, N3, snr3,
                tfr3*N3, 1)
            if snr3 < snr4:
                return (g4, tfr4, N4, snr4,
                tfr4*N4, 1)
        elif (res3_cond==True) and (res4_cond==False):
            return (g3, tfr3, N3, snr3,
            tfr3*N3, 1)
        elif (res3_cond==False) and (res4_cond==True):
            return (g4, tfr4, N4, snr4,
            tfr4*N4, 1)

    # if res3_cond==False:
    #     print("rail(np.array([g3,tfr3]))/(alpha0*fwc): ",
    #     rail(np.array([g3,tfr3]))/(alpha0*fwc))
    #     print("em_rail(np.array([g3,tfr3]))/(alpha1*fwc_em): ",
    #     em_rail(np.array([g3,tfr3]))/(alpha1*fwc_em))
    # if res4_cond==False:
    #     print("rail(np.array([g4,tfr4]))/(alpha0*fwc): ",
    #     rail(np.array([g4,tfr4]))/(alpha0*fwc))
    #     print("em_rail(np.array([g4,tfr4]))/(alpha1*fwc_em): ",
    #     em_rail(np.array([g4,tfr4]))/(alpha1*fwc_em))
    # print("fluxe: ", fluxe, " fluxe_bright: ",fluxe_bright, "total: ",t_tot)
    return (0,0,0,0,0,3)
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

    out = calc_gain_exptime(t_tot=30*120, #args.snr,
                            fluxe=26.366508987303554,
                            fluxe_bright=54.555947811685144,
                            darke=8.33e-4,
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
                            Nmax=49,
                            tmin=0.264,
                            tmax=120.,
                            gmax=5000,
                            status=1,
                            opt_choice=1,
                            n=4,
                            Nem=604,
                            tol=1e-30,
                            delta_constr=1e-4,
                            hard_limit=True
                            )
    print(out)