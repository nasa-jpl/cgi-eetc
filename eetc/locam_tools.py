# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Package for computing CCD settings and exposure times for a given observing
scenario, specific to LOCAM.
"""

import argparse

import numpy as np

import eetc.util.check as check

class LOCAMOptimizeException(Exception):
    """Thin class for optimizer-specific failures"""
    pass


def calc_locam_gain(fluxe_bright, darke, cic, alpha0, fwc, alpha1, fwc_em,
                    g_max_comm, g_max_age, e_max_age, tframe, n):
    """
    Calculate gain for LOCAM for a given flux, taking into account LOCAM
    detector parameters

    Gain calculator is trying to optimize given two competing imperatives:
     1. Read noise and bias drift would prefer the largest gain possible, as
        the read noise and bias terms are not enhanced by EM gain and make up
        a smaller fraction of the signal.  Of the two, bias drift is more
        important, as it tends to produce structured variation that is more
        like to skew LOWFSC estimation than the per-pixel randomness of read
        noise.
     2. The detector ages slightly every time a frame is read out with
        non-unity gain, causing the multiplication gain to drop.  The R02HV
        voltage may be raised to compensate, but at around 4-5V of increase,
        the detector will cease to function.  The magnitude of the voltage
        shift is gain-dependent, with larger gains aging the detector at a
        rate relative to smaller greater than linear (i.e. more aging for the
        same number of electrons through the LOCAM).  Smaller gains keep
        aging better under control, with unity-gain operation causing no
        aging at all.

    Our approach is to introduce the second piece--upper gain limit--as hard
    constraints, and maximize gain within the cost function subject to these
    constraints (along with others set by the hardware).

     maximize g
      subject to:
       1. 1 <= g (gain cannot be less than one)
       2. g <= g_max_comm (limit set by maximum permitted input to
                        the command for setting LOCAM gain)
       3. g <= g_max_age (limit set by maximum permitted gain from lifetime
            tests, tuned to allow successful operation for 135 active days.
            This constraint does not apply when g = 1, as the detector does not
            age under unity gain.)
       4. (peak photo-e + N-sigma) <= alpha0*fwc (expected number of
            electrons from photons and standard noise processes in the detector
            is less than the detector full well multiplied by some
            user-selected fractional capacity)
       5. g*(peak photo-e + N-sigma + cic) <= alpha1*fwc_em (expected number of
            electrons from photons and standard noise processes after EM gain
            multiplication is less than the serial-register full well
            multiplied by some user-selected fractional capacity.  CIC is
            serial CIC for the purposes of this model--parallel CIC is
            neglected--and only applied to the post-gain count.  sigma will
            include an excess noise factor [ENF] due to gain.)
       6. g*(peak photo-e + N-sigma + cic) <= e_max_age (expected number of
            electrons from photons and standard noise processes after EM gain
            multiplication is less than the permitted number of electrons per
            frame set by aging considerations.  This constraint does not apply
            when g = 1, as the detector does not age under unity gain. sigma
            will include an excess noise factor [ENF] due to gain.)

    We will make the approximation that the excess noise factor is equal to
    sqrt(2) for non-unity gain checks, which is a good approximation for gains
    greater than ~10 and and conservative otherwise.  (We will ignore it when
    we want to make checks that are explicitly gain=1, as ENF=1 there.)

    Under this assumption, constraints 2, 3, 5, and 6 are linear and (given a
    peak input flux per pixel, which will vary by target) can be rewritten in a
    form g <= X.  Note: do *not* include bias in aging; bias is applied at a
    point in the signal chain after the EM gain register, and so does not
    contribute to full-well or aging caps in 3/5/6.  Same with read noise--it
    will not be part of the N-sigma.

    Constraint 4 has no gain dependence and should be implemented as a
    feasibility precheck--failing this constraint means the target is too
    bright for LOCAM in this optical configuration.

    Given the structure of the cost function, we can reduce the optimization
    analytically and solve for gain with the following pseudocode without
    calling an optimization tool:

     - Check for constraint 4 feasibility, raise exception if violated
     - Rewrite constraints 5 and 6 in "g <= X" form
     - Select the smallest of X_2, X_3, X_5, and X_6.
     - If the smallest of those values is >= 1, return that value as
       the expected gain.
     - Otherwise, if the smallest of those X values is less than 1:
      * Select the smaller of X_2 and X_5 (the constraints that apply at
        g = 1, as unity gain does not cause aging)
      * If the smallest of those two is less than 1, then the problem is not
        feasible at any gain.  Raise an exception.
      * Otherwise, only viable gain is 1; return that.

    Inputs that violate the specifications given below will raise a TypeError
    or a ValueError.  If the specifications of the problem do not admit a
    feasible solution, a LOCAMOptimizeException will be raised.

    Parameters
    ----------
    fluxe_bright : float
        flux maximum, in electrons per second (i.e. photon flux (phi) * QE
        (eta)), in a pixel.  >= 0.

    darke : float
        dark current, in electrons per second.  >= 0.

    cic : float
        clock-induced charge, in electrons.  >= 0.

    alpha0 : float
        fraction of the per-pixel full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.  Given the
        modeling experience that acquisition and capture can cause intermittent
        bright frames that exceed the aligned-mask maximum, it is recommended
        this be kept to ~0.7 to avoid saturating.

    fwc : int
        number of electrons in the per-pixel full well.  >= 1.

    alpha1 : float
        fraction of the EM gain full well to allow a frame to use.  This
        will be >= 0 and <= 1.  Using a value less than 1 prevents saturation
        and helps to keep the number of counts in the linear regime.  Given the
        modeling experience that acquisition and capture can cause intermittent
        bright frames that exceed the aligned-mask maximum, it is recommended
        this be kept to ~0.7 to avoid saturating.

    fwc_em : int
        number of electrons in the EM gain full well.  >= 1.

    g_max_comm : float
        maximum gain allowed in the command to set LOCAM gain.  This is a pure
        restriction on allowed values in CGI FSW and does not necessarily
        incorporate other physics.  >= 1.

    g_max_age : float
        maximum gain allowed to be used with LOCAM, based on lifetime/aging
        tests.  >= 1.

    e_max_age : int
        maximum number of electrons permitted, per pixel per LOCAM frame, based
        on lifetime/aging tests.  >= 1.

    tframe : float
        Exposure time per LOCAM, in seconds/frame.  >= 0.  This is a fixed
        property of the readout sequence.

    n : float
        number of standard deviations the signal after gain is below the max
        fwc_em.  >= 0.

    Returns
    -------
    gain : float
        Optimal gain setting for detector

    code : int
        value from the following list:
          1: no feasible non-unity gain solution exists due to aging
             considerations, but unity gain operation is feasible
          2: feasible non-unity gain found, constraint 2 set limit
          3: feasible non-unity gain found, constraint 3 set limit
          5: feasible non-unity gain found, constraint 5 set limit
          6: feasible non-unity gain found, constraint 6 set limit
        In the case where two or more constraints are met simultaneously, the
        lowest number will be output.  In the vanishingly-unlikely scenario
        where the only numerical value permitted by aging constraints is 1
        (but it's still permitted--no violation), the appropriate value in
        [2, 3, 5, 6]  will be returned.  If unity-gain operation is infeasible,
        this function will raise a LOCAMOptimizeException (and not return)

    """

    # Check inputs
    check.real_nonnegative_scalar(fluxe_bright, 'fluxe_bright', TypeError)
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

    # Check for constraint 4 feasibility, raise exception if violated
    e_preread = fluxe_bright*tframe + darke*tframe
    sigma_preread = np.sqrt(e_preread)
    if e_preread + n*sigma_preread > alpha0*fwc:
        raise LOCAMOptimizeException('Constraints are internally ' +
                'infeasible (too many electrons for full well)')

    # Rewrite constraints 5 and 6 in "g <= X" form
    ENF_gain = np.sqrt(2)
    e_serial = e_preread + cic
    sigma_serial = ENF_gain*np.sqrt(e_serial)
    if e_serial + n*sigma_serial == 0:
        X_5 = np.inf
        X_6 = np.inf
        pass
    else:
        X_5 = (alpha1*fwc_em)/(e_serial + n*sigma_serial)
        X_6 = (e_max_age)/(e_serial + n*sigma_serial)
        pass

    # Select the smallest of X_2, X_3, X_5, and X_6.
    X_2 = g_max_comm
    X_3 = g_max_age
    X = min(X_2, X_3, X_5, X_6)

    # If the smallest of those values is >= 1, return that value as the
    # expected gain.
    if X >= 1:
        if X == X_2:
            return X, 2
        elif X == X_3:
            return X, 3
        elif X == X_5:
            return X, 5
        elif X == X_6:
            return X, 6
        else: # should never reach here
            raise LOCAMOptimizeException('Gain value corrupted, bug present')
    # Otherwise, if the smallest of those X values is less than 1:
    else:
        # Select the smallest of X_2 and X_5 (the constraints that apply at
        # g = 1, as unity gain does not cause aging)
        sigma_serial_unity = np.sqrt(e_serial) # no ENF
        if (e_serial + n*sigma_serial_unity) == 0:
            X_5a = np.inf
            pass
        else:
            X_5a = (alpha1*fwc_em)/(e_serial + n*sigma_serial_unity)
            pass
        reX = min(X_2, X_5a)

        # If the smallest of those two is less than 1, then the problem is not
        # feasible at any gain.  Raise an exception.
        if reX < 1:
            raise LOCAMOptimizeException('Problem infeasible at any choice ' +
                                         'of gain')
        # Otherwise, only viable gain is 1; return that.
        else:
            return 1, 1 # 1 as status code when aging constraints will never
                        # give a viable answer

    pass



if __name__ == "__main__":
    # testing only
    ap = argparse.ArgumentParser(prog='python locam_tools.py',
                                 description="Compute LOCAM gain")
    ap.add_argument('--flux', default=20, type=float,
                    help="Electrons from photon flux per pixel per frame. " +
                    "Default=20.")
    args = ap.parse_args()

    _tframe = 0.000441 # < 1ms, rest is readout time
    gain, status = calc_locam_gain(
            fluxe_bright=args.flux/_tframe, # e/pix/fr -> e/s
            darke=8.33e-4,
            cic=0.02,
            alpha0=0.70,
            fwc=50000,
            alpha1=0.70,
            fwc_em=90000,
            g_max_comm=7500,
            g_max_age=200,
            e_max_age=12800,
            tframe=_tframe,
            n=4,
    ) # inputs from locam_config.yaml

    print('Gain = ' + str(gain))
    print('Status code = ' + str(status))
