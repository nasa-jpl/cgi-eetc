# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Top level interface for engineering exposure time calculator.
"""
import os
import argparse
import warnings
from pathlib import Path
import logging

import numpy as np
import yaml

import eetc
import eetc.util.check as check
from eetc.excam_tools import (calc_gain_exptime, calc_gain_fixed_N,
                              calc_gain_fixed_time, calc_gain_fixed_g,
                              calc_gain_fixed_Ntime,
                              calc_pc, calc_pc_fixed_N,
                              calc_pc_gain_fixed_Ntime, _ENF, _SNR_CR_resel,
                              _SNR_CR_pc_resel, EXCAMOptimizeException)
from eetc.locam_tools import calc_locam_gain
from eetc.load import (load_sequences, load_thpt_configs, load_excam_config,
                       load_locam_config,
                       load_flux_grid, load_wave_grid)
from eetc.load import _unpack_excam_config, _unpack_locam_config
from eetc.thpt_tools import calc_thpt
from eetc.util.validate import validate_dict_keys
from eetc.constants import valid_thpt_data_keys

LOCAL_PATH = eetc.lib_dir
POINTER_PATH = str(Path(LOCAL_PATH, 'pointer.yaml'))

pointer_defaults = {
    'sequences':str(Path(LOCAL_PATH, 'config/sequences.yaml')),
    'thpt_configs':str(Path(LOCAL_PATH, 'config/thpt_configs.yaml')),
    'thpt_coatings':str(Path(LOCAL_PATH, 'config/thpt_coatings.yaml')),
    'thpt_data':str(Path(LOCAL_PATH, 'config/thpt_data.yaml')),
    'thptcurves_dir':str(Path(LOCAL_PATH, 'thptcurves/')),
    'excam_config':str(Path(LOCAL_PATH, 'config/excam_config.yaml')),
    'locam_config':str(Path(LOCAL_PATH, 'config/locam_config.yaml')),
    'flux_grid':str(Path(LOCAL_PATH,
                         'flux_grid_generation/grid_files/flux_grid.fits')),
    'wave_grid':str(Path(LOCAL_PATH,
                         'flux_grid_generation/grid_files/wave_grid.fits')),
}

pointer_keys = list(pointer_defaults.keys())


# The SLSQP optimizer sometimes has known internal weirdness about bounds and
# scipy will raise a warning that we can't do anything about.  Filter it out.
warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module='scipy.optimize')
#use logging for standard outs concerning optflag
log = logging.getLogger(__name__)

class CGIEETC:
    """
    Class with methods for calculating flux rate and required exposure time for
    CGI calibration target based on precomputed flux rate grid and specified
    OTA + CGI throughput curves.

    Requires reference flux rate grid, with flux rates computed for each CFAM
    filter across a range of stellar spectral types.

    Parameters
    ----------
    mag : float
        Calibration target magnitude.

    phot : str
        Photometric filter. Valid values are {'b', 'v', 'r', 'i'}.

    spt : str
        Calibration target stellar spectral type. See constants.py for
        valid types.

    pointer_path : str
        Full path of pointer yaml file.

        Pointer file contains the paths of the other files needed
        internally by CGIEETC. Paths are either absolute, relative to the
        location of the pointer file, or None (which is input with any of the
        values in https://yaml.org/type/null.pdf).  If None, it uses default
        values built into the repository.  Defaults to the file delivered with
        this repository (pointer.yaml).

        Must be a yaml file with the following keys and values:

        sequences : str
            Path to file containing sequences settings for PAMS.

        thpt_configs : str
            Path to file containing observing mode static optic
            configurations.

        thpt_coatings: str
            Path to file coating paths to throughput curves associated with
            coatings used by thpt_configs

        thpt_data : str
            Relative path to file containing observing mode optic
            configurations.

        thptcurves_dir : str
            Path to directory containing throughput curve files.

        excam_config : str
            Path to file with EXCAM CCD properties.

        locam_config : str
            Path to file with LOCAM CCD properties.

        flux_grid : str
            Path to file containing flux grid.

        wave_grid : str
            Path to file containing flux-weighted wavelength grid.

    S Halverson - JPL - 29-Sep-2019
    S Miller - UAH - 1-Dec-2021

    """

    def __init__(self, mag, phot, spt, pointer_path=POINTER_PATH):
        # Check inputs
        check.real_scalar(mag, 'mag', TypeError)
        check.string(phot, 'phot', TypeError)
        check.string(spt, 'spt', TypeError)
        check.string(pointer_path, 'pointer_path', TypeError)

        self.mag = mag
        self.phot = phot.lower() #so that input spt can be case-insensitive
        self.spt = spt.upper()
        self.pointer_path = pointer_path

        # Load pointer file and validate
        try:
            with open(self.pointer_path) as file:
                self.pointer = yaml.safe_load(file)
        except FileNotFoundError:
            raise IOError('pointer file not found')
        validate_dict_keys(self.pointer, pointer_keys)
        self.pointer_dir = os.path.abspath(os.path.dirname(self.pointer_path))

        # Unpack file paths from pointer dictionary, validate, and make
        # absolute
        for key in pointer_keys:
            if self.pointer[key] is None:
                self.pointer[key] = pointer_defaults[key]
                pass
            check.string(self.pointer[key], key, TypeError)
            tmp = os.path.expandvars(self.pointer[key])
            if not os.path.isabs(tmp):
                self.pointer[key] = os.path.join(self.pointer_dir, tmp)
                pass
            else:
                self.pointer[key] = tmp
                pass
            pass

        # Load config files
        self.coating_configs, self.coating_thptcurves, \
            self.setting_thptcurves = \
            load_thpt_configs(
                self.pointer['thpt_configs'],
                self.pointer['thpt_coatings'],
                self.pointer['thpt_data'],
                self.pointer['thptcurves_dir'],
            )

        self.excam_config = load_excam_config(self.pointer['excam_config'])
        self.locam_config = load_locam_config(self.pointer['locam_config'])

        # Load flux grid file
        self.flux_grid = load_flux_grid(self.pointer['flux_grid'])
        self.wave_grid = load_wave_grid(self.pointer['wave_grid'])

        # Unpack values from flux grid
        if phot.lower() not in self.flux_grid:
            raise KeyError('invalid phot')
        self.flux_grid_vals, self.flux_grid_spts, self.flux_grid_cfams, \
            self.ref_mag = self.flux_grid[self.phot]
        self.wave_grid_vals, self.wave_grid_spts, self.wave_grid_cfams, \
            self.ref_mag_wave = self.wave_grid[self.phot]

        # check spectral types with values in flux grid
        if spt.upper() not in self.flux_grid_spts:
            raise KeyError('invalid spt')

        # Sequences can use any mode we set up the mirrors for
        self.valid_modes = set(self.coating_configs.keys())
        self.valid_spam_lsam = set(self.setting_thptcurves['spam_lsam'].keys())
        self.valid_fpam = set(self.setting_thptcurves['fpam'].keys())
        self.valid_fsam = set(self.setting_thptcurves['fsam'].keys())
        self.valid_dpam = set(self.setting_thptcurves['dpam'].keys())

        self.sequences = load_sequences(
            self.pointer['sequences'],
            self.valid_modes,
            self.valid_spam_lsam,
            self.valid_fpam,
            self.valid_fsam,
            set(self.wave_grid_cfams), # list to validate against: from grid
            self.valid_dpam,
        )

        # Calculate effective wavelengths given stellar type
        self.eff_cfams = dict()
        for cfam in self.wave_grid_cfams:
            self.eff_cfams[cfam] = self._load_wave_val(cfam)
            pass



    def calc_flux_rate(self, sequence_name, manual=1.0):
        """
        Estimates flux rate for a CGI calibration target based on precomputed
        flux rate grid and specified OTA + CGI throughput curves.

        Parameters
        ----------
        sequence_name : str
            Name of optical settings observing sequence.

        manual : float
            Manual fractional scale factor on flux rate, relative to the rate
            defined by mag at class instantiation.  Positive values increase
            flux (e.g. manual=10 will give 10x the flux of manual=1). Defaults
            to 1.0 (no scaling).


        Returns
        -------
        flux_rate : float
            Integrated flux rate in specified CFAM filter at focal plane,
            with quantum efficiency incorporated
            (photoelectrons / second).

        flux_rate_peak_pix : float
            Peak flux rate in a pixel in specified CFAM filter at focal plane,
            with quantum efficiency incorporated
            (photoelectrons / second).

        """
        # Check inputs
        check.string(sequence_name, 'sequence_name', TypeError)
        check.real_nonnegative_scalar(manual, 'manual', TypeError)

        sequence_name = sequence_name.upper()
        if sequence_name not in self.sequences:
            raise KeyError('unknown sequence')
        this_seq = self.sequences[sequence_name]

        # Unpack sequence dictionary for given sequence name
        element_thptcurves = dict()
        for element in valid_thpt_data_keys:
            # For each element, unpack throughput curve corresponding to the
            # specified setting
            curve = this_seq[element]
            element_thptcurves[element] = \
                self.setting_thptcurves[element][curve]
            pass

        # Unpack coating configuration for given mode
        coating_config = self.coating_configs[this_seq['mode']]
        # Unpack center wavelength for given CFAM
        cfam_lam = self.eff_cfams[this_seq['cfam']]

        # Calculate throughput for given coating configuration, optical element
        # settings, and CFAM center wavelength
        thpt = calc_thpt(self.coating_thptcurves, element_thptcurves,
                         coating_config, cfam_lam)

        # Calculate flux rate at top of telescope through specified CFAM filter
        flux_rate = self._load_flux_val(this_seq['cfam'])  # ph/sec

        # Scale flux by throughput
        flux_rate *= thpt # e-/sec

        # Scale flux by manual scale factor
        flux_rate *= manual

        # Calculate max expected photo-e's/pix/second in image based on flux
        # ratio psf value stored in sequence
        flux_rate_peak_pix = flux_rate*this_seq['peak_flux_ratio_pix']

        return flux_rate, flux_rate_peak_pix

    def _load_flux_val(self, cfam):
        """
        Grab relevant flux rate at entrance to OTA through specified CFAM filter
        from flux grid.

        Parameters
        ----------
        cfam : str
            Name of CFAM filter being observed through.

        Returns
        -------
        flux_rate_cfam : float
            Estiamted flux rate at entrance to OTA through specified CFAM
            filter (photons / sec).

        """
        # Unpack flux values for given CFAM
        flux_vals = self.flux_grid_vals[:, self.flux_grid_cfams.index(cfam)]

        # get appropriate flux value for provided spectral type
        flux_val = flux_vals[self.flux_grid_spts.index(self.spt)]

        # Scale to specific magnitude
        delta_mag = self.ref_mag - self.mag
        #using definition for change of stellar magnitude:
        flux_rate_cfam = flux_val * 10. ** (delta_mag * 0.4)  # photons/second

        return flux_rate_cfam

    def _load_wave_val(self, cfam):
        """
        Grab relevant flux-weighted-center wavelength for specified CFAM filter
        from wavelength grid.

        Parameters
        ----------
        cfam : str
            Name of CFAM filter being observed through.

        Returns
        -------
        wave_cfam : float
            Estimated averaged wavelength specified CFAM
            filter (Angstrom).

        """
        # Unpack flux values for given CFAM
        wave_vals = self.wave_grid_vals[:, self.wave_grid_cfams.index(cfam)]

        # get appropriate flux value for provided spectral type
        wave_cfam = wave_vals[self.wave_grid_spts.index(self.spt)]

        return wave_cfam

    def excam_SNR(self, sequence_name, g, exptime, nframes, fraction=None,
                  num_pixels=None, scale=1., scale_bright=1., manual=1.0,
                  mode='analog', type='resel'):
        '''
        Given an observation sequence, and EM gain, an exposure time, and
        a number of frames, this function returns the SNR for the number of
        pixels specified and uses the exposure time
        calculator (ETC) functions used in excam_tools.py.
        The function also indicates if the input conditions result in
        saturation of the detector and gives the maximum non-saturating
        exposure time given the EM gain. We refer to the SNR for the user's
        region as the SNR per spatial resolution element, or SNR per "resel".
        The defaults for fraction and num_pixels are the numbers for the resel
        for the sequence stored in the sequence YAML file.  For SNR/pixel, use
        type='pixel' (and then fraction and num_pixels are not used).
        The default for type is 'resel'.  If considering a
        photon-counting observation (specified by the 'mode' input), warnings
        separate from the saturation constraints are given if the values of EM
        gain and exposure time are incompatible with photon-counting
        conditions.

        See snr_resel.pdf for details on the methodology used.

        The ETC may return values that fall outside the allowed boundaries
        slightly due to the parameter delta_constr utilized in the functions in
        excam_tools.py.  But the SNR returned by this function should be very
        close to the SNR for the same parameters in the ETC functions.

        Parameters
        ----------
        sequence_name : str
            Name of optical settings observing sequence.

        g : float
            Desired EM gain value for the observation. Must be >=1 and <= gmax,
            where gmax is from excam_config.yaml.

        exptime : float
            Desired exposure time per frame for the observation, in s.  Must be
            >= tmin and <= tmax, where tmin and tmax are
            from excam_config.yaml.

        nframes : int
            Desired number of frames for the observation.  Must be an an
            integer >= Nmin and <= Nmax, where Nmin and Nmax are from
            excam_config.yaml.

        fraction : float
            Fraction of total image area flux corresponding to resel.  >=0 and
            <=1.  If None, fraction is drawn from its calculated value stored
            in the sequences YAML file.  If the sequence is not intended for
            obtaining a point-spread function (PSF) of a target, there is no
            stored value for the sequence, and an exception will be raised.
            Defaults to None.

        num_pixels : float, optional
            Number of pixels corresponding to resel.  >= 0.  If None,
            num_pixels is drawn from its calculated value stored in the
            sequences YAML file.  If the sequence is not intended for obtaining
            a point-spread function (PSF) of a target, there is no stored value
            for the sequence, and an exception will be raised. For SNR/pixel,
            use num_pixels=1. Defaults to None.

        scale : float, optional
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at a point of interest, to
            allow the calculator to be used for occulted calculations. Defaults
            to 1.0 (no scaling).  For observing an unocculted point object, 1.0
            is the appropriate choice.  Defaults to 1.0.

        scale_bright : float, optional
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at the brightest point in
            a frame, so the camera settings do not overexpose and saturate this
            point.  Used primarily for occulted calculations.  Defaults to 1.0
            (no scaling).  For observing an unocculted point object, 1.0 is the
            appropriate choice.  scale_bright >= scale so that the flux from
            the brightest pixel is greater than or equal to the flux from a
            typical pixel.  Defaults to 1.0.

        manual : float, optional
            Manual fractional scale factor on flux rate, relative to the rate
            defined by mag at class instantiation.  Positive values increase
            flux (e.g. manual=10 will give 10x the flux of manual=1). Defaults
            to 1.0 (no scaling).  Defaults to 1.0.

        mode : str
            If 'analog', SNR returned is for the case of an analog observation.
            If 'pc', SNR returned is for the case of a photon-counting
            observation.  Defaults to 'analog'.

        type : str
            If 'resel', SNR returned is for the resel.
            If 'pixel', SNR returned is per pixel, and the inputs fraction
            and num_pixels are ignored.  Defaults to 'resel'.

        Returns
        -------
        SNR : float
            SNR given the input parameters.

        etc_max_time : float
            Maximum exposure time allowed by the ETC constraints given the
            sequence and gain inputs.  This time does not take into account
            violations of constraints specific to photon counting that do not
            relate to saturation.  These violations are instead indicated only
            by warnings.

        etc_max_time_status : str
            String indicating the constraint that most directly bounded the
            maximum exposure time, etc_max_time.
            'per-pixel well':  etc_max_time meets the per-pixel fractional well
            'EM gain well':  etc_max_time meets the EM gain fractional well
            'tmax':  etc_max_time meets the maximum allowed ETC exposure time
        '''
        # Unpack values from excam configuration
        darke, cic, rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em, \
            Nmin, Nmax, tmin, tmax, gmax, _, n, Nem, tol, \
            delta_constr, _, pc_ecount_max, T_factor = \
            _unpack_excam_config(self.excam_config)

        # check output from _unpack_excam_config() since they aren't checked in
        # function calls below (copied and pasted from load.load_excam_config,
        # except I leave out gconst, which is not used here)
        check.real_nonnegative_scalar(darke, 'darke', TypeError)
        check.real_nonnegative_scalar(cic, 'cic', TypeError)
        check.real_nonnegative_scalar(rn, 'rn', TypeError)
        check.real_nonnegative_scalar(X, 'X', TypeError)
        check.real_nonnegative_scalar(a, 'a', TypeError)
        check.positive_scalar_integer(Lij, 'Lij', TypeError)
        check.real_positive_scalar(alpha0, 'alpha0', TypeError)
        if alpha0 > 1:
            raise ValueError('alpha0 must be between 0 and 1 '+
                             '(fraction of total)')
        check.positive_scalar_integer(fwc, 'fwc', TypeError)
        check.real_positive_scalar(alpha1, 'alpha1', TypeError)
        if alpha1 > 1:
            raise ValueError('alpha1 must be between 0 and 1 '+
                             '(fraction of total)')
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
        check.real_nonnegative_scalar(n, 'n', TypeError)
        check.positive_scalar_integer(Nem, 'Nem', TypeError)
        check.real_nonnegative_scalar(tol, "tol", TypeError)
        check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)
        check.real_positive_scalar(pc_ecount_max, 'pc_ecount_max', TypeError)
        check.real_positive_scalar(T_factor, 'T_factor', TypeError)

        # check this function's inputs
        check.string(sequence_name, 'sequence_name', TypeError)
        check.real_positive_scalar(g, 'g', TypeError)
        if g < 1:
            raise ValueError("g cannot be less than 1.")
        if g > gmax:
            raise ValueError("g cannot be greater than ", gmax, ".")
        check.real_positive_scalar(exptime, 'exptime', TypeError)
        if exptime < tmin:
            raise ValueError('exptime cannot be less than tmin.')
        if exptime > tmax:
            raise ValueError('exptime cannot be greater than tmax.')
        check.positive_scalar_integer(nframes, 'nframes', TypeError)
        if nframes < Nmin:
            raise ValueError('nframes cannot be less than Nmin.')
        if nframes > Nmax:
            raise ValueError('nframes cannot be greater than Nmax.')
        check.real_positive_scalar(scale, 'scale', TypeError)
        check.real_positive_scalar(scale_bright, 'scale_bright', TypeError)
        check.real_nonnegative_scalar(manual, 'manual', TypeError)
        if mode != 'analog' and mode != 'pc':
            raise ValueError('Mode must be either \'analog\' or \'pc\'.')
        if type != 'resel' and type != 'pixel':
            raise ValueError('Type must be either \'resel\' or \'pixel\'.')

        if type == 'resel':
            if fraction is None:
                fraction = self.sequences[sequence_name]['fraction']
                if fraction is None:
                    raise ValueError('This sequence is not intended for '
                                     'obtaining a PSF and has no value for '
                                     'fraction.')
            check.real_positive_scalar(fraction, 'fraction', TypeError)
            if fraction > 1:
                raise ValueError('fraction must be <= 1.')
            if num_pixels is None:
                num_pixels = self.sequences[sequence_name]['num_pixels']
                if num_pixels is None:
                    raise ValueError('This sequence is not intended for '
                                     'obtaining a PSF and has no value for '
                                     'num_pixels.')
            check.real_nonnegative_scalar(num_pixels, 'num_pixels', TypeError)

            # Calculate flux rate in photoelectrons per second
            total_flux, flux_rate_peak_pix = self.calc_flux_rate(sequence_name,
                                                                        manual)

            # flux in the resel
            flux_resel = total_flux * fraction
            # now apply optional 'scale' to resel flux
            flux_resel_scale = flux_resel * scale
            # average flux per resel pixel
            fluxe = flux_resel_scale/num_pixels

            # apply optional 'scale_bright' to brightest pixel (for
            # saturation constraints)
            fluxe_bright = flux_rate_peak_pix * scale_bright

            if fluxe_bright < fluxe:
                raise ValueError('The combination of fraction, num_pixels, '
                    'scale, and scale_bright is '
                    'not accurate; the brightest pixel\'s flux was lower than '
                    'flux per pixel in the resel')

        elif type == 'pixel':
            # fraction and num_pixels irrelevant in this case; not checked
            # num_pixels should be set to 1 in this case
            num_pixels = 1.
            if scale_bright < scale:
                raise ValueError('scale_bright must be >= scale if '
                                 'type=\'pixel\'.')

            # Calculate flux rate in photoelectrons per second
            _, flux_rate_peak_pix = self.calc_flux_rate(sequence_name, manual)

            # Calculate maximum expected flux rate per pixel. Include optional
            # scale factors here
            fluxe = flux_rate_peak_pix * scale

        # analog case
        if mode == 'analog':
            SNR = _SNR_CR_resel(g, exptime, nframes, fluxe, darke, cic, rn, X,
                                a, Lij, 1, Nem, num_pixels)
        #photon-counting case
        if mode == 'pc':
            if fluxe > 0:
                t_pcmax = (pc_ecount_max)/(fluxe)
            elif fluxe == 0:
                t_pcmax = np.inf
            t_ub = np.min(np.array([tmax, t_pcmax]))
            # threshold, T; need not be 5*cic since we are subtracting pc darks
            T = T_factor*rn
            g_lb = np.max(np.array([1, T]))
            if g_lb >= gmax:
                warnings.warn("No finite-width window of viable EM " +
                'gain for photon counting.  This indicates that the '
                'read noise and/or T_factor from excam_config.yaml ' +
                'is too high or that gmax is too low.')
            if tmin >= t_ub:
                warnings.warn('No finite-width window of viable frame ' +
                'time for photon counting.  Either tmin >= tmax (which would '+
                'have raised an exception already) or tmin ' +
                'is >= max exppsure time allowed for photon counting, which ' +
                'is the max e- count allowed divided by the flux ' +
                '(pc_ecount_max/fluxe). pc_ecount_max is given ' +
                'in excam_config.yaml.')
            if g < g_lb:
                warnings.warn('g is less than the acceptable lower limit for '
                              'EM gain for photon counting for these inputs, '
                              'which is {}'.format(g_lb))
            if exptime > t_ub:
                warnings.warn('exptime is above the acceptable upper limit '
                              'for exposure time for photon counting for '
                              'these inputs, which is {}'.format(t_ub))
            try:
                SNR = _SNR_CR_pc_resel(g, exptime, nframes, fluxe, darke, cic,
                                  T, X, a, Lij, 1, num_pixels)
            except:
                raise OverflowError('The input parameters are not appropriate '
                                'for photon counting.  Most likely, g is '
                                'too low, which can cause an overflow error '
                                'in the SNR calculation for photon '
                                'counting.  The minimum gain appropriate for '
                                'photon counting in this case is {}, and '
                                'the maximum appropriate exposure time '
                                'is {}'.format(g_lb, t_ub))

        # f irrelvant in the call below; we just need the last 2 outputs
        _, _, etc_max_time, etc_max_time_status = \
            self.excam_saturation_time(sequence_name, g, f=1.,
                                     scale_bright=scale_bright, manual=manual)
        if exptime > etc_max_time:
            warnings.warn('For the given g, exptime saturates the EXCAM!')

        return SNR, etc_max_time, etc_max_time_status


    def excam_saturation_time(self, sequence_name, g, f=1., scale_bright=1.,
                        manual=1.0):
        """
        Given an observation sequence, an EM gain, a fraction f of the full
        well (applied to both per-pixel and EM gain full wells), this function
        returns the exposure time needed to reach the fractional full well,
        regardless of the number of pixels in the user's desired region of
        the detector.  It also returns the maximum exposure time
        allowed by the exposure time
        calculator (ETC) functions used in excam_tools.py, which is based on
        the parameters specified in excam_config.yaml.  The ETC conservatively
        does not allow the number of electrons per pixel to reach above
        n standard deviations of alpha*full well, where alpha is a scalar
        between 0 and 1, and n is a positve scalar.  Both are specified by
        excam_config.yaml.  So it is possible that the time needed to reach
        the user's fractional full well value is not allowed by the ETC.
        For each of this function's returns, a string is returned that
        indicates what condition was met with the time
        (met the per-pixel fractional well, EM gain fractional
        well, or met the maximum allowed time in the case of the ETC time).

        The ETC may return values that fall outside the allowed boundaries
        slightly due to the parameter delta_constr utilized in the functions in
        excam_tools.py.  But the time returned by this function should be very
        close to the maximum time constrained by saturation coming from the ETC
        functions.

        Note:  This function does not consider the additional constraints on
        exposure time and gain in the case of a photon-counting observation.
        The function excam_SNR(), however, does provide these guard rails when
        the photon-counting mode is selected in that function.

        Parameters
        ----------
        sequence_name : str
            Name of optical settings observing sequence.

        g : float
            Desired EM gain value for the observation. Must be >=1 and <= gmax,
            where gmax is from excam_config.yaml.

        f : float
            Desired fraction of the full wells to use to calculate the exposure
            time.  Must be positive and less than or equal to 1.  Defaults to
            1.

        scale_bright : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at the brightest point in
            a frame, so the camera settings do not overexpose and saturate this
            point.  Used primarily for occulted calculations.  Defaults to 1.0
            (no scaling).  For observing an unocculted point object, 1.0 is the
            appropriate choice.

        manual : float
            Manual fractional scale factor on flux rate, relative to the rate
            defined by mag at class instantiation.  Positive values increase
            flux (e.g. manual=10 will give 10x the flux of manual=1). Defaults
            to 1.0 (no scaling).

        Returns
        -------
        max_time : float
            Exposure time needed to reach the fractional full well, without
            regard to ETC constraints.

        max_time_status : str
            'per-pixel well':  max_time meets the per-pixel fractional well
            'EM gain well':  max_time meets the EM gain fractional well

        etc_max_time : float
            Maximum exposure time allowed by the ETC constraints given the
            sequence and gain inputs.

        etc_max_time_status : str
            'per-pixel well':  etc_max_time meets the per-pixel fractional well
            'EM gain well':  etc_max_time meets the EM gain fractional well
            'tmax':  etc_max_time meets the maximum allowed ETC exposure time
        """
        # Unpack values from excam configuration
        darke, cic, rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em, \
            Nmin, Nmax, tmin, tmax, gmax, _, n, Nem, tol, \
            delta_constr, _, _, _ = _unpack_excam_config(self.excam_config)

        # check output from _unpack_excam_config() since they aren't checked in
        # function calls below (copied and pasted from load.load_excam_config,
        # except I leave out gconst, pc_ecount_max, and T_factor, which are
        # not used here)
        check.real_nonnegative_scalar(darke, 'darke', TypeError)
        check.real_nonnegative_scalar(cic, 'cic', TypeError)
        check.real_nonnegative_scalar(rn, 'rn', TypeError)
        check.real_nonnegative_scalar(X, 'X', TypeError)
        check.real_nonnegative_scalar(a, 'a', TypeError)
        check.positive_scalar_integer(Lij, 'Lij', TypeError)
        check.real_positive_scalar(alpha0, 'alpha0', TypeError)
        if alpha0 > 1:
            raise ValueError('alpha0 must be between 0 and 1 '+
                             '(fraction of total)')
        check.positive_scalar_integer(fwc, 'fwc', TypeError)
        check.real_positive_scalar(alpha1, 'alpha1', TypeError)
        if alpha1 > 1:
            raise ValueError('alpha1 must be between 0 and 1 '+
                             '(fraction of total)')
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
        check.real_nonnegative_scalar(n, 'n', TypeError)
        check.positive_scalar_integer(Nem, 'Nem', TypeError)
        check.real_nonnegative_scalar(tol, "tol", TypeError)
        check.real_nonnegative_scalar(delta_constr, "delta_constr", TypeError)

        # Check this function's inputs
        check.string(sequence_name, 'sequence_name', TypeError)
        check.real_positive_scalar(g, 'g', TypeError)
        if g < 1:
            raise ValueError("g cannot be less than 1.")
        if g > gmax:
            raise ValueError("g cannot be greater than ", gmax, ".")
        check.real_positive_scalar(f, 'f', TypeError)
        if f > 1:
            raise ValueError('f cannot be greater than 1.')
        check.real_positive_scalar(scale_bright, 'scale_bright', TypeError)
        check.real_nonnegative_scalar(manual, 'manual', TypeError)

        # Calculate flux rate in photoelectrons per second
        _, flux_rate_peak_pix = self.calc_flux_rate(sequence_name, manual)

        # Calculate maximum expected flux rate per pixel. Include optional
        # scale factors here
        fluxe_bright = flux_rate_peak_pix * scale_bright

        # max time from emrail constraint, from Eq. 40 of etc_snr_v3b.pdf in
        # doc folder
        emrail_t = (2*alpha1*fwc_em - 2*cic*g + _ENF(g, Nem)**2*g*n**2 -
            g*np.sqrt(4*alpha1*_ENF(g, Nem)**2*fwc_em*n**2/g +
            _ENF(g, Nem)**4*n**4))/(2*g*(fluxe_bright+darke))
        # max time from rail constraint, from Eq. 39 of etc_snr_v3b.pdf in
        # doc folder
        rail_t = (2*alpha0*fwc + n*(n - np.sqrt(4*alpha0*fwc + n**2)))/(2*
                (fluxe_bright+darke))
        # max time considering ETC constraints
        etc_max_time = min(tmax, emrail_t, rail_t)
        if etc_max_time == tmax:
            etc_max_time_status = 'tmax'
        if etc_max_time == rail_t:
            etc_max_time_status = "per-pixel well"
        # this equation last since more likely than rail_t
        if etc_max_time == emrail_t:
            etc_max_time_status = 'EM gain well'

        # max time until fraction of fwc, from Eq. 39 of etc_snr_v3b.pdf in
        # doc folder, in the case where n=0 and alpha0 is f
        fwc_t = f*fwc/(fluxe_bright + darke)
        # max time until fraction of fwc_em, from Eq. 40 of etc_snr_v3b.pdf in
        # doc folder, in the case where n=0 and alpah1 is f
        fwcem_t = (f*fwc_em - cic*g)/(g*(fluxe_bright + darke))
        # max time without considering ETC constraints
        max_time = min(fwc_t, fwcem_t)
        if max_time == fwc_t:
            max_time_status = 'per-pixel well'
        if max_time == fwcem_t:
            max_time_status = 'EM gain well'

        return max_time, max_time_status, etc_max_time, etc_max_time_status

    def calc_const_int_time(self, sequence_name, t_tot, scale=1.,
                            scale_bright=1., manual=1.0, hard_limit=True):
        """
        For a given total integration time, this function estimates
        required exposure time for CGI calibration target based
        on precomputed flux rate grid and specified OTA + CGI throughput
        curves.

        There is no input for target SNR for this function since the usual
        first optimization of minimizing the total integration time is
        irrelevant since that time is fixed.  This function simply tries to
        maximize the SNR.

        Parameters
        ----------
        sequence_name : str
            Name of optical settings observing sequence.

        t_tot : float
            The fixed total integration time (i.e., the number of frames times
            the exposure time per frame).  >= 0.

        scale : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at a point of interest, to
            allow the calculator to be used for occulted calculations. Defaults
            to 1.0 (no scaling).  For observing an unocculted point object, 1.0
            is the appropriate choice.

        scale_bright : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at the brightest point in
            a frame, so the camera settings do not overexpose and saturate this
            point.  Used primarily for occulted calculations.  Defaults to 1.0
            (no scaling).  For observing an unocculted point object, 1.0 is the
            appropriate choice.  scale_bright >= scale so that the flux from
            the brightest pixel is greater than or equal to the flux from a
            typical pixel.

        manual : float
            Manual fractional scale factor on flux rate, relative to the rate
            defined by mag at class instantiation.  Positive values increase
            flux (e.g. manual=10 will give 10x the flux of manual=1). Defaults
            to 1.0 (no scaling).

        hard_limit: boolean
            True if a hard limit needed for t_tot.  The number of frames (N)
            has to be treated as a float in the optimization calculation, and
            the frame time (t) is optimized under that assumption, but the
            optimized output N will be rounded up to the next integer.
            If the user wants the optimized N*t to equal the input t_tot
            exactly, hard_limit should be True, and the output t is equal to
            t_tot/N.  In this case, the optimized SNR output may be a bit
            smaller than what it could have been with the optimized float
            value of N.  If the user is okay with a little variation
            between N*t and t_tot, then hard_limit should be False, and then
            the optimized SNR will be as big as possible since N is rounded up
            and t is the value from the optimization, and N*t may be slightly
            bigger than t_tot.  Defaults to True.

        Returns
        -------
        n_frames : float
            Total number of exposures to maximize SNR (snr_actual).

        exp_time : float
            Exposure time for each frame in n_frames to reach snr_actual (s).

        gain : float
            Optimal gain setting for detector.

        snr_actual : float
            Expected SNR per pixel when using above exposure settings.

        t_tot_out : float
            Actual total integration time used, exp_time*n_frames.  If
            hard_limit is True, this should equal the input t_tot.
        """
        # Check inputs
        check.string(sequence_name, 'sequence_name', TypeError)
        check.real_nonnegative_scalar(t_tot, 't_tot', TypeError)
        check.real_positive_scalar(scale, 'scale', TypeError)
        check.real_positive_scalar(scale_bright, 'scale_bright', TypeError)
        if scale_bright < scale:
            raise ValueError('scale_bright must be >= scale.')
        check.real_nonnegative_scalar(manual, 'manual', TypeError)
        if not isinstance(hard_limit, bool):
            raise TypeError('hard_limit must be boolean.')

        # Unpack values from excam configuration
        darke, cic, rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em, \
            Nmin, Nmax, tmin, tmax, gmax, gconst, n, Nem, tol, \
            delta_constr, overhead, _, _ \
            = _unpack_excam_config(self.excam_config)

        # Fixed parameters (max val = min val) must be handled separately as
        # finite derivatives in the optimizer will throw up on them
        fixedg = False
        fixedt = False
        fixedn = False
        if Nmin == Nmax:
            fixedn = True
        if tmin == tmax:
            fixedt = True
        if gconst is not None:
            fixedg = True

        if fixedn or fixedt or fixedg:
            raise EXCAMOptimizeException('Unsupported optimization for a ' +
                                         'fixed variable')

        # Calculate flux rate in photoelectrons per second
        _, flux_rate_peak_pix = self.calc_flux_rate(sequence_name, manual)

        # Calculate maximum expected flux rate per pixel. Include optional
        # scale factors here
        fluxe = flux_rate_peak_pix * scale
        fluxe_bright = flux_rate_peak_pix * scale_bright

        # Calculate optimum gain, exposure time, and number of frames
        gain, exp_time, n_frames, snr_actual, t_tot_out = \
            calc_gain_fixed_Ntime(t_tot, fluxe, fluxe_bright, darke, cic,
                                rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                Nmin, Nmax, tmin, tmax, gmax, overhead,
                                n=n, Nem=Nem,
                                tol=tol, delta_constr=delta_constr,
                                hard_limit=hard_limit)

        log.warning('Output total integration time:  %s', t_tot_out)
        log.warning('SNR per pixel output from optimization:  %s', snr_actual)
        return n_frames, exp_time, gain, snr_actual, t_tot_out


    def calc_const_int_time_resel(self, sequence_name, t_tot, fraction=None,
                                num_pixels=None, scale=1., scale_bright=1.,
                                manual=1.0, hard_limit=True):
        """
        For a given total integration time, this function estimates
        required exposure time for CGI calibration target based
        on precomputed flux rate grid and specified OTA + CGI throughput
        curves.  Inputs are based on the user's desired region
        (e.g., PSF or a subregion of a PSF).  We refer to the SNR for the
        user's region as the SNR per spatial resolution element, or SNR per
        "resel".

        See snr_resel.pdf for details on the methodology used.

        There is no input for target SNR for this function since the usual
        first optimization of minimizing the total integration time is
        irrelevant since that time is fixed.  This function simply tries to
        maximize the SNR.

        Parameters
        ----------
        sequence_name : str
            Name of optical settings observing sequence.

        t_tot : float
            The fixed total integration time (i.e., the number of frames times
            the exposure time per frame).  >= 0.

        fraction : float
            Fraction of total image area flux corresponding to resel.  >=0 and
            <=1.  If None, fraction is drawn from its calculated value stored
            in the sequences YAML file.  If the sequence is not intended for
            obtaining a point-spread function (PSF) of a target, there is no
            stored value for the sequence, and an exception will be raised.
            Defaults to None.

        num_pixels : float
            Number of pixels corresponding to resel.  >= 0. The inputs
            fraction and num_pixels must be accurate.  If they are not, they
            could cause the brightest pixel's flux (used for
            saturation constraints) to be smaller than the flux per pixel in
            the resel, and the function will raise an exception.  If None,
            num_pixels is drawn from its calculated value stored in the
            sequences YAML file.  If the sequence is not intended for obtaining
            a point-spread function (PSF) of a target, there is no stored value
            for the sequence, and an exception will be raised.
            Defaults to None.

        scale : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux of the resel, to
            allow the calculator to be used for occulted calculations. Defaults
            to 1.0 (no scaling).  For observing an unocculted point object, 1.0
            is the appropriate choice.

        scale_bright : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at the brightest pixel in
            a frame, so the camera settings do not overexpose and saturate this
            point.  Used primarily for occulted calculations.  Defaults to 1.0
            (no scaling).  For observing an unocculted point object, 1.0 is the
            appropriate choice.

        manual : float
            Manual fractional scale factor on flux rate, relative to the rate
            defined by mag at class instantiation.  Positive values increase
            flux (e.g. manual=10 will give 10x the flux of manual=1). Defaults
            to 1.0 (no scaling).

        hard_limit: boolean
            True if a hard limit needed for t_tot.  The number of frames (N)
            has to be treated as a float in the optimization calculation, and
            the frame time (t) is optimized under that assumption, but the
            optimized output N will be rounded up to the next integer.
            If the user wants the optimized N*t to equal the input t_tot
            exactly, hard_limit should be True, and the output t is equal to
            t_tot/N.  In this case, the optimized SNR output may be a bit
            smaller than what it could have been with the optimized float
            value of N.  If the user is okay with a little variation
            between N*t and t_tot, then hard_limit should be False, and then
            the optimized SNR will be as big as possible since N is rounded up
            and t is the value from the optimization, and N*t may be slightly
            bigger than t_tot.  Defaults to True.

        Returns
        -------
        n_frames : float
            Total number of exposures to maximize SNR (snr_actual).

        exp_time : float
            Exposure time for each frame in n_frames to reach snr_actual (s).

        gain : float
            Optimal gain setting for detector.

        snr_actual : float
            Expected SNR per resel when using above exposure settings.

        t_tot_out : float
            Actual total integration time used, exp_time*n_frames.  If
            hard_limit is True, this should equal the input t_tot.
        """
        # Check inputs
        check.string(sequence_name, 'sequence_name', TypeError)
        check.real_nonnegative_scalar(t_tot, 't_tot', TypeError)
        if fraction is None:
            fraction = self.sequences[sequence_name]['fraction']
            if fraction is None:
                raise ValueError('This sequence is not intended for obtaining '
                                 'a PSF and has no value for fraction.')
        check.real_positive_scalar(fraction, 'fraction', TypeError)
        if fraction > 1:
            raise ValueError('fraction must be <= 1.')
        if num_pixels is None:
            num_pixels = self.sequences[sequence_name]['num_pixels']
            if num_pixels is None:
                raise ValueError('This sequence is not intended for obtaining '
                                 'a PSF and has no value for num_pixels.')
        check.real_nonnegative_scalar(num_pixels, 'num_pixels', TypeError)
        check.real_positive_scalar(scale, 'scale', TypeError)
        check.real_positive_scalar(scale_bright, 'scale_bright', TypeError)
        if scale_bright < scale:
            raise ValueError('scale_bright must be >= scale.')
        check.real_nonnegative_scalar(manual, 'manual', TypeError)
        if not isinstance(hard_limit, bool):
            raise TypeError('hard_limit must be boolean.')

        # Calculate flux rate in photoelectrons per second
        total_flux, flux_rate_peak_pix = self.calc_flux_rate(sequence_name,
                                                                    manual)

        # flux in the resel
        flux_resel = total_flux * fraction
        # now apply optional 'scale' to resel flux
        flux_resel_scale = flux_resel * scale
        # average flux per resel pixel
        fluxe = flux_resel_scale/num_pixels

        # apply optional 'scale_bright' to brightest pixel (for
        # saturation constraints)
        fluxe_bright = flux_rate_peak_pix * scale_bright

        if fluxe_bright < fluxe:
            raise ValueError('The combination of fraction, num_pixels, ' +
                'scale, and scale_bright is ' +
                'not accurate; the brightest pixel\'s flux was lower than ' +
                'flux per pixel in the resel')

        # Unpack values from excam configuration
        darke, cic, rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em, \
            Nmin, Nmax, tmin, tmax, gmax, gconst, n, Nem, tol, \
            delta_constr, overhead, _, _ \
            = _unpack_excam_config(self.excam_config)

        # Fixed parameters (max val = min val) must be handled separately as
        # finite derivatives in the optimizer will throw up on them
        fixedg = False
        fixedt = False
        fixedn = False
        if Nmin == Nmax:
            fixedn = True
        if tmin == tmax:
            fixedt = True
        if gconst is not None:
            fixedg = True

        if fixedn or fixedt or fixedg:
            raise EXCAMOptimizeException('Unsupported optimization for a ' +
                                         'fixed variable')

        # Calculate optimum gain, exposure time, and number of frames
        gain, exp_time, n_frames, snr_actual, t_tot_out = \
            calc_gain_fixed_Ntime(t_tot, fluxe, fluxe_bright, darke, cic,
                                rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                Nmin, Nmax, tmin, tmax, gmax, overhead,
                                n=n, Nem=Nem,
                                tol=tol, delta_constr=delta_constr,
                                num_pixels=num_pixels, hard_limit=hard_limit)

        log.warning('Output total integration time:  %s', t_tot_out)
        log.warning('SNR per resel output from optimization:  %s', snr_actual)
        return n_frames, exp_time, gain, snr_actual, t_tot_out

    def calc_exp_time(self, sequence_name, snr, scale=1.,
                      scale_bright=1., manual=1.0):
        """
        Estimates required exposure time for CGI calibration target based on
        precomputed flux rate grid and specified OTA + CGI throughput curves.

        Parameters
        ----------
        sequence_name : str
            Name of optical settings observing sequence.

        snr : float
            Desired SNR per pixel to expose to, or None.

            If given a value, the optimization will attempt to get the shortest
            total integration time (number of frames*exposure time per frame)
            which achieves that SNR (opt=0).  If there is no feasible
            combination of gain, exposure time, and number of frames which can
            provide that SNR, or if None is given as an input, the optimization
            will attempt to get the best SNR possible given remaining
            constraints (opt=1), with no constraint on total integration time.
            See below for information on the output 'optflag'.

        scale : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at a point of interest, to
            allow the calculator to be used for occulted calculations. Defaults
            to 1.0 (no scaling).  For observing an unocculted point object, 1.0
            is the appropriate choice.

        scale_bright : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at the brightest point in
            a frame, so the camera settings do not overexpose and saturate this
            point.  Used primarily for occulted calculations.  Defaults to 1.0
            (no scaling).  For observing an unocculted point object, 1.0 is the
            appropriate choice.  scale_bright >= scale so that the flux from
            the brightest pixel is greater than or equal to the flux from a
            typical pixel.

        manual : float
            Manual fractional scale factor on flux rate, relative to the rate
            defined by mag at class instantiation.  Positive values increase
            flux (e.g. manual=10 will give 10x the flux of manual=1). Defaults
            to 1.0 (no scaling).

        Returns
        -------
        n_frames : float
            Total number of exposures to reach snr.

        exp_time : float
            Exposure time for each frame in n_frames to reach snr (s).

        gain : float
            Optimal gain setting for detector.

        snr_actual : float
            Expected SNR per pixel when using above exposure settings.  This
            may be higher than the target SNR input from the user ('snr')
            because the optimization assumes a floating-point number for the
            number of frames (N) and then rounds up to the nearest integer
            value, which would increase the SNR value if there was an increase
            in N in the rounding.

        optflag : int
            0 or 1: 0 if the first optimization succeeded, 1 if the first
            failed but the second succeeded.

        """
        # Check inputs
        check.string(sequence_name, 'sequence_name', TypeError)
        if snr is not None:
            check.real_positive_scalar(snr, 'snr', TypeError)
        check.real_positive_scalar(scale, 'scale', TypeError)
        check.real_positive_scalar(scale_bright, 'scale_bright', TypeError)
        if scale_bright < scale:
            raise ValueError('scale_bright must be >= scale.')
        check.real_nonnegative_scalar(manual, 'manual', TypeError)

        # Unpack values from excam configuration
        darke, cic, rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em, \
            Nmin, Nmax, tmin, tmax, gmax, gconst, n, Nem, tol, \
            delta_constr, overhead, \
            _, _ = _unpack_excam_config(self.excam_config)

        # Fixed parameters (max val = min val) must be handled separately as
        # finite derivatives in the optimizer will throw up on them
        fixedg = False
        fixedt = False
        fixedn = False
        if Nmin == Nmax:
            fixedn = True
        if tmin == tmax:
            fixedt = True
        if gconst is not None:
            fixedg = True

        if (fixedt and fixedn) or (fixedn and fixedg) or (fixedg and fixedt):
            raise EXCAMOptimizeException('Unsupported optimization for two ' +
                                         'or more fixed variables')

        # Calculate flux rate in photoelectrons per second
        _, flux_rate_peak_pix = self.calc_flux_rate(sequence_name, manual)

        # Calculate maximum expected flux rate per pixel. Include optional
        # scale factors here
        fluxe = flux_rate_peak_pix * scale
        fluxe_bright = flux_rate_peak_pix * scale_bright

        # Calculate optimum gain, exposure time, and number of frames
        if snr is not None:
            if fixedn:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_fixed_N(snr, fluxe, fluxe_bright, darke, cic,
                                  rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                  Nmin, tmin, tmax, gmax, overhead,
                                  n=n, Nem=Nem,
                                  tol=tol, delta_constr=delta_constr)
                pass
            elif fixedt:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_fixed_time(snr, fluxe, fluxe_bright, darke, cic,
                                     rn, X, a, Lij, alpha0, fwc, alpha1,
                                     fwc_em, Nmin, Nmax, tmin, gmax,
                                     overhead, n=n, Nem=Nem, tol=tol,
                                     delta_constr=delta_constr)
            elif fixedg:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_fixed_g(snr, fluxe, fluxe_bright, darke, cic,
                                  rn, X, a, Lij, alpha0, fwc, alpha1,
                                  fwc_em, Nmin, Nmax, tmin, tmax, gconst,
                                  overhead, n=n, Nem=Nem, tol=tol,
                                  delta_constr=delta_constr)
            else:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_exptime(snr, fluxe, fluxe_bright, darke, cic,
                                  rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                  Nmin, Nmax, tmin, tmax, gmax,
                                  overhead=overhead, n=n, Nem=Nem,
                                  tol=tol, delta_constr=delta_constr)
            pass
        else:
            # use snr = 0 because it has to be something to get through the
            # input check (though it's not used)
            if fixedn:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_fixed_N(0, fluxe, fluxe_bright, darke, cic,
                                  rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                  Nmin, tmin, tmax, gmax, overhead,
                                  n=n, Nem=Nem,
                                  tol=tol, delta_constr=delta_constr,
                                  opt_choice=1)
                pass
            elif fixedt:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_fixed_time(0, fluxe, fluxe_bright, darke, cic,
                                     rn, X, a, Lij, alpha0, fwc, alpha1,
                                     fwc_em, Nmin, Nmax, tmin, gmax,
                                     overhead, n=n, Nem=Nem, tol=tol,
                                     delta_constr=delta_constr, opt_choice=1)
            elif fixedg:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_fixed_g(0, fluxe, fluxe_bright, darke, cic,
                                  rn, X, a, Lij, alpha0, fwc, alpha1,
                                  fwc_em, Nmin, Nmax, tmin, tmax, gconst,
                                  overhead, n=n, Nem=Nem, tol=tol,
                                  delta_constr=delta_constr, opt_choice=1)
            else:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_exptime(0, fluxe, fluxe_bright, darke, cic,
                                  rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                  Nmin, Nmax, tmin, tmax, gmax, overhead,
                                  n=n, Nem=Nem, tol=tol,
                                  delta_constr=delta_constr, opt_choice=1)
            pass

        log.warning('Optimization scheme used:  optflag = %s', optflag)
        log.warning('target SNR per pixel:  %s', snr)
        log.warning('SNR per pixel output from optimization:  %s', snr_actual)
        return n_frames, exp_time, gain, snr_actual, optflag

    def calc_exp_time_resel(self, sequence_name, snr, fraction=None,
                            num_pixels=None, scale=1., scale_bright=1.,
                            manual=1.0):
        """
        Estimates required exposure time for CGI calibration target based on
        precomputed flux rate grid and specified OTA + CGI throughput curves.
        Inputs are based on the user's desired region (e.g., PSF or a subregion
        of a PSF).  We refer to the SNR for the user's region as the SNR per
        spatial resolution element, or SNR per "resel".

        See snr_resel.pdf for details on the methodology used.

        Parameters
        ----------
        sequence_name : str
            Name of optical settings observing sequence.

        snr : float
            Desired SNR per resel to expose to, or None.

            If given a value, the optimization will attempt to get the shortest
            total integration time (number of frames*exposure time per frame)
            which achieves that SNR (opt=0).  If there is no feasible
            combination of gain, exposure time, and number of frames which can
            provide that SNR, or if None is given as an input, the optimization
            will attempt to get the best SNR possible given remaining
            constraints (opt=1), with no constraint on total integration time.
            See below for information on the output 'optflag'.

        fraction : float
            Fraction of total image area flux corresponding to resel.  >=0 and
            <=1.  If None, fraction is drawn from its calculated value stored
            in the sequences YAML file.  If the sequence is not intended for
            obtaining a point-spread function (PSF) of a target, there is no
            stored value for the sequence, and an exception will be raised.
            Defaults to None.

        num_pixels : float
            Number of pixels corresponding to resel.  >= 0. The inputs
            fraction and num_pixels must be accurate.  If they are not, they
            could cause the brightest pixel's flux (used for
            saturation constraints) to be smaller than the flux per pixel in
            the resel, and the function will raise an exception.  If None,
            num_pixels is drawn from its calculated value stored in the
            sequences YAML file.  If the sequence is not intended for obtaining
            a point-spread function (PSF) of a target, there is no stored value
            for the sequence, and an exception will be raised.
            Defaults to None.

        scale : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux of the resel, to
            allow the calculator to be used for occulted calculations. Defaults
            to 1.0 (no scaling).  For observing an unocculted point object, 1.0
            is the appropriate choice.

        scale_bright : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at the brightest pixel in
            a frame, so the camera settings do not overexpose and saturate this
            point.  Used primarily for occulted calculations.  Defaults to 1.0
            (no scaling).  For observing an unocculted point object, 1.0 is the
            appropriate choice.

        manual : float
            Manual fractional scale factor on flux rate, relative to the rate
            defined by mag at class instantiation.  Positive values increase
            flux (e.g. manual=10 will give 10x the flux of manual=1). Defaults
            to 1.0 (no scaling).

        Returns
        -------
        n_frames : float
            Total number of exposures to reach snr.

        exp_time : float
            Exposure time for each frame in n_frames to reach snr (s).

        gain : float
            Optimal gain setting for detector.

        snr_actual : float
            Expected SNR per resel when using above exposure settings.  This
            may be higher than the target SNR input from the user ('snr')
            because the optimization assumes a floating-point number for the
            number of frames (N) and then rounds up to the nearest integer
            value, which would increase the SNR value if there was an increase
            in N in the rounding.

        optflag : int
            0 or 1: 0 if the first optimization succeeded, 1 if the first
            failed but the second succeeded.

        """
        # Check inputs
        check.string(sequence_name, 'sequence_name', TypeError)
        if snr is not None:
            check.real_positive_scalar(snr, 'snr', TypeError)
        if fraction is None:
            fraction = self.sequences[sequence_name]['fraction']
            if fraction is None:
                raise ValueError('This sequence is not intended for obtaining '
                                 'a PSF and has no value for fraction.')
        check.real_positive_scalar(fraction, 'fraction', TypeError)
        if fraction > 1:
            raise ValueError('fraction must be <= 1.')
        if num_pixels is None:
            num_pixels = self.sequences[sequence_name]['num_pixels']
            if num_pixels is None:
                raise ValueError('This sequence is not intended for obtaining '
                                 'a PSF and has no value for num_pixels.')
        check.real_nonnegative_scalar(num_pixels, 'num_pixels', TypeError)
        check.real_positive_scalar(scale, 'scale', TypeError)
        check.real_positive_scalar(scale_bright, 'scale_bright', TypeError)
        check.real_nonnegative_scalar(manual, 'manual', TypeError)

        # Calculate flux rate in photoelectrons per second
        total_flux, flux_rate_peak_pix = self.calc_flux_rate(sequence_name,
                                                                    manual)

        # flux in the resel
        flux_resel = total_flux * fraction
        # now apply optional 'scale' to resel flux
        flux_resel_scale = flux_resel * scale
        # average flux per resel pixel
        fluxe = flux_resel_scale/num_pixels

        # apply optional 'scale_bright' to brightest pixel (for
        # saturation constraints)
        fluxe_bright = flux_rate_peak_pix * scale_bright

        if fluxe_bright < fluxe:
            raise ValueError('The combination of fraction, num_pixels, ' +
                'scale, and scale_bright is ' +
                'not accurate; the brightest pixel\'s flux was lower than ' +
                'flux per pixel in the resel')

        # Unpack values from excam configuration
        darke, cic, rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em, \
            Nmin, Nmax, tmin, tmax, gmax, gconst, n, Nem, tol, \
            delta_constr, overhead, _, _ \
            = _unpack_excam_config(self.excam_config)

        # Fixed parameters (max val = min val) must be handled separately as
        # finite derivatives in the optimizer will throw up on them
        fixedg = False
        fixedt = False
        fixedn = False
        if Nmin == Nmax:
            fixedn = True
        if tmin == tmax:
            fixedt = True
        if gconst is not None:
            fixedg = True

        if (fixedt and fixedn) or (fixedn and fixedg) or (fixedg and fixedt):
            raise EXCAMOptimizeException('Unsupported optimization for two ' +
                                         'or more fixed variables')

        # Calculate optimum gain, exposure time, and number of frames
        if snr is not None:
            if fixedn:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_fixed_N(snr, fluxe, fluxe_bright, darke, cic,
                                  rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                  Nmin, tmin, tmax, gmax, overhead,
                                  n=n, Nem=Nem,
                                  tol=tol, delta_constr=delta_constr,
                                  num_pixels=num_pixels)
                pass
            elif fixedt:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_fixed_time(snr, fluxe, fluxe_bright, darke, cic,
                                     rn, X, a, Lij, alpha0, fwc, alpha1,
                                     fwc_em, Nmin, Nmax, tmin, gmax,
                                     overhead, n=n, Nem=Nem,
                                     tol=tol, delta_constr=delta_constr,
                                     num_pixels=num_pixels)
            elif fixedg:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_fixed_g(snr, fluxe, fluxe_bright, darke, cic,
                                  rn, X, a, Lij, alpha0, fwc, alpha1,
                                  fwc_em, Nmin, Nmax, tmin, tmax, gconst,
                                  overhead, n=n, Nem=Nem, tol=tol,
                                  delta_constr=delta_constr,
                                  num_pixels=num_pixels)
            else:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_exptime(snr, fluxe, fluxe_bright, darke, cic,
                                  rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                  Nmin, Nmax, tmin, tmax, gmax,
                                  overhead, n=n, Nem=Nem,
                                  tol=tol, delta_constr=delta_constr,
                                  num_pixels=num_pixels)
            pass
        else:
            # use snr = 0 because it has to be something to get through the
            # input check (though it's not used)
            if fixedn:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_fixed_N(0, fluxe, fluxe_bright, darke, cic,
                                  rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                  Nmin, tmin, tmax, gmax, overhead,
                                  n=n, Nem=Nem,
                                  tol=tol, delta_constr=delta_constr,
                                  opt_choice=1, num_pixels=num_pixels)
                pass
            elif fixedt:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_fixed_time(0, fluxe, fluxe_bright, darke, cic,
                                     rn, X, a, Lij, alpha0, fwc, alpha1,
                                     fwc_em, Nmin, Nmax, tmin, gmax,
                                     overhead, n=n, Nem=Nem, tol=tol,
                                     delta_constr=delta_constr,
                                     opt_choice=1, num_pixels=num_pixels)
            elif fixedg:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_fixed_g(0, fluxe, fluxe_bright, darke, cic,
                                  rn, X, a, Lij, alpha0, fwc, alpha1,
                                  fwc_em, Nmin, Nmax, tmin, tmax, gconst,
                                  overhead, n=n, Nem=Nem, tol=tol,
                                  delta_constr=delta_constr,
                                  opt_choice=1, num_pixels=num_pixels)
            else:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_gain_exptime(0, fluxe, fluxe_bright, darke, cic,
                                  rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                  Nmin, Nmax, tmin, tmax, gmax,
                                  overhead, n=n, Nem=Nem, tol=tol,
                                  delta_constr=delta_constr, opt_choice=1,
                                  num_pixels=num_pixels)
            pass

        log.warning('Optimization scheme used:  optflag = %s', optflag)
        log.warning('target SNR per resel:  %s', snr)
        log.warning('SNR per resel output from optimization:  %s', snr_actual)
        return n_frames, exp_time, gain, snr_actual, optflag


    def calc_pc_exp_time(self, sequence_name, snr, scale=1.,
                         scale_bright=1., manual=1.0):
        """
        Estimates required exposure time for CGI calibration target based on
        precomputed flux rate grid and specified OTA + CGI throughput curves,
        for use with photon-counting.

        Parameters
        ----------
        sequence_name : str
            Name of optical settings observing sequence.

        snr : float
            Desired SNR per pixel to expose to, or None.

            If given a value, the optimization will attempt to get the shortest
            total integration time (number of frames*exposure time per frame)
            which achieves that SNR (opt=0).  If there is no feasible
            combination of gain, exposure time, and number of frames which can
            provide that SNR, or if None is given as an input, the optimization
            will attempt to get the best SNR possible given remaining
            constraints (opt=1), with no constraint on total integration time.
            See below for information on the output 'optflag'.

        scale : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at a point of interest, to
            allow the calculator to be used for occulted calculations. Defaults
            to 1.0 (no scaling).  For observing an unocculted point object, 1.0
            is the appropriate choice.

        scale_bright : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at the brightest point in
            a frame, so the camera settings do not overexpose and saturate this
            point.  Used primarily for occulted calculations.  Defaults to 1.0
            (no scaling).  For observing an unocculted point object, 1.0 is the
            appropriate choice.  scale_bright >= scale so that the flux from
            the brightest pixel is greater than or equal to the flux from a
            typical pixel.

        manual : float
            Manual fractional scale factor on flux rate, relative to the rate
            defined by mag at class instantiation.  Positive values increase
            flux (e.g. manual=10 will give 10x the flux of manual=1). Defaults
            to 1.0 (no scaling).

        Returns
        -------
        n_frames : float
            Total number of exposures to reach snr.

        exp_time : float
            Exposure time for each frame in n_frames to reach snr (s).

        gain : float
            Optimal gain setting for detector.

        snr_actual : float
            Expected SNR per pixel when using above exposure settings.  This
            may be higher than the target SNR input from the user ('snr')
            because the optimization assumes a floating-point number for the
            number of frames (N) and then rounds up to the nearest integer
            value, which would increase the SNR value if there was an increase
            in N in the rounding.

        optflag : int
            0 or 1: 0 if the first optimization succeeded, 1 if the first
            failed but the second succeeded.

        """
        # Check inputs
        check.string(sequence_name, 'sequence_name', TypeError)
        if snr is not None:
            check.real_positive_scalar(snr, 'snr', TypeError)
        check.real_positive_scalar(scale, 'scale', TypeError)
        check.real_positive_scalar(scale_bright, 'scale_bright', TypeError)
        if scale_bright < scale:
            raise ValueError('scale_bright must be >= scale')
        check.real_nonnegative_scalar(manual, 'manual', TypeError)

        # Unpack values from excam configuration
        darke, cic, rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em, \
            Nmin, Nmax, tmin, tmax, gmax, gconst, n, Nem, tol, delta_constr, \
            overhead, pc_ecount_max, T_factor \
            = _unpack_excam_config(self.excam_config)

        # Fixed parameters (max val = min val) must be handled separately as
        # finite derivatives in the optimizer will throw up on them
        fixedg = False
        fixedt = False
        fixedn = False
        if Nmin == Nmax:
            fixedn = True
        if tmin == tmax:
            fixedt = True
        if gconst is not None:
            fixedg = True

        if fixedt:
            raise EXCAMOptimizeException('Unsupported optimization for ' +
                                         'fixed exposure time')
        if fixedg:
            raise EXCAMOptimizeException('Unsupported optimization for ' +
                                         'fixed gain')
        if (fixedt and fixedn) or (fixedn and fixedg) or (fixedg and fixedt):
            raise EXCAMOptimizeException('Unsupported optimization for two ' +
                                         'or more fixed variables')

        # Calculate flux rate in photoelectrons per second
        _, flux_rate_peak_pix = self.calc_flux_rate(sequence_name, manual)

        # Calculate maximum expected flux rate per pixel. Include optional
        # scale factors here
        fluxe = flux_rate_peak_pix * scale
        fluxe_bright = flux_rate_peak_pix * scale_bright

        # Calculate optimum gain, exposure time, and number of frames
        if snr is not None:
            if fixedn:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_pc_fixed_N(snr, fluxe, fluxe_bright, darke, cic,
                                rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                Nmin, tmin, tmax, gmax,
                                overhead, n=n, Nem=Nem, tol=tol,
                                delta_constr=delta_constr,
                                pc_ecount_max=pc_ecount_max, T_factor=T_factor)
                pass
            else:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_pc(snr, fluxe, fluxe_bright, darke, cic, rn, X, a,
                        Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax,
                        tmin, tmax, gmax, overhead, n=n, Nem=Nem, tol=tol,
                        delta_constr=delta_constr,
                        pc_ecount_max=pc_ecount_max, T_factor=T_factor)
                pass
            pass
        else:
            # use snr = 0 because it has to be something to get through the
            # input check (though it's not used)
            if fixedn:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_pc_fixed_N(0, fluxe, fluxe_bright, darke, cic, rn, X, a,
                        Lij, alpha0, fwc, alpha1, fwc_em, Nmin,
                        tmin, tmax, gmax, overhead, n=n, Nem=Nem, tol=tol,
                        delta_constr=delta_constr,
                        pc_ecount_max=pc_ecount_max, T_factor=T_factor,
                        opt_choice=1)
                pass
            else:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_pc(0, fluxe, fluxe_bright, darke, cic, rn, X, a,
                        Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax,
                        tmin, tmax, gmax, overhead, n=n, Nem=Nem, tol=tol,
                        delta_constr=delta_constr,
                        pc_ecount_max=pc_ecount_max, T_factor=T_factor,
                        opt_choice=1)
                pass
            pass

        log.warning('Optimization scheme used:  optflag = %s', optflag)
        log.warning('target SNR per pixel:  %s', snr)
        log.warning('SNR per pixel output from optimization:  %s', snr_actual)
        return n_frames, exp_time, gain, snr_actual, optflag


    def calc_pc_exp_time_resel(self, sequence_name, snr, fraction=None,
                               num_pixels=None, scale=1., scale_bright=1.,
                               manual=1.0):
        """
        Estimates required exposure time for CGI calibration target based on
        precomputed flux rate grid and specified OTA + CGI throughput curves,
        for use with photon counting.
        Inputs are based on the user's desired region (e.g., PSF or a subregion
        of a PSF).  We refer to the SNR for the user's region as the SNR per
        spatial resolution element, or SNR per "resel".

        See snr_resel.pdf for details on the methodology used.

        Parameters
        ----------
        sequence_name : str
            Name of optical settings observing sequence.

        snr : float
            Desired SNR per resel to expose to, or None.

            If given a value, the optimization will attempt to get the shortest
            total integration time (number of frames*exposure time per frame)
            which achieves that SNR (opt=0).  If there is no feasible
            combination of gain, exposure time, and number of frames which can
            provide that SNR, or if None is given as an input, the optimization
            will attempt to get the best SNR possible given remaining
            constraints (opt=1), with no constraint on total integration time.
            See below for information on the output 'optflag'.

        fraction : float
            Fraction of total image area flux corresponding to resel.  >=0 and
            <=1.  If None, fraction is drawn from its calculated value stored
            in the sequences YAML file.  If the sequence is not intended for
            obtaining a point-spread function (PSF) of a target, there is no
            stored value for the sequence, and an exception will be raised.
            Defaults to None.

        num_pixels : float
            Number of pixels corresponding to resel.  >= 0. The inputs
            fraction and num_pixels must be accurate.  If they are not, they
            could cause the brightest pixel's flux (used for
            saturation constraints) to be smaller than the flux per pixel in
            the resel, and the function will raise an exception.  If None,
            num_pixels is drawn from its calculated value stored in the
            sequences YAML file.  If the sequence is not intended for obtaining
            a point-spread function (PSF) of a target, there is no stored value
            for the sequence, and an exception will be raised.
            Defaults to None.

        scale : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux of the resel, to
            allow the calculator to be used for occulted calculations. Defaults
            to 1.0 (no scaling).  For observing an unocculted point object, 1.0
            is the appropriate choice.

        scale_bright : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at the brightest pixel in
            a frame, so the camera settings do not overexpose and saturate this
            point.  Used primarily for occulted calculations.  Defaults to 1.0
            (no scaling).  For observing an unocculted point object, 1.0 is the
            appropriate choice.

        manual : float
            Manual fractional scale factor on flux rate, relative to the rate
            defined by mag at class instantiation.  Positive values increase
            flux (e.g. manual=10 will give 10x the flux of manual=1). Defaults
            to 1.0 (no scaling).

        Returns
        -------
        n_frames : float
            Total number of exposures to reach snr.

        exp_time : float
            Exposure time for each frame in n_frames to reach snr (s).

        gain : float
            Optimal gain setting for detector.

        snr_actual : float
            Expected SNR per resel when using above exposure settings.  This
            may be higher than the target SNR input from the user ('snr')
            because the optimization assumes a floating-point number for the
            number of frames (N) and then rounds up to the nearest integer
            value, which would increase the SNR value if there was an increase
            in N in the rounding.

        optflag : int
            0 or 1: 0 if the first optimization succeeded, 1 if the first
            failed but the second succeeded.

        """
        # Check inputs
        check.string(sequence_name, 'sequence_name', TypeError)
        if snr is not None:
            check.real_positive_scalar(snr, 'snr', TypeError)
        if fraction is None:
            fraction = self.sequences[sequence_name]['fraction']
            if fraction is None:
                raise ValueError('This sequence is not intended for obtaining '
                                 'a PSF and has no value for fraction.')
        check.real_positive_scalar(fraction, 'fraction', TypeError)
        if fraction > 1:
            raise ValueError('fraction must be <= 1.')
        if num_pixels is None:
            num_pixels = self.sequences[sequence_name]['num_pixels']
            if num_pixels is None:
                raise ValueError('This sequence is not intended for obtaining '
                                 'a PSF and has no value for num_pixels.')
        check.real_nonnegative_scalar(num_pixels, 'num_pixels', TypeError)
        check.real_positive_scalar(scale, 'scale', TypeError)
        check.real_positive_scalar(scale_bright, 'scale_bright', TypeError)
        check.real_nonnegative_scalar(manual, 'manual', TypeError)

        # Calculate flux rate in photoelectrons per second
        total_flux, flux_rate_peak_pix = self.calc_flux_rate(sequence_name,
                                                                    manual)

        # flux in the resel
        flux_resel = total_flux * fraction
        # now apply optional 'scale' to resel flux
        flux_resel_scale = flux_resel * scale
        # average flux per resel pixel
        fluxe = flux_resel_scale/num_pixels

        # apply optional 'scale_bright' to brightest pixel (for
        # saturation constraints)
        fluxe_bright = flux_rate_peak_pix * scale_bright

        if fluxe_bright < fluxe:
            raise ValueError('The combination of fraction, num_pixels, ' +
                'scale, and scale_bright is ' +
                'not accurate; the brightest pixel\'s flux was lower than ' +
                'flux per pixel in the resel')

        # Unpack values from excam configuration
        darke, cic, rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em, \
            Nmin, Nmax, tmin, tmax, gmax, gconst, n, Nem, tol, delta_constr, \
            overhead, pc_ecount_max, T_factor = \
            _unpack_excam_config(self.excam_config)

        # Fixed parameters (max val = min val) must be handled separately as
        # finite derivatives in the optimizer will throw up on them
        fixedg = False
        fixedt = False
        fixedn = False
        if Nmin == Nmax:
            fixedn = True
        if tmin == tmax:
            fixedt = True
        if gconst is not None:
            fixedg = True

        if fixedt:
            raise EXCAMOptimizeException('Unsupported optimization for ' +
                                         'fixed exposure time')
        if fixedg:
            raise EXCAMOptimizeException('Unsupported optimization for ' +
                                         'fixed gain')
        if (fixedt and fixedn) or (fixedn and fixedg) or (fixedg and fixedt):
            raise EXCAMOptimizeException('Unsupported optimization for two ' +
                                         'or more fixed variables')

        # Calculate optimum gain, exposure time, and number of frames
        if snr is not None:
            if fixedn:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_pc_fixed_N(snr, fluxe, fluxe_bright, darke, cic,
                                rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                Nmin, tmin, tmax, gmax, overhead,
                                n=n, Nem=Nem, tol=tol,
                                delta_constr=delta_constr,
                                pc_ecount_max=pc_ecount_max,
                                T_factor=T_factor, num_pixels=num_pixels)
                pass
            else:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_pc(snr, fluxe, fluxe_bright, darke, cic, rn, X, a,
                        Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax,
                        tmin, tmax, gmax, overhead, n=n, Nem=Nem, tol=tol,
                        delta_constr=delta_constr, pc_ecount_max=pc_ecount_max,
                        T_factor=T_factor, num_pixels=num_pixels)
                pass
            pass
        else:
            # use snr = 0 because it has to be something to get through the
            # input check (though it's not used)
            if fixedn:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_pc_fixed_N(0, fluxe, fluxe_bright, darke, cic, rn, X, a,
                        Lij, alpha0, fwc, alpha1, fwc_em, Nmin,
                        tmin, tmax, gmax, overhead, n=n, Nem=Nem, tol=tol,
                        delta_constr=delta_constr, pc_ecount_max=pc_ecount_max,
                        T_factor=T_factor, opt_choice=1, num_pixels=num_pixels)
                pass
            else:
                gain, exp_time, n_frames, snr_actual, optflag = \
                calc_pc(0, fluxe, fluxe_bright, darke, cic, rn, X, a,
                        Lij, alpha0, fwc, alpha1, fwc_em, Nmin, Nmax,
                        tmin, tmax, gmax, overhead, n=n, Nem=Nem, tol=tol,
                        delta_constr=delta_constr, pc_ecount_max=pc_ecount_max,
                        T_factor=T_factor, opt_choice=1, num_pixels=num_pixels)
                pass
            pass

        log.warning('Optimization scheme used:  optflag = %s', optflag)
        log.warning('target SNR per resel:  %s', snr)
        log.warning('SNR per resel output from optimization:  %s', snr_actual)
        return n_frames, exp_time, gain, snr_actual, optflag


    def calc_pc_const_int_time(self, sequence_name, t_tot, scale=1.,
                            scale_bright=1., manual=1.0, hard_limit=True):
        """
        For a given total integration time, this function estimates
        required exposure time for CGI calibration target based
        on precomputed flux rate grid and specified OTA + CGI throughput
        curves, for use with photon counting.

        There is no input for target SNR for this function since the usual
        first optimization of minimizing the total integration time is
        irrelevant since that time is fixed.  This function simply tries to
        maximize the SNR.

        Parameters
        ----------
        sequence_name : str
            Name of optical settings observing sequence.

        t_tot : float
            The fixed total integration time in s
            (i.e., the number of frames times the exposure time per frame).
            >= 0.

        scale : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at a point of interest, to
            allow the calculator to be used for occulted calculations. Defaults
            to 1.0 (no scaling).  For observing an unocculted point object, 1.0
            is the appropriate choice.

        scale_bright : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at the brightest point in
            a frame, so the camera settings do not overexpose and saturate this
            point.  Used primarily for occulted calculations.  Defaults to 1.0
            (no scaling).  For observing an unocculted point object, 1.0 is the
            appropriate choice.  scale_bright >= scale so that the flux from
            the brightest pixel is greater than or equal to the flux from a
            typical pixel.

        manual : float
            Manual fractional scale factor on flux rate, relative to the rate
            defined by mag at class instantiation.  Positive values increase
            flux (e.g. manual=10 will give 10x the flux of manual=1). Defaults
            to 1.0 (no scaling).

        hard_limit: boolean
            True if a hard limit needed for t_tot.  The number of frames (N)
            has to be treated as a float in the optimization calculation, and
            the frame time (t) is optimized under that assumption, but the
            optimized output N will be rounded up to the next integer.
            If the user wants the optimized N*t to equal the input t_tot
            exactly, hard_limit should be True, and the output t is equal to
            t_tot/N.  In this case, the optimized SNR output may be a bit
            smaller than what it could have been with the optimized float
            value of N.  If the user is okay with a little variation
            between N*t and t_tot, then hard_limit should be False, and then
            the optimized SNR will be as big as possible since N is rounded up
            and t is the value from the optimization, and N*t may be slightly
            bigger than t_tot.  Defaults to True.

        Returns
        -------
        n_frames : float
            Total number of exposures to maximize SNR (snr_actual).

        exp_time : float
            Exposure time for each frame in n_frames to reach snr_actual (s).

        gain : float
            Optimal gain setting for detector.

        snr_actual : float
            Expected SNR per pixel when using above exposure settings.

        t_tot_out : float
            Actual total integration time used, exp_time*n_frames.  If
            hard_limit is True, this should equal the input t_tot.
        """
        # Check inputs
        check.string(sequence_name, 'sequence_name', TypeError)
        check.real_nonnegative_scalar(t_tot, 't_tot', TypeError)
        check.real_positive_scalar(scale, 'scale', TypeError)
        check.real_positive_scalar(scale_bright, 'scale_bright', TypeError)
        if scale_bright < scale:
            raise ValueError('scale_bright must be >= scale.')
        check.real_nonnegative_scalar(manual, 'manual', TypeError)
        if not isinstance(hard_limit, bool):
            raise TypeError('hard_limit must be boolean.')

        # Unpack values from excam configuration
        darke, cic, rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em, \
            Nmin, Nmax, tmin, tmax, gmax, gconst, n, Nem, tol, delta_constr, \
            overhead, pc_ecount_max, T_factor = \
            _unpack_excam_config(self.excam_config)

        # Fixed parameters (max val = min val) must be handled separately as
        # finite derivatives in the optimizer will throw up on them
        fixedg = False
        fixedt = False
        fixedn = False
        if Nmin == Nmax:
            fixedn = True
        if tmin == tmax:
            fixedt = True
        if gconst is not None:
            fixedg = True

        if fixedn or fixedt or fixedg:
            raise EXCAMOptimizeException('Unsupported optimization for a ' +
                                         'fixed variable')

        # Calculate flux rate in photoelectrons per second
        _, flux_rate_peak_pix = self.calc_flux_rate(sequence_name, manual)

        # Calculate maximum expected flux rate per pixel. Include optional
        # scale factors here
        fluxe = flux_rate_peak_pix * scale
        fluxe_bright = flux_rate_peak_pix * scale_bright

        # Calculate optimum gain, exposure time, and number of frames
        gain, exp_time, n_frames, snr_actual, t_tot_out = \
            calc_pc_gain_fixed_Ntime(t_tot, fluxe, fluxe_bright, darke, cic,
                                rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                Nmin, Nmax, tmin, tmax, gmax, overhead,
                                n=n, Nem=Nem,
                                tol=tol, delta_constr=delta_constr,
                                pc_ecount_max=pc_ecount_max, T_factor=T_factor,
                                hard_limit=hard_limit)

        log.warning('Output total integration time:  %s', t_tot_out)
        log.warning('SNR per pixel output from optimization:  %s', snr_actual)
        return n_frames, exp_time, gain, snr_actual, t_tot_out


    def calc_pc_const_int_time_resel(self, sequence_name, t_tot, fraction=None,
                                num_pixels=None, scale=1., scale_bright=1.,
                                manual=1.0, hard_limit=True):
        """
        For a given total integration time, this function estimates
        required exposure time for CGI calibration target based
        on precomputed flux rate grid and specified OTA + CGI throughput
        curves, for use with photon counting.
        Inputs are based on the user's desired region
        (e.g., PSF or a subregion of a PSF).  We refer to the SNR for the
        user's region as the SNR per spatial resolution element, or SNR per
        "resel".

        See snr_resel.pdf for details on the methodology used.

        There is no input for target SNR for this function since the usual
        first optimization of minimizing the total integration time is
        irrelevant since that time is fixed.  This function simply tries to
        maximize the SNR.

        Parameters
        ----------
        sequence_name : str
            Name of optical settings observing sequence.

        t_tot : float
            The fixed total integration time (i.e., the number of frames times
            the exposure time per frame).  >= 0.

        fraction : float
            Fraction of total image area flux corresponding to resel.  >=0 and
            <=1.  If None, fraction is drawn from its calculated value stored
            in the sequences YAML file.  If the sequence is not intended for
            obtaining a point-spread function (PSF) of a target, there is no
            stored value for the sequence, and an exception will be raised.
            Defaults to None.

        num_pixels : float
            Number of pixels corresponding to resel.  >= 0. The inputs
            fraction and num_pixels must be accurate.  If they are not, they
            could cause the brightest pixel's flux (used for
            saturation constraints) to be smaller than the flux per pixel in
            the resel, and the function will raise an exception.  If None,
            num_pixels is drawn from its calculated value stored in the
            sequences YAML file.  If the sequence is not intended for obtaining
            a point-spread function (PSF) of a target, there is no stored value
            for the sequence, and an exception will be raised.
            Defaults to None.

        scale : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux of the resel, to
            allow the calculator to be used for occulted calculations. Defaults
            to 1.0 (no scaling).  For observing an unocculted point object, 1.0
            is the appropriate choice.

        scale_bright : float
            Flux scale factor for optional attenuation (achromatic, fixed value
            scalar).  This is used to scale the flux at the brightest pixel in
            a frame, so the camera settings do not overexpose and saturate this
            point.  Used primarily for occulted calculations.  Defaults to 1.0
            (no scaling).  For observing an unocculted point object, 1.0 is the
            appropriate choice.

        manual : float
            Manual fractional scale factor on flux rate, relative to the rate
            defined by mag at class instantiation.  Positive values increase
            flux (e.g. manual=10 will give 10x the flux of manual=1). Defaults
            to 1.0 (no scaling).

        hard_limit: boolean
            True if a hard limit needed for t_tot.  The number of frames (N)
            has to be treated as a float in the optimization calculation, and
            the frame time (t) is optimized under that assumption, but the
            optimized output N will be rounded up to the next integer.
            If the user wants the optimized N*t to equal the input t_tot
            exactly, hard_limit should be True, and the output t is equal to
            t_tot/N.  In this case, the optimized SNR output may be a bit
            smaller than what it could have been with the optimized float
            value of N.  If the user is okay with a little variation
            between N*t and t_tot, then hard_limit should be False, and then
            the optimized SNR will be as big as possible since N is rounded up
            and t is the value from the optimization, and N*t may be slightly
            bigger than t_tot.  Defaults to True.

        Returns
        -------
        n_frames : float
            Total number of exposures to maximize SNR (snr_actual).

        exp_time : float
            Exposure time for each frame in n_frames to reach snr_actual (s).

        gain : float
            Optimal gain setting for detector.

        snr_actual : float
            Expected SNR per resel when using above exposure settings.

        t_tot_out : float
            Actual total integration time used, exp_time*n_frames.  If
            hard_limit is True, this should equal the input t_tot.
        """
        # Check inputs
        check.string(sequence_name, 'sequence_name', TypeError)
        check.real_nonnegative_scalar(t_tot, 't_tot', TypeError)
        if fraction is None:
            fraction = self.sequences[sequence_name]['fraction']
            if fraction is None:
                raise ValueError('This sequence is not intended for obtaining '
                                 'a PSF and has no value for fraction.')
        check.real_positive_scalar(fraction, 'fraction', TypeError)
        if fraction > 1:
            raise ValueError('fraction must be <= 1.')
        if num_pixels is None:
            num_pixels = self.sequences[sequence_name]['num_pixels']
            if num_pixels is None:
                raise ValueError('This sequence is not intended for obtaining '
                                 'a PSF and has no value for num_pixels.')
        check.real_nonnegative_scalar(num_pixels, 'num_pixels', TypeError)
        check.real_positive_scalar(scale, 'scale', TypeError)
        check.real_positive_scalar(scale_bright, 'scale_bright', TypeError)
        if scale_bright < scale:
            raise ValueError('scale_bright must be >= scale.')
        check.real_nonnegative_scalar(manual, 'manual', TypeError)
        if not isinstance(hard_limit, bool):
            raise TypeError('hard_limit must be boolean.')

        # Calculate flux rate in photoelectrons per second
        total_flux, flux_rate_peak_pix = self.calc_flux_rate(sequence_name,
                                                                    manual)

        # flux in the resel
        flux_resel = total_flux * fraction
        # now apply optional 'scale' to resel flux
        flux_resel_scale = flux_resel * scale
        # average flux per resel pixel
        fluxe = flux_resel_scale/num_pixels

        # apply optional 'scale_bright' to brightest pixel (for
        # saturation constraints)
        fluxe_bright = flux_rate_peak_pix * scale_bright

        if fluxe_bright < fluxe:
            raise ValueError('The combination of fraction, num_pixels, ' +
                'scale, and scale_bright is ' +
                'not accurate; the brightest pixel\'s flux was lower than ' +
                'flux per pixel in the resel')

        # Unpack values from excam configuration
        darke, cic, rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em, \
            Nmin, Nmax, tmin, tmax, gmax, gconst, n, Nem, tol, delta_constr, \
            overhead, pc_ecount_max, T_factor = \
            _unpack_excam_config(self.excam_config)

        # Fixed parameters (max val = min val) must be handled separately as
        # finite derivatives in the optimizer will throw up on them
        fixedg = False
        fixedt = False
        fixedn = False
        if Nmin == Nmax:
            fixedn = True
        if tmin == tmax:
            fixedt = True
        if gconst is not None:
            fixedg = True

        if fixedn or fixedt or fixedg:
            raise EXCAMOptimizeException('Unsupported optimization for a ' +
                                         'fixed variable')

        # Calculate optimum gain, exposure time, and number of frames
        gain, exp_time, n_frames, snr_actual, t_tot_out = \
            calc_pc_gain_fixed_Ntime(t_tot, fluxe, fluxe_bright, darke, cic,
                                rn, X, a, Lij, alpha0, fwc, alpha1, fwc_em,
                                Nmin, Nmax, tmin, tmax, gmax, overhead,
                                n=n, Nem=Nem,
                                tol=tol, delta_constr=delta_constr,
                                pc_ecount_max=pc_ecount_max, T_factor=T_factor,
                                num_pixels=num_pixels, hard_limit=hard_limit)

        log.warning('Output total integration time:  %s', t_tot_out)
        log.warning('SNR per resel output from optimization:  %s', snr_actual)
        return n_frames, exp_time, gain, snr_actual, t_tot_out


    def calc_locam_gain(self, sequence_name, manual=1.0):
        """
        Estimates LOCAM gain for an optical configuration on the stellar
        target associated with this class.

        Parameters
        ----------
        sequence_name : str
            Name of optical settings observing sequence.

        manual : float
            Manual fractional scale factor on flux rate, relative to the rate
            defined by mag at class instantiation.  Positive values increase
            flux (e.g. manual=10 will give 10x the flux of manual=1). Defaults
            to 1.0 (no scaling).

        Returns
        -------

        gain : float
            Optimal gain setting for detector.

        code : int
            value from the following list:
              1: no feasible non-unity gain solution exists due to aging
                 considerations, but unity gain operation is feasible
              2: feasible non-unity gain found, constraint 2 set limit
              3: feasible non-unity gain found, constraint 3 set limit
              5: feasible non-unity gain found, constraint 5 set limit
              6: feasible non-unity gain found, constraint 6 set limit
            In the case where two or more constraints are met simultaneously,
            the lowest number will be output.  In the vanishingly-unlikely
            scenario where the only numerical value permitted by aging
            constraints is 1 (but it's still permitted--no violation), the
            appropriate value in [2, 3, 5, 6]  will be returned.  If
            unity-gain operation is infeasible this function will raise a
            LOCAMOptimizeException (and not return)

        """
        # Check inputs
        check.string(sequence_name, 'sequence_name', TypeError)
        check.real_nonnegative_scalar(manual, 'manual', TypeError)

        # Unpack values from excam configuration
        darke, cic, alpha0, fwc, alpha1, fwc_em, g_max_comm, g_max_age, \
            e_max_age, tframe, n = _unpack_locam_config(self.locam_config)

        # Calculate flux rate in photoelectrons per second
        _, flux_rate_peak_pix = self.calc_flux_rate(sequence_name, manual)

        # Calculate optimum gain, exposure time, and number of frames
        gain, code = calc_locam_gain(
            fluxe_bright=flux_rate_peak_pix,
            darke=darke,
            cic=cic,
            alpha0=alpha0,
            fwc=fwc,
            alpha1=alpha1,
            fwc_em=fwc_em,
            g_max_comm=g_max_comm,
            g_max_age=g_max_age,
            e_max_age=e_max_age,
            tframe=tframe,
            n=n,
        )

        return gain, code



if __name__ == "__main__":
    # use argparse so we can do functional tests without changing the code
    ap = argparse.ArgumentParser(
        prog='python cgi_eetc.py',
        description="Compute fluxes and EXCAM settings based on CGI " +
                    "configuration and stellar properties"
    )
    ap.add_argument('--snr', default=100, help="Target SNR.  Default = 100.",
                    type=float)
    ap.add_argument('--mag', default=10.25, type=float,
                    help="Stellar magnitude, default=10.25.")
    ap.add_argument('--spt', default='O5', type=str,
                    help="Stellar type, default=O5.")
    ap.add_argument('--seq', default='CGI_SEQ_NFOV_UNOCC_ASTROM_PHOTOM_PS_1B',
                    type=str,
                    help="CGI configuration name, defaults to " +
                         "CGI_SEQ_NFOV_UNOCC_ASTROM_PHOTOM_PS_1B " +
                         "(unocculted NFOV with Lyot in, HLC DM settings " +
                         "applied and imaging lens in DPAM)")
    ap.add_argument('--manual', default=1.0, type=float,
                    help="manual scale factor, default=1.0")
    ap.add_argument('--scale', default=1.0, type=float,
                    help="scale relative to peak unocculted pixel in region " +
                         "of interest, default=1.0")
    ap.add_argument('--scale_bright', type=float,
                    help="scale relative to peak unocculted pixel in " +
                         "brightest area of image, defaults to be same as" +
                         "scale")
    args = ap.parse_args()

    # Load from arguments
    _mag = args.mag
    _spt = args.spt
    _seq = args.seq
    _manual = args.manual
    _snr = args.snr
    _scale = args.scale
    if args.scale_bright is None:
        _scale_bright = args.scale
        pass
    else:
        _scale_bright = args.scale_bright

    print('Star is vmag = %g, type = %s' % (_mag, _spt))
    print('CGI config: PAM config = %s, manual scale = %g' %
          (_seq, _manual))
    print('Asking for SNR = %g' % (_snr,))

    cgieetc = CGIEETC(mag=_mag, phot='v', spt=_spt, pointer_path=POINTER_PATH)

    # Flux calc
    flux_all, flux_pix = cgieetc.calc_flux_rate(sequence_name=_seq,
                                                manual=_manual)
    print('Expected integrated flux = %g photoelectrons/sec' % (flux_all,))
    print('Expected flux at peak pixel = %g photoelectrons/sec' % (flux_pix,))
    print('Scale = %g, so flux/pixel in ROI = %g photoelectrons/sec' %
          (_scale, flux_pix*_scale))
    print('Bright scale = %g, ' % (_scale_bright,) + 'so flux/pixel in the ' +
          'brightest pixel in the frame = %g photoelectrons/sec'
          % (flux_pix*_scale_bright,))

    # EXCAM settings calc
    try:
        n_fr, exp, g_, snr_o, opt = cgieetc.calc_exp_time(
            sequence_name=_seq, snr=_snr, scale=_scale,
            scale_bright=_scale_bright, manual=_manual
        )
        print('Gain = %g' % g_)
        print('Exposure time per frame = %g' % exp)
        print('Number of frames = %d' % n_fr)
        print('Expected SNR per pixel = %g' % snr_o)
        print('Total wall-clock exposure time = %g' % (exp*n_fr,))
        print('Optimization result: %d' % opt)
    except EXCAMOptimizeException:
        print('No feasible solution found--too many electrons for full ' +
              'well even at minimum operating settings')
        pass

    mtime, mtime_status, etc_mtime, etc_mtime_status = \
        cgieetc.excam_saturation_time(sequence_name=_seq, g=10, f=0.9,
        scale_bright=_scale_bright, manual=_manual)
    print("Max time to the fractional full well:  ", mtime)
    print('Max time met condition:  ', mtime_status)
    print("ETC max time to the fractional full well:  ", etc_mtime)
    print('ETC max time met condition:  ', etc_mtime_status)

    # # what's in README:

    # # analog
    # cgi_eetc = CGIEETC(mag=5, phot='v', spt='g2v')
    # max_time, max_time_status, etc_max_time, etc_max_time_status = \
    #     cgi_eetc.excam_saturation_time(
    #         sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0',
    #         g=10, f=0.9, scale_bright=1.)
    # num_frames, exp_time_frame, gain, snr_out, optflag = \
    #     cgi_eetc.calc_exp_time(sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0',
    #                            snr=100)
    # num_frames, exp_time_frame, gain, snr_out, optflag = \
    #     cgi_eetc.calc_exp_time_resel(
    #         sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0',
    #         snr=100, fraction=1e-5, num_pixels=10)
    # num_frames, exp_time_frame, gain, snr_out, t_tot_out = \
    #     cgi_eetc.calc_const_int_time(
    #         sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0', t_tot=80)
    # num_frames, exp_time_frame, gain, snr_out, t_tot_out = \
    #     cgi_eetc.calc_const_int_time_resel(
    #         sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0',
    #         t_tot=80, fraction=1e-5, num_pixels=10)
    # gain, code = cgi_eetc.calc_locam_gain(sequence_name='LOCAM_NFOV_DM')

    # # photon counting
    # cgi_eetc_pc = CGIEETC(mag=19, phot='v', spt='g2v')
    # num_frames, exp_time_frame, gain, snr_out, optflag = \
    #     cgi_eetc_pc.calc_pc_exp_time(
    #         sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0', snr=1.5)
    # num_frames, exp_time_frame, gain, snr_out, optflag = \
    #     cgi_eetc_pc.calc_pc_exp_time_resel(
    #         sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0',
    #         snr=4, fraction=1e-5, num_pixels=10)
    # num_frames, exp_time_frame, gain, snr_out, t_tot_out = \
    #     cgi_eetc_pc.calc_pc_const_int_time(
    #         sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0', t_tot=80)
    # num_frames, exp_time_frame, gain, snr_out, t_tot_out = \
    #     cgi_eetc_pc.calc_pc_const_int_time_resel(
    #         sequence_name='CGI_SEQ_NFOV_ALIGN_LSAM_0',
    #         t_tot=80, fraction=1e-5, num_pixels=10)
    pass
