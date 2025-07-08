# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Miscellaneous tools which use the CGIEETC class
"""

import os

import eetc.util.check as check
from eetc.cgi_eetc import CGIEETC, POINTER_PATH

def get_signal_ratio(mag1, spt1, gain1, mag2, spt2, gain2, sequence_name,
                     pointer_path=POINTER_PATH):
    """
    Tool to generate a manual signal ratio for bootstrapping LOWFSC on the
    target star.

    1074400: Given 1) the magnitude and stellar type of a star with LOWFS
    reference frames taken on it (star 1), 2) the magnitude and stellar type of
    a star which must be acquired without a LOWFS reference frame (star 2), and
    3) the LOCAM gain to be used with each, the CTC GSW shall compute the
    expected ratio of signal (star 2/star 1) between the two.

    Note: does not generate the gain internally.  Use the calc_locam_gain()
    method of a CGIEETC instance to compute an optimized gain, if this is
    desired.  This takes in the gain as a input to handle the case where
    gains are set to non-optimized values for some reason (e.g. need to run
    with unity gain).

    Arguments:
     mag1: V-band magnitude of first star.  Corresponds to the reference star
      in the OS11-like use case.  Floating-point value.
     spt1: stellar type of first star.  Must be one of the values supported
      by flux grid.  Corresponds to the reference star
      in the OS11-like use case.  String.
     gain1: LOCAM EM gain to use with the first star.  Corresponds to the
      reference star in the OS11-like use case.  Floating-point value >= 1.
     mag2: V-band magnitude of second star.  Corresponds to the target star
      in the OS11-like use case.  Floating-point value.
     spt2: stellar type of second star.  Must be one of the values supported
      by flux grid.  Corresponds to the target star
      in the OS11-like use case.  String.
     gain2: LOCAM EM gain to use with the second star.  Corresponds to the
      target star in the OS11-like use case.  Floating-point value >= 1.
     sequence_name: Name of optical settings observing sequence.  String.

    Keyword Arguments:
     pointer_path: Full path of pointer yaml file.  String.  Pointer file
      contains the relative paths of the other files needed
      internally by CGIEETC. Paths are relative to the location of the
      pointer file. Defaults to the file delivered with this repository
      (pointer.yaml).

    Returns:
     single floating-point value with the star 2/star 1 ratio of signal,
      including effects of gain.

    """

    # Check inputs
    check.real_scalar(mag1, 'mag1', TypeError)
    check.string(spt1, 'spt1', TypeError)
    check.real_scalar(gain1, 'gain1', TypeError)
    if gain1 < 1:
        raise ValueError('gain1 must be a real scalar >= 1')

    check.real_scalar(mag2, 'mag2', TypeError)
    check.string(spt2, 'spt2', TypeError)
    check.real_scalar(gain2, 'gain2', TypeError)
    if gain2 < 1:
        raise ValueError('gain2 must be a real scalar >= 1')

    check.string(sequence_name, 'sequence_name', TypeError)
    # check sequence content below (see note)
    check.string(pointer_path, 'pointer_path', TypeError)
    if not os.path.exists(pointer_path):
        raise ValueError('pointer_path must point at an existing file')

    # use mag + type to make class object 1 and 2
    star1_eetc = CGIEETC(
        mag=mag1,
        phot='v',
        spt=spt1,
        pointer_path=pointer_path,
    )
    star2_eetc = CGIEETC(
        mag=mag2,
        phot='v',
        spt=spt2,
        pointer_path=pointer_path,
    )

    # Get flux rate for 1 and 2 from class methods
    # Both will use same sequence in an OS11-like scenario
    # Ideally would check sequence name at variable load, but would need to
    # duplicate functionality to do so.
    if sequence_name.upper() not in star1_eetc.sequences:
        raise ValueError('sequence_name not found in supplied sequences (1)')
    if sequence_name.upper() not in star2_eetc.sequences:
        raise ValueError('sequence_name not found in supplied sequences (2)')

    flux1, _ = star1_eetc.calc_flux_rate(sequence_name)
    flux2, _ = star2_eetc.calc_flux_rate(sequence_name)

    # signal ratio = target flux/reference flux as per FDD (used for target)
    sr = (flux2*gain2)/(flux1*gain1)

    return sr


def get_effective_wavelength(cfam, spt, pointer_path=POINTER_PATH):
    """
    Get the effective wavelength of a filter in the context of a stellar
    spectral type (integrated jointly on both)

    1050919 - Given 1) the transmission profile of a CFAM filter, and 2) a
    stellar spectral type, the CTC GSW shall compute the effective central
    wavelength of that filter when observing a star of that type.

    Note this uses Angstroms, not nm or microns; expect to convert as
    necessary.

    Arguments:
     cfam: list of strings denoting filter.  Each string must be a member of
      the cfams in the grid.
     spt: stellar type, must be one of the values supported by grid.  String.

    Keyword Arguments:
     pointer_path: Full path of pointer yaml file.  String.  Pointer file
      contains the paths of the other files needed
      internally by CGIEETC. Paths can be relative to the eetc module folder
      or absolute. Defaults to the file delivered with this repository
      (pointer.yaml).

    Returns:
     list of wavelengths in Angstroms, of the same length as cfam

    """

    # Check inputs
    if not isinstance(cfam, list):
        raise TypeError('cfam must be a list of filters')
    check.string(spt, 'spt', TypeError)
    check.string(pointer_path, 'pointer_path', TypeError)
    if not os.path.exists(pointer_path):
        raise ValueError('pointer_path must point at an existing file')

    # create class object with dummy magnitude
    tmp_eetc = CGIEETC(
        mag=0, # dummy value, magnitude doesn't matter and we won't use it
        phot='v',
        spt=spt,
        pointer_path=pointer_path,
    )
    for index, f in enumerate(cfam):
        check.string(f, 'cfam[' + str(index) + ']', TypeError)
        if f not in tmp_eetc.wave_grid_cfams:
            raise ValueError('cfam[' + str(index) +
                                '] not a filter supported by grid')

    # read out appropriate pre-computed value from wave grid
    out = []
    for f in cfam:
        out.append(tmp_eetc._load_wave_val(f))
    return out
