# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
constants used by load.py
"""

# Keys that must be in every sequence in the 'sequences' file pointed to by the
# pointer file.  All other keys will be stored for informational purposes but
# ignored for calculations (e.g. 'dms' for information about the DM setting
# used to get peak_flux_ratio_pix)
valid_sequence_keys = [
    'mode', 'spam_lsam', 'fpam', 'fsam', 'cfam', 'dpam', 'peak_flux_ratio_pix',
    'fraction', 'num_pixels'
]

# Top-level keys that must be present in the 'thpt_data' file pointed to by the
# pointer file.  No other keys should be present.
valid_thpt_data_keys = [
    'spam_lsam', 'fpam', 'fsam', 'dpam'
]

valid_excam_config_keys = [
    'darke', 'cic', 'rn', 'X', 'a', 'Lij', 'alpha0', 'fwc', 'alpha1',
    'fwc_em', 'Nmin', 'Nmax', 'tmin', 'tmax', 'gmax', 'gconst', 'n', 'Nem',
    'tol', 'delta_constr', 'overhead', 'pc_ecount_max', 'T_factor'
]

valid_locam_config_keys = [
    'darke', 'cic', 'alpha0', 'fwc', 'alpha1', 'fwc_em', 'g_max_comm',
    'g_max_age', 'e_max_age', 'tframe', 'n',
]

# Fits keys reference: https://archive.stsci.edu/fits/fits_standard/node39.html
header_primary_keys = [
    'BITPIX', 'EXTEND', 'NAXIS1', 'NAXIS2', 'NAXIS', 'SIMPLE'
]
header_ext_keys = [
    'BITPIX', 'NAXIS1', 'NAXIS2', 'NAXIS', 'GCOUNT', 'XTENSION', 'PCOUNT'
]
header_custom_keys = [
    'REFBAND', 'REFMAG', 'SPECTYPE',
    'CFAMCOLS', 'PHOTEXTS'
]
