# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Tools for validating file data
"""


def validate_dict_keys(d, keys, custom_exception=KeyError):
    """
    Verify that a dictionary has exactly the set of input keys
    Will raise the exception in custom_exception if the set of keys in the
    dictionary is not an exact match (no extra, no missing).
    No returns.
    Arguments:
     d: input dictionary
     keys: set object, or object castable to set, containing keys
    Keyword Arguments:
     custom_exception: exception to raise in the case of key mismatch.
      Defaults to KeyError.
    """

    # Check inputs
    if not isinstance(d, dict):
        raise TypeError('d must be a dict')
    if not issubclass(custom_exception, Exception):
        raise TypeError('custom_exception must be descended from Exception')
    try:
        skeys = set(keys)
    except TypeError: # not castable to set
        raise TypeError('keys must be an object castable to a set') # reraise

    # Missing
    misskeys = skeys - set(d.keys())
    if misskeys != set():
        raise custom_exception('Missing keys in input config: ' +
                               str(misskeys))

    # Extra
    extrakeys = set(d.keys()) - skeys
    if extrakeys != set():
        raise custom_exception('Extra top-level keys in input file: + ' +
                               str(extrakeys))


def validate_set(s, keys, custom_exception=KeyError):
    """
    Verify that a set has exactly the same values as a specified set of keys
    Will raise the exception in custom_exception if the input set
    is not an exact match to keys (no extra, no missing).
    No returns.
    Arguments:
     d: input set object, or object castable to set
     keys: set object, or object castable to set, containing keys to match
    Keyword Arguments:
     custom_exception: exception to raise in the case of key mismatch.
      Defaults to KeyError.
    """

    # Check inputs
    try:
        ss = set(s)
    except TypeError: # not castable to set
        raise TypeError('s must be an object castable to a set') # reraise
    if not issubclass(custom_exception, Exception):
        raise TypeError('custom_exception must be descended from Exception')
    try:
        skeys = set(keys)
    except TypeError: # not castable to set
        raise TypeError('keys must be an object castable to a set') # reraise

    # Missing
    misskeys = skeys - ss
    if misskeys != set():
        raise custom_exception('Missing keys in input config: ' +
                               str(misskeys))

    # Extra
    extrakeys = ss - skeys
    if extrakeys != set():
        raise custom_exception('Extra top-level keys in input file: + ' +
                               str(extrakeys))
