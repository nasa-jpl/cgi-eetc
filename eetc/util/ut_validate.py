# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for validate.py functions"""

import unittest

from eetc.util.validate import validate_dict_keys, validate_set

class TestValidateDictKeys(unittest.TestCase):
    """
    Tests for the function that checks if a dict's keys match the keys in a
    given iterable
    """

    def test_success(self):
        """Good inputs success without issues"""
        d = {'a':0, 'b':1, 'c':3}
        keys = ['a', 'b', 'c']
        validate_dict_keys(d, keys)
        pass


    def test_invalid_d(self):
        """invalid inputs caught"""
        d = ['a', 'b', 'c']
        keys = ['a', 'b', 'c']
        with self.assertRaises(TypeError):
            validate_dict_keys(d, keys, custom_exception=Exception)
        pass


    def test_invalid_keys(self):
        """invalid inputs caught"""
        d = {'a':0, 'b':1, 'c':3}
        keys = [{'a':0}, 'b', 'c']
        with self.assertRaises(TypeError):
            validate_dict_keys(d, keys, custom_exception=Exception)
        pass


    def test_invalid_custom_exception(self):
        """invalid inputs caught"""
        d = {'a':0, 'b':1, 'c':3}
        keys = ['a', 'b', 'c']
        with self.assertRaises(TypeError):
            validate_dict_keys(d, keys, custom_exception=None)
        pass


    def test_ok_iterables(self):
        """check that hashable inputs for keys are OK"""
        d = {'a':0, 'b':1, 'c':3}
        keyslist = [{'a', 'b', 'c'},
                    ['a', 'b', 'c'],
                    ('a', 'b', 'c'),
                    d.keys(),
                    ]

        for keys in keyslist:
            validate_dict_keys(d, keys)
            pass
        pass


    def test_missing_keys(self):
        """missing keys caught as expected"""
        d = {'a':0, 'b':1, 'c':3}
        keys = ['a', 'b']
        with self.assertRaises(KeyError):
            validate_dict_keys(d, keys)
        with self.assertRaises(TypeError):
            validate_dict_keys(d, keys, custom_exception=TypeError)
        pass


    def test_extra_keys(self):
        """extra keys caught as expected"""
        d = {'a':0, 'b':1, 'c':3}
        keys = ['a', 'b', 'c', 'd']
        with self.assertRaises(KeyError):
            validate_dict_keys(d, keys)
        with self.assertRaises(TypeError):
            validate_dict_keys(d, keys, custom_exception=TypeError)
        pass


    def test_missing_and_extra_keys(self):
        """missing/extra keys caught as expected"""
        d = {'a':0, 'b':1, 'c':3}
        keys = ['a', 'b', 'd']
        with self.assertRaises(KeyError):
            validate_dict_keys(d, keys)
        with self.assertRaises(TypeError):
            validate_dict_keys(d, keys, custom_exception=TypeError)
        pass


class TestValidateSet(unittest.TestCase):
    """
    Tests for the function that checks if a set's values match the keys in a
    given iterable
    """

    def test_success(self):
        """Good inputs success without issues"""
        s = {'a', 'b', 'c'}
        keys = ['a', 'b', 'c']
        validate_set(s, keys)
        pass


    def test_success_castable(self):
        """Good inputs success without issues"""
        s = ['b', 'a', 'c']
        keys = ['a', 'b', 'c']
        validate_set(s, keys)
        pass


    def test_invalid_s(self):
        """invalid inputs caught"""
        s = 1
        keys = ['a', 'b', 'c']
        with self.assertRaises(TypeError):
            validate_set(s, keys, custom_exception=Exception)
        pass


    def test_invalid_keys(self):
        """invalid inputs caught"""
        s = {'a', 'b', 'c'}
        keys = 1
        with self.assertRaises(TypeError):
            validate_set(s, keys, custom_exception=Exception)
        pass


    def test_invalid_custom_exception(self):
        """invalid inputs caught"""
        s = {'a', 'b', 'c'}
        keys = ['a', 'b', 'c']
        with self.assertRaises(TypeError):
            validate_set(s, keys, custom_exception=None)
        pass


    def test_ok_iterables(self):
        """check that hashable inputs for keys are OK"""
        s = {'a', 'b', 'c'}
        keyslist = [{'a', 'b', 'c'},
                    ['a', 'b', 'c'],
                    ('a', 'b', 'c'),
                    ]

        for keys in keyslist:
            validate_set(s, keys)
            pass
        pass


    def test_missing_keys(self):
        """missing keys caught as expected"""
        d = {'a', 'b', 'c'}
        keys = ['a', 'b']
        with self.assertRaises(KeyError):
            validate_set(d, keys)
        with self.assertRaises(TypeError):
            validate_set(d, keys, custom_exception=TypeError)
        pass


    def test_extra_keys(self):
        """extra keys caught as expected"""
        d = {'a', 'b', 'c'}
        keys = ['a', 'b', 'c', 'd']
        with self.assertRaises(KeyError):
            validate_set(d, keys)
        with self.assertRaises(TypeError):
            validate_set(d, keys, custom_exception=TypeError)
        pass


    def test_missing_and_extra_keys(self):
        """missing/extra keys caught as expected"""
        d = {'a', 'b', 'c'}
        keys = ['a', 'b', 'd']
        with self.assertRaises(KeyError):
            validate_set(d, keys)
        with self.assertRaises(TypeError):
            validate_set(d, keys, custom_exception=TypeError)
        pass


if __name__ == '__main__':
    unittest.main()
