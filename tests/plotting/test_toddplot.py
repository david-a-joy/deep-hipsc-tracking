""" Tests for the plotting framework """

import unittest

import pandas as pd

from deep_hipsc_tracking.plotting import toddplot

# Tests


class TestDataManager(unittest.TestCase):
    """ Test the bar plotter code """

    def test_calc_bar_xcoords_one_class(self):

        df = pd.DataFrame({
            'Class': ['a', 'a', 'a'],
            'Value': [1.0, 2.0, 1.0],
        })

        mgr = toddplot.DataManager(xcolumn='Class', ycolumn='Value')
        mgr.load_raw_data(df)
        mgr.calc_bar_xcoords()

        xcoords = [0.9]
        xtick_coords = [0.9]
        xcoord_map = {'a': 0.9}

        self.assertEqual(mgr._xcoords, xcoords)
        self.assertEqual(mgr._xtick_coords, xtick_coords)
        self.assertEqual(mgr._xcoord_map, xcoord_map)

    def test_calc_bar_xcoords_two_class(self):

        df = pd.DataFrame({
            'Class': ['a', 'a', 'a', 'b', 'b', 'b'],
            'Value': [1.0, 2.0, 1.0, 2.0, 4.0, 2.0],
        })

        mgr = toddplot.DataManager(xcolumn='Class', ycolumn='Value')
        mgr.load_raw_data(df)
        mgr.calc_bar_xcoords()

        xcoords = [0.9, 2.3]
        xtick_coords = [0.9, 2.3]
        xcoord_map = {'a': 0.9, 'b': 2.3}

        self.assertEqual(mgr._xcoords, xcoords)
        self.assertEqual(mgr._xtick_coords, xtick_coords)
        self.assertEqual(mgr._xcoord_map, xcoord_map)
