# Imports
import json
import unittest

# 3rd party
import numpy as np

import pandas as pd

# Our own imports

from deep_hipsc_tracking.utils import save_image, stat_utils

from ..helpers import FileSystemTestCase

# Tests


class TestCalcBatchEffect(unittest.TestCase):

    def test_regresses_data_no_batch(self):

        data = pd.DataFrame({
            'Treatment': ['A'] * 5 + ['B'] * 5,
            'Score': [0]*5 + [1]*5,
        })
        data['Score'] += np.random.rand(data.shape[0])*0.1

        res = stat_utils.calc_pairwise_batch_effect(data, category='Treatment',
                                                    batch=[], score='Score')
        exp = data

        np.testing.assert_allclose(res['Score'].values, exp['Score'].values, atol=0.05)

    def test_regresses_data_with_batch_no_effect(self):

        data = pd.DataFrame({
            'Batch': ['1']*10 + ['2']*10,
            'Treatment': ['A']*5 + ['B']*5 + ['A']*5 + ['B']*5,
            'Score': [0]*5 + [1]*5 + [0]*5 + [1]*5,
        })
        data['Score'] += np.random.rand(data.shape[0])*0.1

        res = stat_utils.calc_pairwise_batch_effect(data,
                                                    category='Treatment',
                                                    batch='Batch', score='Score')
        exp = data

        np.testing.assert_allclose(res['Score'].values, exp['Score'].values, atol=0.05)

    def test_regresses_data_with_batch_with_effect(self):

        data = pd.DataFrame({
            'Batch': ['1']*10 + ['2']*10 + ['3']*10,
            'Treatment': ['A']*5 + ['B']*5 + ['A']*5 + ['B']*5 + ['A']*5 + ['B']*5,
            'Score': [0]*5 + [1]*5 + [0]*5 + [1]*5 + [0.2]*5 + [1.2]*5,
        })
        data['Score'] += np.random.rand(data.shape[0])*0.1

        res = stat_utils.calc_pairwise_batch_effect(data,
                                                    category='Treatment',
                                                    batch='Batch', score='Score')
        exp = data.copy()

        exp.loc[exp['Batch'] == '1', 'Score'] += 0.05
        exp.loc[exp['Batch'] == '2', 'Score'] += 0.05
        exp.loc[exp['Batch'] == '3', 'Score'] -= 0.15

        np.testing.assert_allclose(res['Score'].values, exp['Score'].values, atol=0.05)

    def test_regresses_data_with_batch_with_effect_multiple_treatments(self):

        data = pd.DataFrame({
            'Batch': ['1']*12 + ['2']*12 + ['E']*12,
            'Treatment': ['A']*6 + ['B']*6 + ['A']*6 + ['B']*6 + ['A']*6 + ['B']*6,
            'Radius': ['C', 'C', 'M', 'M', 'E', 'E']*6,
            'Score': [0, 0, 0.1, 0.1, 0.2, 0.2, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2]*3,
        })
        data.loc[data['Batch'] == 'E', 'Score'] += 0.2
        data['Score'] += np.random.rand(data.shape[0])*0.1

        res = stat_utils.calc_pairwise_batch_effect(data,
                                                    category=['Treatment', 'Radius'],
                                                    batch='Batch', score='Score')
        exp = data.copy()

        exp.loc[exp['Batch'] == '1', 'Score'] += 0.05
        exp.loc[exp['Batch'] == '2', 'Score'] += 0.05
        exp.loc[exp['Batch'] == 'E', 'Score'] -= 0.15

        np.testing.assert_allclose(res['Score'].values, exp['Score'].values, atol=0.05)

    def test_regresses_data_with_batch_with_effect_no_covariate(self):

        data = pd.DataFrame({
            'Batch': ['E']*10 + ['2']*10 + ['3']*10,
            'Treatment': ['A']*5 + ['B']*5 + ['A']*5 + ['B']*5 + ['A']*5 + ['B']*5,
            'Score': [0]*5 + [1]*5 + [0]*5 + [1]*5 + [0.2]*5 + [1.2]*5,
        })
        data['Score'] += np.random.rand(data.shape[0])*0.1

        res = stat_utils.calc_pairwise_batch_effect(data,
                                                    category=[],
                                                    batch='Batch', score='Score')
        exp = data.copy()
        exp.loc[exp['Batch'] == 'E', 'Score'] += 0.05
        exp.loc[exp['Batch'] == '2', 'Score'] += 0.05
        exp.loc[exp['Batch'] == '3', 'Score'] -= 0.15

        np.testing.assert_allclose(res['Score'].values, exp['Score'].values, atol=0.05)

    def test_regresses_data_with_batch_with_effect_multiple_batches(self):

        data = pd.DataFrame({
            'Batch': ['1']*12 + ['2']*12 + ['E']*12,
            'Treatment': ['A']*6 + ['B']*6 + ['A']*6 + ['B']*6 + ['A']*6 + ['B']*6,
            'Radius': ['C', 'C', 'M', 'M', 'E', 'E']*6,
            'Score': [0, 0, 0.1, 0.1, 0.2, 0.2, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2]*3,
        })
        data.loc[data['Batch'] == 'E', 'Score'] += 0.2
        data['Score'] += np.random.rand(data.shape[0])*0.1

        res = stat_utils.calc_pairwise_batch_effect(data,
                                                    category=['Treatment'],
                                                    batch=['Batch', 'Radius'],
                                                    score='Score')
        exp = data.copy()

        # Offset for batch
        exp.loc[exp['Batch'] == '1', 'Score'] += 0.05
        exp.loc[exp['Batch'] == '2', 'Score'] += 0.05
        exp.loc[exp['Batch'] == 'E', 'Score'] -= 0.15

        # Offset for radius
        exp.loc[exp['Radius'] == 'C', 'Score'] += 0.1
        exp.loc[exp['Radius'] == 'M', 'Score'] += 0.0
        exp.loc[exp['Radius'] == 'E', 'Score'] -= 0.1

        np.testing.assert_allclose(res['Score'].values, exp['Score'].values, atol=0.05)


class TestBinByRadius(unittest.TestCase):

    def test_bins_using_equal_radii(self):

        radius = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        value = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        res = stat_utils.bin_by_radius(radius, value, bin_type='radius', num_bins=2)
        exp = pd.DataFrame({
            'Radius': ['0.0-0.5']*5 + ['0.5-1.0']*5,
            'Value': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })

        pd.testing.assert_frame_equal(res, exp)

    def test_bins_using_equal_area(self):

        radius = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        value = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        res = stat_utils.bin_by_radius(radius, value, bin_type='area', num_bins=2)
        exp = pd.DataFrame({
            'Radius': ['0.0-0.7']*8 + ['0.7-1.0']*2,
            'Value': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })

        pd.testing.assert_frame_equal(res, exp)

    def test_bins_ignoring_nans(self):

        radius = np.array([0, 0.1, 0.2, 0.3, np.nan, 0.5, 0.6, 0.7, 0.8, 0.9])
        value = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        res = stat_utils.bin_by_radius(radius, value, bin_type='area', num_bins=2)
        exp = pd.DataFrame({
            'Radius': ['0.0-0.7']*7 + ['0.7-1.0']*2,
            'Value': [0, 1, 0, 1, 1, 0, 1, 0, 1],
        })

        pd.testing.assert_frame_equal(res, exp)


class TestCalcPairwiseEffectSize(unittest.TestCase):

    def test_pairwise_effect_size_one_comp(self):

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A']*2 + ['B', 'B', 'B']*2,
            'score': [1.0, 1.1, 0.9]*2 + [0.5, 0.4, 0.6]*2,
        })

        res = stat_utils.calc_pairwise_effect_size(df, 'cond', 'score')
        exp = {
            ('A', 'B'): 6.7082,
        }

        self.assertEqual(res.keys(), exp.keys())
        for key in res:
            np.testing.assert_almost_equal(res[key], exp[key], decimal=3)

    def test_pairwise_effect_size_two_comp(self):

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A']*2 + ['B', 'B', 'B']*2 + ['C', 'C', 'C']*2,
            'score': [1.0, 1.1, 0.9]*2 + [0.5, 0.4, 0.6]*2 + [0.51, 0.41, 0.61]*2,
        })

        res = stat_utils.calc_pairwise_effect_size(df, 'cond', 'score')
        exp = {
            ('A', 'B'): 6.7082,
            ('A', 'C'): 6.5740,
            ('B', 'C'): 0.1342,
        }

        self.assertEqual(res.keys(), exp.keys())
        for key in res:
            np.testing.assert_almost_equal(res[key], exp[key], decimal=3)


class TestCalcPairwiseANOVA(unittest.TestCase):

    def test_pairwise_anova_one_term(self):

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A']*2 + ['B', 'B', 'B']*2,
            'score': [1.0, 1.1, 0.9]*2 + [0.5, 0.4, 0.6]*2,
        })

        res = stat_utils.calc_pairwise_anova(df, 'cond', 'score')
        exp = pd.DataFrame({
            'df': [1.0, 10.0],
            'sum_sq': [0.75, 0.08],
            'mean_sq': [0.75, 0.008],
            'F': [93.75, np.nan],
            'PR(>F)': [0.000002, np.nan],
        }, index=['cond', 'Residual'])

        self.assertTrue(np.all(res.columns == exp.columns))
        for key in exp.columns:
            np.testing.assert_almost_equal(res[key].values, exp[key].values)

    def test_pairwise_anova_two_term_no_interactions(self):

        df = pd.DataFrame({
            'cond1': ['A', 'A', 'A']*2 + ['B', 'B', 'B']*2,
            'cond2': ['A', 'B', 'A', 'B', 'A', 'B']*2,
            'score': [1.0, 1.1, 0.9]*2 + [0.5, 0.4, 0.6]*2,
        })

        res = stat_utils.calc_pairwise_anova(
            df, ('cond1', 'cond2'), 'score', include_interactions=False)
        exp = pd.DataFrame({
            'df': [1.0, 1.0, 9.0],
            'sum_sq': [7.500000e-01, 2.773339e-32, 0.08],
            'mean_sq': [7.500000e-01, 2.773339e-32, 8.888889e-03],
            'F': [8.437500e+01, 3.120007e-30, np.nan],
            'PR(>F)': [7.2228848e-06, 1.000000, np.nan],
        }, index=['cond1', 'cond2', 'Residual'])

        self.assertTrue(np.all(res.columns == exp.columns))
        for key in exp.columns:
            np.testing.assert_almost_equal(res[key].values, exp[key].values)

    def test_pairwise_anova_two_term_with_interactions(self):

        df = pd.DataFrame({
            'cond1': ['A', 'A', 'A']*2 + ['B', 'B', 'B']*2,
            'cond2': ['A', 'B', 'A', 'B', 'A', 'B']*2,
            'score': [1.0, 1.1, 0.9]*2 + [0.5, 0.4, 0.6]*2,
        })

        res = stat_utils.calc_pairwise_anova(
            df, ('cond1', 'cond2'), 'score', include_interactions=True)
        exp = pd.DataFrame({
            'df': [1.0, 1.0, 1.0, 8.0],
            'sum_sq': [0.75, 4.930381e-32, 3.081488e-33, 8.0e-2],
            'mean_sq': [0.75, 4.930381e-32, 3.081488e-33, 1.0e-2],
            'F': [7.500000e+01, 4.930381e-30, 3.081488e-31, np.nan],
            'PR(>F)': [2.4568408e-05, 1.0, 1.0, np.nan],
        }, index=['cond1', 'cond2', 'cond1:cond2', 'Residual'])

        self.assertTrue(np.all(res.columns == exp.columns))
        for key in exp.columns:
            np.testing.assert_almost_equal(res[key].values, exp[key].values)


class TestCalcPairwiseSignificance(unittest.TestCase):

    def test_pairwise_significance_one_condition(self):

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A'],
            'score': [1.0, 1.1, 0.9],
        })

        res = stat_utils.calc_pairwise_significance(df, 'cond', 'score', alpha=0.05)
        exp = {}

        self.assertEqual(res, exp)

    def test_pairwise_significance_two_conditions(self):

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A', 'B', 'B', 'B'],
            'score': [1.0, 1.1, 0.9, 0.5, 0.4, 0.6],
        })

        res = stat_utils.calc_pairwise_significance(df, 'cond', 'score', alpha=0.05)
        exp = {
            ('A', 'B'): 0.0036,
        }
        self.assertEqual(set(res), set(exp))
        for key in res:
            self.assertEqual(round(res[key], 4), round(exp[key], 4))

    def test_pairwise_significance_three_conditions_non_significant(self):

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A']*2 + ['B', 'B', 'B']*2 + ['C', 'C', 'C']*2,
            'score': [1.0, 1.1, 0.9]*2 + [0.5, 0.4, 0.6]*2 + [0.5, 0.45, 0.6]*2,
        })

        res = stat_utils.calc_pairwise_significance(df, 'cond', 'score',
                                                    alpha=0.05,
                                                    test_fxn='t-test',
                                                    keep_non_significant=True)
        exp = {
            ('A', 'B'): 6.40469e-6,
            ('A', 'C'): 6.40469e-6,
            ('B', 'C'): 0.72435,
        }
        self.assertEqual(set(res), set(exp))
        for key in res:
            self.assertEqual(round(res[key], 4), round(exp[key], 4))

    def test_pairwise_significance_two_conditions_u_test(self):

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A']*2 + ['B', 'B', 'B']*2,
            'score': [1.0, 1.1, 0.9]*2 + [0.5, 0.4, 0.6]*2,
        })

        res = stat_utils.calc_pairwise_significance(df, 'cond', 'score',
                                                    alpha=0.05,
                                                    test_fxn='u-test')
        exp = {
            ('A', 'B'): 0.0046,
        }
        self.assertEqual(set(res), set(exp))
        for key in res:
            self.assertEqual(round(res[key], 4), round(exp[key], 4))

    def test_pairwise_significance_two_conditions_ks_test(self):

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A']*2 + ['B', 'B', 'B']*2,
            'score': [1.0, 1.1, 0.9]*2 + [0.5, 0.4, 0.6]*2,
        })

        res = stat_utils.calc_pairwise_significance(df, 'cond', 'score',
                                                    alpha=0.05,
                                                    test_fxn='ks-test')
        exp = {
            ('A', 'B'): 0.0022,
        }
        self.assertEqual(set(res), set(exp))
        for key in res:
            self.assertEqual(round(res[key], 4), round(exp[key], 4))

    def test_pairwise_significance_two_conditions_custom(self):

        def test_fxn(x, y):
            return 0.0

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A']*2 + ['B', 'B', 'B']*2,
            'score': [1.0, 1.1, 0.9]*2 + [0.5, 0.4, 0.6]*2,
        })

        res = stat_utils.calc_pairwise_significance(df, 'cond', 'score',
                                                    alpha=0.05,
                                                    test_fxn=test_fxn)
        exp = {
            ('A', 'B'): 0.0,
        }
        self.assertEqual(set(res), set(exp))
        for key in res:
            self.assertEqual(round(res[key], 4), round(exp[key], 4))

    def test_pairwise_significance_three_conditions(self):

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
            'score': [1.0, 1.1, 0.9, 0.5, 0.4, 0.6, 0.49, 0.39, 0.59],
        })

        res = stat_utils.calc_pairwise_significance(df, 'cond', 'score', alpha=0.05)
        exp = {
            ('A', 'B'): 0.01,
            ('A', 'C'): 0.01,
        }
        self.assertEqual(set(res), set(exp))
        for key in res:
            self.assertEqual(round(res[key], 4), round(exp[key], 4))

    def test_pairwise_significance_three_conditions_one_control(self):

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
            'score': [1.0, 1.1, 0.9, 0.5, 0.4, 0.6, 0.49, 0.39, 0.59],
        })

        res = stat_utils.calc_pairwise_significance(df, 'cond', 'score', alpha=0.05, control='A')
        exp = {
            ('A', 'B'): 0.0067,
            ('A', 'C'): 0.0067,
        }
        self.assertEqual(set(res), set(exp))
        for key in res:
            self.assertEqual(round(res[key], 4), round(exp[key], 4))

    def test_pairwise_significance_two_by_three_conditions_welchs_t(self):

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
            'type': ['1', '1', '1', '2', '1', '2', '1', '2', '1'],
            'score': [1.0, 1.1, 0.9, 0.5, 0.4, 0.6, 0.49, 0.39, 0.59],
        })

        res = stat_utils.calc_pairwise_significance(df, ('cond', 'type'), 'score', alpha=0.05,
                                                    test_fxn='welch-t-test')
        exp = {
            (('A', '1'), ('B', '2')): 0.03083,
            (('A', '1'), ('C', '1')): 0.03083,
        }
        self.assertEqual(set(res), set(exp))
        for key in res:
            self.assertEqual(round(res[key], 4), round(exp[key], 4))

    def test_pairwise_significance_two_by_three_conditions_students_t(self):

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
            'type': ['1', '1', '1', '2', '1', '2', '1', '2', '1'],
            'score': [1.0, 1.1, 0.9, 0.5, 0.4, 0.6, 0.49, 0.39, 0.59],
        })

        res = stat_utils.calc_pairwise_significance(df, ('cond', 'type'),
                                                    'score',
                                                    alpha=0.05,
                                                    test_fxn='t-test')
        exp = {
            (('A', '1'), ('B', '2')): 0.0351,
            (('A', '1'), ('C', '1')): 0.0351,
        }
        self.assertEqual(set(res), set(exp))
        for key in res:
            self.assertEqual(round(res[key], 4), round(exp[key], 4))

    def test_pairwise_significance_with_several_groups(self):

        df = pd.DataFrame({
            'cond': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B'],
            'type': ['1', '1', '1', '2', '2', '2', '1', '1', '1', '2', '2', '2'],
            'score': [1.0, 1.1, 0.9, 0.5, 0.4, 0.6, 2.0, 2.1, 1.9, 0.1, 0.2, 0.15],
        })

        # Note that A1 vs B1 and A1 vs B2 would also be significant
        res = stat_utils.calc_pairwise_significance(df, 'type', 'score',
                                                    alpha=0.05,
                                                    group='cond',
                                                    test_fxn='t-test')
        exp = {
            (('A', '1'), ('A', '2')): 0.003602,
            (('B', '1'), ('B', '2')): 1.7642e-05,
        }
        self.assertEqual(set(res), set(exp))
        for key in res:
            self.assertEqual(round(res[key], 4), round(exp[key], 4))


class TestLoadTrainTestSplit(FileSystemTestCase):

    def test_loads_basic_file(self):

        split_data = {
            "train_files": [
                "/foo/bar/640cell.png",
                "/foo/bar/1505cell.png",
                "/foo/bar/826cell.png",
            ],
            "validation_files": [
                "/foo/bar/555cell.png",
                "/foo/bar/001cell.png",
                "/foo/bar/1900cell.png",
            ]
        }
        split_file = self.tempdir / "snapshot" / "train_test_split.json"
        split_file.parent.mkdir(exist_ok=True, parents=True)
        with split_file.open('wt') as fp:
            json.dump(split_data, fp)

        res = stat_utils.load_train_test_split(self.tempdir)
        exp = {
            "train": [640, 1505, 826],
            "test": [],
            "validation": [555, 1, 1900],
        }
        self.assertEqual(res, exp)


class TestPairAllTileData(unittest.TestCase):

    def tile_data(self, experiment, tile, timepoint):

        class BaseTileData(object):

            def __init__(self, experiment, tile, timepoint):

                self.experiment = experiment
                self.tile = tile
                self.timepoint = timepoint

                self.prev_tile = None
                self.next_tile = None

        return BaseTileData(experiment, tile, timepoint)

    def test_pairs_one_timepoint_same_tile(self):

        all_tile_data = [
            self.tile_data('2017-01-30', 1, 1),
        ]

        res = stat_utils.pair_all_tile_data(all_tile_data)
        exp = []
        self.assertEqual(res, exp)

    def test_errors_on_exact_duplicates(self):

        all_tile_data = [
            self.tile_data('2017-01-30', 1, 1),
            self.tile_data('2017-01-30', 1, 1),
        ]

        with self.assertRaises(ValueError):
            stat_utils.pair_all_tile_data(all_tile_data)

    def test_pairs_one_timepoint_different_tiles(self):

        all_tile_data = [
            self.tile_data('2017-01-30', 1, 1),
            self.tile_data('2017-01-30', 2, 1),
        ]

        res = stat_utils.pair_all_tile_data(all_tile_data)
        exp = []
        self.assertEqual(res, exp)

    def test_pairs_one_timepoint_different_experiments(self):

        all_tile_data = [
            self.tile_data('2017-01-30', 1, 1),
            self.tile_data('2017-03-03', 1, 1),
        ]

        res = stat_utils.pair_all_tile_data(all_tile_data)
        exp = []
        self.assertEqual(res, exp)

    def test_pairs_two_timepoints_same_tile(self):

        all_tile_data = [
            self.tile_data('2017-01-30', 1, 1),
            self.tile_data('2017-01-30', 1, 2),
        ]

        res = stat_utils.pair_all_tile_data(all_tile_data)
        exp = [
            all_tile_data[0],
        ]
        self.assertEqual(res, exp)

    def test_pairs_three_timepoints_same_tile(self):

        all_tile_data = [
            self.tile_data('2017-01-30', 1, 1),
            self.tile_data('2017-01-30', 1, 2),
            self.tile_data('2017-01-30', 1, 3),
        ]

        res = stat_utils.pair_all_tile_data(all_tile_data)
        exp = [
            all_tile_data[0], all_tile_data[1],
        ]
        self.assertEqual(res, exp)

    def test_pairs_two_timepoints_different_tiles(self):

        all_tile_data = [
            self.tile_data('2017-01-30', 1, 1),
            self.tile_data('2017-01-30', 1, 2),
            self.tile_data('2017-01-30', 2, 1),
            self.tile_data('2017-01-30', 2, 2),
        ]

        res = stat_utils.pair_all_tile_data(all_tile_data)
        exp = [
            all_tile_data[0], all_tile_data[2],
        ]
        self.assertEqual(res, exp)

    def test_pairs_three_timepoints_different_tiles(self):

        all_tile_data = [
            self.tile_data('2017-01-30', 1, 1),
            self.tile_data('2017-01-30', 1, 2),
            self.tile_data('2017-01-30', 1, 3),
            self.tile_data('2017-01-30', 2, 1),
            self.tile_data('2017-01-30', 2, 2),
            self.tile_data('2017-01-30', 2, 3),
        ]

        res = stat_utils.pair_all_tile_data(all_tile_data)
        exp = [
            all_tile_data[0], all_tile_data[1], all_tile_data[3], all_tile_data[4],
        ]
        self.assertEqual(res, exp)

    def test_pairs_two_timepoints_different_experiments(self):

        all_tile_data = [
            self.tile_data('2017-01-30', 1, 1),
            self.tile_data('2017-01-30', 1, 2),
            self.tile_data('2017-03-03', 1, 1),
            self.tile_data('2017-03-03', 1, 2),
        ]

        res = stat_utils.pair_all_tile_data(all_tile_data)
        exp = [
            all_tile_data[0], all_tile_data[2],
        ]
        self.assertEqual(res, exp)

    def test_pairs_three_timepoints_different_experiments(self):

        all_tile_data = [
            self.tile_data('2017-01-30', 1, 1),
            self.tile_data('2017-01-30', 1, 2),
            self.tile_data('2017-01-30', 1, 3),
            self.tile_data('2017-03-03', 1, 1),
            self.tile_data('2017-03-03', 1, 2),
            self.tile_data('2017-03-03', 1, 3),
        ]

        res = stat_utils.pair_all_tile_data(all_tile_data)
        exp = [
            all_tile_data[0], all_tile_data[1], all_tile_data[3], all_tile_data[4],
        ]
        self.assertEqual(res, exp)

    def test_linked_tiles_have_next_tile(self):

        all_tile_data = [
            self.tile_data('2017-01-30', 1, 1),
            self.tile_data('2017-01-30', 1, 2),
            self.tile_data('2017-01-30', 1, 3),
            self.tile_data('2017-01-30', 2, 1),
            self.tile_data('2017-01-30', 2, 2),
            self.tile_data('2017-01-30', 2, 3),
            self.tile_data('2017-03-03', 1, 1),
            self.tile_data('2017-03-03', 1, 2),
            self.tile_data('2017-03-03', 1, 3),
        ]

        res = stat_utils.pair_all_tile_data(all_tile_data)
        exp = [
            all_tile_data[0], all_tile_data[1],
            all_tile_data[3], all_tile_data[4],
            all_tile_data[6], all_tile_data[7],
        ]
        self.assertEqual(res, exp)

        # Next tile and prev tile should be sensible
        for tile_data in res:
            self.assertIsNotNone(tile_data.next_tile)
            next_tile = tile_data.next_tile
            self.assertIs(next_tile.prev_tile, tile_data)

        # Start tiles shouldn't have a predecessor
        start_tiles = [exp[0], exp[2], exp[4]]
        for tile_data in start_tiles:
            self.assertIsNone(tile_data.prev_tile)

        # End tiles are one from the end because next_tile is always defined
        end_tiles = [exp[1], exp[3], exp[5]]
        for tile_data in end_tiles:
            self.assertIsNone(tile_data.next_tile.next_tile)


class TestPairTrainTestData(FileSystemTestCase):

    def test_pair_points_test_with_train_split(self):
        mode = 'points'
        datatype = 'test'

        split_data = {
            "train_files": [
                "/foo/bar/640cell.png",
                "/foo/bar/1505cell.png",
                "/foo/bar/826cell.png",
            ],
            "validation_files": [
                "/foo/bar/555cell.png",
                "/foo/bar/001cell.png",
                "/foo/bar/1900cell.png",
            ]
        }

        train_data = self.tempdir / 'train'
        train_data.mkdir(exist_ok=True, parents=True)
        (train_data / '640dots.png').touch()
        (train_data / '1505dots.png').touch()
        (train_data / '826dots.png').touch()
        (train_data / '555dots.png').touch()
        (train_data / '001dots.png').touch()
        (train_data / '1900dots.png').touch()

        test_data = self.tempdir / 'test'
        test_data.mkdir(exist_ok=True, parents=True)
        (test_data / '640cell.csv').touch()
        (test_data / '1505cell.csv').touch()
        (test_data / '826cell.csv').touch()
        (test_data / '555cell.csv').touch()
        (test_data / '001cell.csv').touch()
        (test_data / '1900cell.csv').touch()

        split_file = self.tempdir / "test" / "snapshot" / "train_test_split.json"
        split_file.parent.mkdir(exist_ok=True, parents=True)
        with split_file.open('wt') as fp:
            json.dump(split_data, fp)

        res = stat_utils.pair_train_test_data(train_data, test_data, datatype=datatype, mode=mode,
                                              data_split="train")
        exp = [
            (train_data / '640dots.png', test_data / '640cell.csv'),
            (train_data / '826dots.png', test_data / '826cell.csv'),
            (train_data / '1505dots.png', test_data / '1505cell.csv'),
        ]
        self.assertEqual(res, exp)

    def test_pair_points_test_with_validation_split(self):
        mode = 'points'
        datatype = 'test'

        split_data = {
            "train_files": [
                "/foo/bar/640cell.png",
                "/foo/bar/1505cell.png",
                "/foo/bar/826cell.png",
            ],
            "validation_files": [
                "/foo/bar/555cell.png",
                "/foo/bar/001cell.png",
                "/foo/bar/1900cell.png",
            ]
        }

        train_data = self.tempdir / 'train'
        train_data.mkdir(exist_ok=True, parents=True)
        (train_data / '640dots.png').touch()
        (train_data / '1505dots.png').touch()
        (train_data / '826dots.png').touch()
        (train_data / '555dots.png').touch()
        (train_data / '001dots.png').touch()
        (train_data / '1900dots.png').touch()

        test_data = self.tempdir / 'test'
        test_data.mkdir(exist_ok=True, parents=True)
        (test_data / '640cell.csv').touch()
        (test_data / '1505cell.csv').touch()
        (test_data / '826cell.csv').touch()
        (test_data / '555cell.csv').touch()
        (test_data / '001cell.csv').touch()
        (test_data / '1900cell.csv').touch()

        split_file = self.tempdir / "test" / "snapshot" / "train_test_split.json"
        split_file.parent.mkdir(exist_ok=True, parents=True)
        with split_file.open('wt') as fp:
            json.dump(split_data, fp)

        res = stat_utils.pair_train_test_data(train_data, test_data, datatype=datatype, mode=mode,
                                              data_split="validation")
        exp = [
            (train_data / '001dots.png', test_data / '001cell.csv'),
            (train_data / '555dots.png', test_data / '555cell.csv'),
            (train_data / '1900dots.png', test_data / '1900cell.csv'),
        ]
        self.assertEqual(res, exp)

    def test_pair_points_test(self):
        mode = 'points'
        datatype = 'test'

        train_data = self.tempdir / 'train'
        train_data.mkdir(exist_ok=True, parents=True)
        (train_data / '000dots.png').touch()
        (train_data / '100dots.png').touch()
        (train_data / '999dots.png').touch()
        (train_data / '1000dots.png').touch()

        test_data = self.tempdir / 'test'
        test_data.mkdir(exist_ok=True, parents=True)
        (test_data / '000cell.csv').touch()
        (test_data / '100cell.csv').touch()
        (test_data / '999cell.csv').touch()
        (test_data / '1000cell.csv').touch()

        res = stat_utils.pair_train_test_data(train_data, test_data, datatype=datatype, mode=mode)
        exp = [
            (train_data / '000dots.png', test_data / '000cell.csv'),
            (train_data / '100dots.png', test_data / '100cell.csv'),
            (train_data / '999dots.png', test_data / '999cell.csv'),
            (train_data / '1000dots.png', test_data / '1000cell.csv'),
        ]
        self.assertEqual(res, exp)

    def test_pair_masks_test(self):
        mode = 'masks'
        datatype = 'test'

        train_data = self.tempdir / 'train'
        train_data.mkdir(exist_ok=True, parents=True)
        (train_data / '000dots.png').touch()
        (train_data / '100dots.png').touch()
        (train_data / '999dots.png').touch()
        (train_data / '1000dots.png').touch()

        test_data = self.tempdir / 'test'
        test_data.mkdir(exist_ok=True, parents=True)
        (test_data / '000cell_resp.png').touch()
        (test_data / '100cell_resp.png').touch()
        (test_data / '999cell_resp.png').touch()
        (test_data / '1000cell_resp.png').touch()

        res = stat_utils.pair_train_test_data(train_data, test_data, datatype=datatype, mode=mode)
        exp = [
            (train_data / '000dots.png', test_data / '000cell_resp.png'),
            (train_data / '100dots.png', test_data / '100cell_resp.png'),
            (train_data / '999dots.png', test_data / '999cell_resp.png'),
            (train_data / '1000dots.png', test_data / '1000cell_resp.png'),
        ]
        self.assertEqual(res, exp)

    def test_pair_points_real(self):

        mode = 'points'
        datatype = 'real'

        train_data = self.tempdir / 'train'
        train_data.mkdir(exist_ok=True, parents=True)

        df = pd.DataFrame({
            'cell_number': [0, 100, 999, 1000],
            'experiment': ['2017-01-30', '2017-01-30', '2017-03-03', '2017-03-03'],
            'file':	[
                '/data/Experiment/2017-01-30/Corrected/EGFP/s01/EGFP-s01t001.tif',
                '/data/Experiment/2017-01-30/Corrected/EGFP/s01/EGFP-s01t002.tif',
                '/data/Experiment/2017-03-03/Corrected/EGFP/s02/EGFP-s02t001.tif',
                '/data/Experiment/2017-03-03/Corrected/EGFP/s03/EGFP-s03t001.tif',
            ],
            'flip': ['none', 'none', 'none', 'none'],
            'next_cell': [1, np.nan, np.nan, np.nan],
            'prev_cell': [np.nan, 0, np.nan, np.nan],
            'rot90': [0, 0, 0, 0],
            'tile': [1, 1, 2, 3],
            'timepoint': [1, 2, 1, 1],
        })
        df.to_excel(train_data / 'index.xlsx')

        (train_data / '000dots.png').touch()
        (train_data / '100dots.png').touch()
        (train_data / '999dots.png').touch()
        (train_data / '1000dots.png').touch()

        test_data = self.tempdir / 'test'
        test_data.mkdir(exist_ok=True, parents=True)
        f1 = test_data / '2017-01-30/SingleCell/Corrected/EGFP/s01/EGFP-s01t001.csv'
        f2 = test_data / '2017-01-30/SingleCell/Corrected/EGFP/s01/EGFP-s01t002.csv'
        f3 = test_data / '2017-03-03/SingleCell/Corrected/EGFP/s02/EGFP-s02t001.csv'
        f4 = test_data / '2017-03-03/SingleCell/Corrected/EGFP/s03/EGFP-s03t001.csv'
        for f in [f1, f2, f3, f4]:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()

        res = stat_utils.pair_train_test_data(train_data, test_data, datatype=datatype, mode=mode)
        exp = [
            (train_data / '000dots.png', f1),
            (train_data / '100dots.png', f2),
            (train_data / '999dots.png', f3),
            (train_data / '1000dots.png', f4),
        ]
        self.assertEqual(res, exp)

    def test_pair_masks_real(self):

        mode = 'masks'
        datatype = 'real'

        train_data = self.tempdir / 'train'
        train_data.mkdir(exist_ok=True, parents=True)

        df = pd.DataFrame({
            'cell_number': [0, 100, 999, 1000],
            'experiment': ['2017-01-30', '2017-01-30', '2017-03-03', '2017-03-03'],
            'file':	[
                '/data/Experiment/2017-01-30/Corrected/EGFP/s01/EGFP-s01t001.tif',
                '/data/Experiment/2017-01-30/Corrected/EGFP/s01/EGFP-s01t002.tif',
                '/data/Experiment/2017-03-03/Corrected/EGFP/s02/EGFP-s02t001.tif',
                '/data/Experiment/2017-03-03/Corrected/EGFP/s03/EGFP-s03t001.tif',
            ],
            'flip': ['none', 'none', 'none', 'none'],
            'next_cell': [1, np.nan, np.nan, np.nan],
            'prev_cell': [np.nan, 0, np.nan, np.nan],
            'rot90': [0, 0, 0, 0],
            'tile': [1, 1, 2, 3],
            'timepoint': [1, 2, 1, 1],
        })
        df.to_excel(train_data / 'index.xlsx')

        (train_data / '000dots.png').touch()
        (train_data / '100dots.png').touch()
        (train_data / '999dots.png').touch()
        (train_data / '1000dots.png').touch()

        test_data = self.tempdir / 'test'
        test_data.mkdir(exist_ok=True, parents=True)
        f1 = test_data / '2017-01-30/SingleCell/Corrected/EGFP/s01/EGFP-s01t001_resp.png'
        f2 = test_data / '2017-01-30/SingleCell/Corrected/EGFP/s01/EGFP-s01t002_resp.png'
        f3 = test_data / '2017-03-03/SingleCell/Corrected/EGFP/s02/EGFP-s02t001_resp.png'
        f4 = test_data / '2017-03-03/SingleCell/Corrected/EGFP/s03/EGFP-s03t001_resp.png'
        for f in [f1, f2, f3, f4]:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()

        res = stat_utils.pair_train_test_data(train_data, test_data, datatype=datatype, mode=mode)
        exp = [
            (train_data / '000dots.png', f1),
            (train_data / '100dots.png', f2),
            (train_data / '999dots.png', f3),
            (train_data / '1000dots.png', f4),
        ]
        self.assertEqual(res, exp)


class TestIndexAllTestFiles(FileSystemTestCase):

    def test_pairs_test_files_all_matches(self):

        f1 = self.tempdir / '2017-01-30/SingleCell/Corrected/EGFP/s01/EGFP-s01t001.tif'
        f2 = self.tempdir / '2017-01-30/SingleCell/Corrected/EGFP/s01/EGFP-s01t002.tif'
        f3 = self.tempdir / '2017-01-30/SingleCell/Corrected/EGFP/s02/EGFP-s02t001.tif'

        f1.parent.mkdir(exist_ok=True, parents=True)
        f1.touch()
        f2.parent.mkdir(exist_ok=True, parents=True)
        f2.touch()
        f3.parent.mkdir(exist_ok=True, parents=True)
        f3.touch()

        index = {
            stat_utils.CellIndex('2017-01-30', 1, 1, 0, 'none'): 1,
            stat_utils.CellIndex('2017-01-30', 1, 2, 0, 'none'): 2,
            stat_utils.CellIndex('2017-01-30', 2, 1, 0, 'none'): 3,
        }
        res = stat_utils.index_all_test_files(self.tempdir, index)
        exp = {
            1: f1,
            2: f2,
            3: f3,
        }

        self.assertEqual(res, exp)


class TestLoadIndex(FileSystemTestCase):

    def test_sane_with_reasonable_index(self):

        df = pd.DataFrame({
            'cell_number': [0, 1, 2],
            'experiment': ['2017-01-30', '2017-01-30', '2017-01-30'],
            'file':	[
                '/data/Experiment/2017-01-30/Corrected/EGFP/s01/EGFP-s01t001.tif',
                '/data/Experiment/2017-01-30/Corrected/EGFP/s01/EGFP-s01t002.tif',
                '/data/Experiment/2017-01-30/Corrected/EGFP/s01/EGFP-s01t003.tif',
            ],
            'flip': ['none', 'none', 'none'],
            'next_cell': [1, 2, np.nan],
            'prev_cell': [np.nan, 0, 1],
            'rot90': [0, 0, 0],
            'tile': [1, 1, 1],
            'timepoint': [1, 2, 3],
        })
        df.to_excel(self.tempdir / 'index.xlsx')

        res = stat_utils.load_index(self.tempdir / 'index.xlsx')
        exp = {
            ('2017-01-30', 1, 1, 0, 'none'): 0,
            ('2017-01-30', 1, 2, 0, 'none'): 1,
            ('2017-01-30', 1, 3, 0, 'none'): 2,
        }

        self.assertEqual(res, exp)

    def test_sane_with_rots_flips(self):

        df = pd.DataFrame({
            'cell_number': [0, 1, 2],
            'experiment': ['2017-01-30', '2017-01-30', '2017-01-30'],
            'file':	[
                '/data/Experiment/2017-01-30/Corrected/EGFP/s01/EGFP-s01t001.tif',
                '/data/Experiment/2017-01-30/Corrected/EGFP/s01/EGFP-s01t002.tif',
                '/data/Experiment/2017-01-30/Corrected/EGFP/s01/EGFP-s01t003.tif',
            ],
            'flip': ['none', 'none', 'horizontal'],
            'next_cell': [np.nan, np.nan, np.nan],
            'prev_cell': [np.nan, np.nan, np.nan],
            'rot90': [0, 1, 1],
            'tile': [1, 1, 1],
            'timepoint': [1, 1, 1],
        })
        df.to_excel(self.tempdir / 'index.xlsx')

        res = stat_utils.load_index(self.tempdir / 'index.xlsx')
        exp = {
            ('2017-01-30', 1, 1, 0, 'none'): 0,
            ('2017-01-30', 1, 1, 1, 'none'): 1,
            ('2017-01-30', 1, 1, 1, 'horizontal'): 2,
        }

        self.assertEqual(res, exp)


class TestLoadPointsFromMaskfile(FileSystemTestCase):

    def test_loads_basic_mask(self):

        maskfile = self.tempdir / 'foo.png'

        img = np.zeros((64, 65))
        img[2, 14] = 1
        img[35, 55] = 1

        assert not maskfile.is_file()

        save_image(maskfile, img)

        assert maskfile.is_file()

        res_x, res_y, res_v = stat_utils.load_points_from_maskfile(maskfile)
        exp_x = np.array([14, 55])/65
        exp_y = 1.0 - np.array([2, 35])/64
        exp_v = np.array([1, 1])

        np.testing.assert_almost_equal(res_x, exp_x, decimal=2)
        np.testing.assert_almost_equal(res_y, exp_y, decimal=2)
        np.testing.assert_almost_equal(res_v, exp_v, decimal=2)

    def test_loads_mask_different_levels_cutoff(self):

        maskfile = self.tempdir / 'foo.png'

        img = np.zeros((64, 65))
        img[2, 14] = 1
        img[35, 55] = 0.75
        img[2, 55] = 0.49

        assert not maskfile.is_file()

        save_image(maskfile, img)

        assert maskfile.is_file()

        res_x, res_y, res_v = stat_utils.load_points_from_maskfile(maskfile)
        exp_x = np.array([14, 55])/65
        exp_y = 1.0 - np.array([2, 35])/64
        exp_v = np.array([1, 0.75])

        np.testing.assert_almost_equal(res_x, exp_x, decimal=2)
        np.testing.assert_almost_equal(res_y, exp_y, decimal=2)
        np.testing.assert_almost_equal(res_v, exp_v, decimal=2)

    def test_loads_mask_scaled_different_levels_cutoff(self):

        maskfile = self.tempdir / 'foo.png'

        img = np.zeros((64, 65))
        img[2, 14] = 1
        img[35, 55] = 0.75
        img[2, 55] = 0.49

        assert not maskfile.is_file()

        save_image(maskfile, img)

        assert maskfile.is_file()

        res_x, res_y, res_v = stat_utils.load_points_from_maskfile(maskfile, keep_scale=True)
        exp_x = np.array([14, 55])
        exp_y = np.array([2, 35])
        exp_v = np.array([1, 0.75])

        np.testing.assert_almost_equal(res_x, exp_x, decimal=2)
        np.testing.assert_almost_equal(res_y, exp_y, decimal=2)
        np.testing.assert_almost_equal(res_v, exp_v, decimal=2)


class TestScorePoints(unittest.TestCase):

    def test_score_same_points_max(self):

        train_x = np.array([0.1, 0.2, 0.3, 0.4])
        train_y = np.array([1.1, 1.2, 1.3, 1.4])

        test_x = np.array([0.1, 0.2, 0.3, 0.4])
        test_y = np.array([1.1, 1.2, 1.3, 1.4])
        test_v = np.array([1.0, 1.0, 1.0, 1.0])

        exp_real = np.array([True, True, True, True], dtype=np.bool)
        exp_score = np.array([1.0, 1.0, 1.0, 1.0])

        res = stat_utils.score_points(
            (train_x, train_y),
            (test_x, test_y, test_v))

        np.testing.assert_almost_equal(exp_real, res[0])
        np.testing.assert_almost_equal(exp_score, res[1])

    def test_scores_same_points_2d_vectors_max(self):

        train_x = np.array([0.1, 0.2, 0.3])
        train_y = np.array([1.1, 1.2, 1.3])

        test_x = np.array([0.1, 0.2, 0.3])
        test_y = np.array([1.1, 1.2, 1.3])
        test_v = np.array([1.0, 1.0, 1.0])

        exp_real = np.array([True, True, True], dtype=np.bool)
        exp_score = np.array([1.0, 1.0, 1.0])

        res = stat_utils.score_points(
            np.stack((train_x, train_y), axis=1),
            np.stack((test_x, test_y, test_v), axis=1))

        np.testing.assert_almost_equal(exp_real, res[0])
        np.testing.assert_almost_equal(exp_score, res[1])

    def test_scores_superset_onto(self):

        train_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        train_y = np.array([1.1, 1.2, 1.3, 1.4, 1.5])

        test_x = np.array([0.1, 0.2, 0.3])
        test_y = np.array([1.1, 1.2, 1.3])
        test_v = np.array([1.0, 1.0, 1.0])

        exp_real = np.array([True, True, True, True, True], dtype=np.bool)
        exp_score = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

        res = stat_utils.score_points(
            np.stack((train_x, train_y), axis=1),
            np.stack((test_x, test_y, test_v), axis=1))

        np.testing.assert_almost_equal(exp_real, res[0])
        np.testing.assert_almost_equal(exp_score, res[1])

    def test_scores_subset_onto(self):

        train_x = np.array([0.1, 0.2, 0.3])
        train_y = np.array([1.1, 1.2, 1.3])

        test_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        test_y = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
        test_v = np.array([1.0, 1.0, 1.0, 0.9, 0.8])

        exp_real = np.array([True, True, True, False, False], dtype=np.bool)
        exp_score = np.array([1.0, 1.0, 1.0, 0.9, 0.8])

        res = stat_utils.score_points(
            np.stack((train_x, train_y), axis=1),
            np.stack((test_x, test_y, test_v), axis=1))

        np.testing.assert_almost_equal(exp_real, res[0])
        np.testing.assert_almost_equal(exp_score, res[1])

    def test_scores_subset_by_nearest_neighbor(self):

        train_x = np.array([0.11, 0.22, 0.33])
        train_y = np.array([1.11, 1.22, 1.33])

        test_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        test_y = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
        test_v = np.array([1.0, 1.0, 1.0, 0.9, 0.8])

        exp_real = np.array([True, True, True, False, False], dtype=np.bool)
        exp_score = np.array([1.0, 1.0, 1.0, 0.9, 0.8])

        res = stat_utils.score_points(
            np.stack((train_x, train_y), axis=1),
            np.stack((test_x, test_y, test_v), axis=1))

        np.testing.assert_almost_equal(exp_real, res[0])
        np.testing.assert_almost_equal(exp_score, res[1])

    def test_scores_subset_by_nearest_neighbor_max_distance(self):

        train_x = np.array([0.11, 0.22, 0.33])
        train_y = np.array([1.11, 1.22, 1.33])

        test_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        test_y = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
        test_v = np.array([1.0, 1.1, 1.2, 0.9, 0.8])

        exp_real = np.array([True, True, True, False, False, False, False], dtype=np.bool)
        exp_score = np.array([1.0, 0.0, 0.0, 1.1, 1.2, 0.9, 0.8])

        res = stat_utils.score_points(
            np.stack((train_x, train_y), axis=1),
            np.stack((test_x, test_y, test_v), axis=1),
            max_distance=0.015)

        np.testing.assert_almost_equal(exp_real, res[0])
        np.testing.assert_almost_equal(exp_score, res[1])

    def test_calc_irr_from_point_scores_subset(self):

        train_x = np.array([0.11, 0.22, 0.33])
        train_y = np.array([1.11, 1.22, 1.33])

        test_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        test_y = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
        test_v = np.array([1.0, 1.1, 1.2, 0.9, 0.8])

        res_matches, res_total = stat_utils.score_points(
            np.stack((train_x, train_y), axis=1),
            np.stack((test_x, test_y, test_v), axis=1),
            max_distance=0.015,
            return_irr=True)
        self.assertEqual(res_matches, 1/7)
        self.assertEqual(res_total, 7)

    def test_calc_irr_from_point_scores_perfect(self):

        train_x = np.array([0.1, 0.2, 0.3, 0.4])
        train_y = np.array([1.1, 1.2, 1.3, 1.4])

        test_x = np.array([0.1, 0.2, 0.3, 0.4])
        test_y = np.array([1.1, 1.2, 1.3, 1.4])
        test_v = np.array([1.0, 1.0, 1.0, 1.0])

        res_matches, res_total = stat_utils.score_points(
            np.stack((train_x, train_y), axis=1),
            np.stack((test_x, test_y, test_v), axis=1),
            max_distance=0.015,
            return_irr=True)
        self.assertEqual(res_matches, 1.0)
        self.assertEqual(res_total, 4)

    def test_calc_irr_from_point_scores_superset(self):

        train_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        train_y = np.array([1.1, 1.2, 1.3, 1.4, 1.5])

        test_x = np.array([0.1, 0.2, 0.3])
        test_y = np.array([1.1, 1.2, 1.3])
        test_v = np.array([1.0, 1.0, 1.0])

        res_matches, res_total = stat_utils.score_points(
            np.stack((train_x, train_y), axis=1),
            np.stack((test_x, test_y, test_v), axis=1),
            return_irr=True)
        self.assertEqual(res_matches, 3/5)
        self.assertEqual(res_total, 5)

    def test_sane_with_perfect_match_no_scores(self):

        train_points = np.array([
            [1, 2.1],
            [2, 2.2],
            [3, 2.3],
            [4, 2.4],
            [5, 2.5],
        ])
        test_points = train_points.copy()

        frac_match, num_total = stat_utils.score_points(
            train_points, test_points, return_irr=True)

        self.assertEqual(frac_match, 1.0)
        self.assertEqual(num_total, 5)

    def test_sane_with_perfect_match_and_scores(self):

        train_points = np.array([
            [1, 2.1],
            [2, 2.2],
            [3, 2.3],
            [4, 2.4],
            [5, 2.5],
        ])
        test_points = np.array([
            [1, 2.1, 0.1],
            [2, 2.2, 0.2],
            [3, 2.3, 0.3],
            [4, 2.4, 0.4],
            [5, 2.5, 0.3],
        ])

        frac_match, num_total = stat_utils.score_points(
            train_points, test_points, return_irr=True)

        self.assertEqual(frac_match, 1.0)
        self.assertEqual(num_total, 5)
