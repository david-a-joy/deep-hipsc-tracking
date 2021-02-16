#!/usr/bin/env python3

""" Tests for the path database """

# Standard lib
import pathlib
import unittest

# Our own imports
from deep_hipsc_tracking import utils
from .. import helpers

# Tests


class TestIsNonemptyDir(helpers.FileSystemTestCase):

    def test_returns_false_for_empty_dirs(self):

        p1 = self.tempdir / 'foo'
        p1.mkdir()

        (p1 / '.hidden').touch()

        res = utils.is_nonempty_dir(p1)

        self.assertFalse(res)

    def test_returns_true_for_dirs_with_things(self):

        p1 = self.tempdir / 'foo'
        p1.mkdir()

        (p1 / 'not_hidden').touch()

        res = utils.is_nonempty_dir(p1)

        self.assertTrue(res)


class TestParseTrainingDir(unittest.TestCase):

    def test_parses_basedir_no_run(self):

        path = pathlib.Path('ai-upsample-peaks-composite-d3-opt')

        res = utils.parse_training_dir(path)

        exp = {'detector': 'composite-d3-opt',
               'run': None,
               'num_iters': None,
               'training_set': 'peaks'}
        self.assertEqual(res, exp)

    def test_parses_basedir_with_run(self):

        path = pathlib.Path('ai-upsample-peaks-fcrn_a_wide-run003')

        res = utils.parse_training_dir(path)

        exp = {'detector': 'fcrn_a_wide',
               'run': 3,
               'num_iters': None,
               'training_set': 'peaks'}
        self.assertEqual(res, exp)

    def test_parses_iterdir_with_run(self):

        path = pathlib.Path('ai-upsample-peaks-fcrn_a_wide-run003/ai-upsample-peaks-n75000')

        res = utils.parse_training_dir(path)

        exp = {'detector': 'fcrn_a_wide',
               'run': 3,
               'num_iters': 75000,
               'training_set': 'peaks'}
        self.assertEqual(res, exp)

    def test_parses_iterdir_with_run_different_set(self):

        path = pathlib.Path('ai-upsample-confocal-fcrn_a_wide-run003/ai-upsample-confocal-n75000')

        res = utils.parse_training_dir(path)

        exp = {'detector': 'fcrn_a_wide',
               'run': 3,
               'num_iters': 75000,
               'training_set': 'confocal'}
        self.assertEqual(res, exp)

    def test_errors_on_iterdir_rundir_mismatch(self):

        path = pathlib.Path('ai-upsample-confocal-fcrn_a_wide-run003/ai-upsample-peaks-n75000')

        with self.assertRaises(KeyError):
            utils.parse_training_dir(path)


class TestGetRootdir(helpers.FileSystemTestCase):

    def test_no_rootdir(self):
        inpath = self.tempdir / 'foo/bar'
        inpath.mkdir(parents=True)

        res = utils.get_rootdir(inpath)

        self.assertIsNone(res)

    def test_gets_rootdir_from_imagefile_example(self):
        rootdir = self.tempdir / 'foo/bar/example_confocal'
        rootdir.mkdir(parents=True)

        inpath = rootdir / 'EGFP/s04-bees/foo-s04t003.tif'

        res = utils.get_rootdir(inpath)
        self.assertEqual(res, rootdir)

    def test_gets_rootdir_from_imagefile_config_file(self):
        rootdir = self.tempdir / 'foo/bar/baz'
        rootdir.mkdir(parents=True)

        (rootdir / 'deep_tracking.ini').touch()
        inpath = rootdir / 'EGFP/s04-bees/foo-s04t003.tif'

        res = utils.get_rootdir(inpath)
        self.assertEqual(res, rootdir)

    def test_gets_rootdir_from_rootdir(self):
        inpath = self.tempdir / 'foo/bar/2017-02-12'
        inpath.mkdir(parents=True)

        res = utils.get_rootdir(inpath)
        self.assertEqual(res, inpath)

    def test_gets_rootdir_from_imagefile(self):
        rootdir = self.tempdir / 'foo/bar/2017-02-12'
        rootdir.mkdir(parents=True)

        inpath = rootdir / 'EGFP/s04-bees/foo-s04t003.tif'

        res = utils.get_rootdir(inpath)
        self.assertEqual(res, rootdir)

    def test_gets_rootdir_from_tiledir(self):
        rootdir = self.tempdir / 'foo/bar/2017-02-12'
        rootdir.mkdir(parents=True)

        inpath = rootdir / 'EGFP/s04-bees'

        res = utils.get_rootdir(inpath)
        self.assertEqual(res, rootdir)

    def test_gets_rootdir_from_channeldir(self):
        rootdir = self.tempdir / 'foo/bar/2017-02-12'
        rootdir.mkdir(parents=True)

        inpath = rootdir / 'EGFP/'

        res = utils.get_rootdir(inpath)
        self.assertEqual(res, rootdir)

    def test_ignores_weird_rootdir(self):

        rootdir = self.tempdir / 'foo/bar/2017-02-12-bees'
        rootdir.mkdir(parents=True)

        inpath = rootdir / 'EGFP/'

        res = utils.get_rootdir(inpath)
        self.assertIsNone(res)

    def test_ignores_non_existant_rootdir(self):

        rootdir = self.tempdir / 'foo/bar/2017-02-12'
        self.assertFalse(rootdir.is_dir())

        inpath = rootdir / 'EGFP/'

        res = utils.get_rootdir(inpath)
        self.assertIsNone(res)


class TestFindCommonBasedir(helpers.FileSystemTestCase):

    def test_find_basedir_no_idea(self):

        res = utils.find_common_basedir()

        self.assertIsNone(res)

    def test_find_basedir_no_rootdirs(self):

        basedir = self.tempdir / 'base'

        res = utils.find_common_basedir(basedir=basedir)

        self.assertEqual(res, basedir)

    def test_find_basedir_rootdirs_and_basedir(self):

        basedir = self.tempdir / 'base'
        r1 = basedir / '2016-10-01'
        r2 = basedir / '2017-01-23'

        res = utils.find_common_basedir(rootdirs=[r1, r2], basedir=basedir)

        self.assertEqual(res, basedir)

    def test_find_basedir_rootdirs_same_level(self):

        basedir = self.tempdir / 'base'
        r1 = basedir / '2016-10-01'
        r2 = basedir / '2017-01-23'

        res = utils.find_common_basedir(rootdirs=[r1, r2])

        self.assertEqual(res, basedir)

    def test_find_basedir_rootdirs_nested_levels(self):

        basedir = self.tempdir / 'base'
        r1 = basedir / 'p1' / '2016-10-01'
        r2 = basedir / 'p2' / '2017-01-23'

        res = utils.find_common_basedir(rootdirs=[r1, r2])

        self.assertEqual(res, basedir)

    def test_find_basedir_rootdirs_staggered_levels(self):

        basedir = self.tempdir / 'base'
        r1 = basedir / '2016-10-01'
        r2 = basedir / 'p2' / '2017-01-23'

        res = utils.find_common_basedir(rootdirs=[r1, r2])

        self.assertEqual(res, basedir)


class TestFindTimepoints(helpers.FileSystemTestCase):

    def test_no_tiledir(self):

        r1 = self.tempdir / 's03'

        with self.assertRaises(OSError):
            list(utils.find_timepoints(r1, timepoints=3, suffix='.tif'))

    def test_one_timepoint(self):

        r1 = self.tempdir / 's03'
        r1.mkdir()

        f1 = r1 / 'foo_s03t003.tif'
        f1.touch()

        res = list(utils.find_timepoints(r1, timepoints=3, suffix='.tif'))

        exp = [(3, f1)]
        self.assertEqual(res, exp)

    def test_three_timepoints(self):

        r1 = self.tempdir / 's03'
        r1.mkdir()

        f1 = r1 / 'foo_s03t003.tif'
        f1.touch()
        (r1 / 'foo_s03t005.tif').touch()
        f3 = r1 / 'foo_s03t099.tif'
        f3.touch()
        (r1 / 'foo_s03t099.npz').touch()
        f4 = r1 / 'foo_s03t100.tif'
        f4.touch()

        res = list(utils.find_timepoints(r1, timepoints=[3, 99, 100], suffix='.tif'))

        exp = [(3, f1), (99, f3), (100, f4)]
        self.assertEqual(res, exp)


class TestFindTimepoint(helpers.FileSystemTestCase):

    def test_no_tiledir(self):

        r1 = self.tempdir / 's03'

        res = utils.find_timepoint(r1, tile=3, timepoint=5)

        self.assertIsNone(res)

    def test_tiledir_empty(self):

        r1 = self.tempdir / 's03'
        r1.mkdir(parents=True)

        res = utils.find_timepoint(r1, tile=3, timepoint=5)

        self.assertIsNone(res)

    def test_tiledir_no_match(self):

        r1 = self.tempdir / 's03'
        r1.mkdir(parents=True)

        f1 = r1 / 'grr_s03t001.jpg'
        with f1.open('wt') as fp:
            fp.write('BAD')

        res = utils.find_timepoint(r1, tile=3, timepoint=5)

        self.assertIsNone(res)

    def test_tiledir_match(self):

        r1 = self.tempdir / 's03'
        r1.mkdir(parents=True)

        f1 = r1 / 'grr_s03t005.jpg'
        with f1.open('wt') as fp:
            fp.write('BAD')

        res = utils.find_timepoint(r1, tile=3, timepoint=5)

        self.assertEqual(res, f1)

    def test_tiledir_matches_first(self):

        r1 = self.tempdir / 's03'
        r1.mkdir(parents=True)

        f1 = r1 / 'a_s03t005.jpg'
        with f1.open('wt') as fp:
            fp.write('BAD')

        f2 = r1 / 'b_s03t005.jpg'
        with f2.open('wt') as fp:
            fp.write('BAD')

        res = utils.find_timepoint(r1, tile=3, timepoint=5)

        self.assertEqual(res, f1)

    def test_tiledir_matches_prefix(self):

        r1 = self.tempdir / 's03'
        r1.mkdir(parents=True)

        f1 = r1 / 'a_s03t005.jpg'
        with f1.open('wt') as fp:
            fp.write('BAD')

        f2 = r1 / 'b_s03t005.jpg'
        with f2.open('wt') as fp:
            fp.write('BAD')

        res = utils.find_timepoint(r1, tile=3, timepoint=5, prefix='b')

        self.assertEqual(res, f2)


class TestFindTiledirs(helpers.FileSystemTestCase):

    def test_finds_no_tiles(self):
        res = list(utils.find_tiledirs(self.tempdir))
        self.assertEqual(res, [])

    def test_finds_one_tile_no_condition(self):
        r1 = self.tempdir / 's03'
        r1.mkdir(parents=True)

        res = list(utils.find_tiledirs(self.tempdir))
        self.assertEqual(res, [(3, r1)])

    def test_finds_one_tile_with_condition(self):
        r1 = self.tempdir / 's03-ponies'
        r1.mkdir(parents=True)

        res = list(utils.find_tiledirs(self.tempdir))
        self.assertEqual(res, [(3, r1)])

    def test_finds_three_tiles(self):
        r5 = self.tempdir / 's05'
        r5.mkdir(parents=True)

        r1 = self.tempdir / 's01-toast'
        r1.mkdir(parents=True)

        r2 = self.tempdir / 's02-grr'
        r2.mkdir(parents=True)

        res = list(utils.find_tiledirs(self.tempdir))
        self.assertEqual(res, [(1, r1), (2, r2), (5, r5)])

    def test_ignores_files(self):
        r5 = self.tempdir / 's05'
        r5.mkdir(parents=True)

        r1 = self.tempdir / 's01-toast'
        r1.mkdir(parents=True)

        r2 = self.tempdir / 's02-grr'
        r2.mkdir(parents=True)

        bad = self.tempdir / 's04-bad'
        with bad.open('wt') as fp:
            fp.write('bad')

        res = list(utils.find_tiledirs(self.tempdir))
        self.assertEqual(res, [(1, r1), (2, r2), (5, r5)])

    def test_ignores_unparsable_dirs(self):
        r5 = self.tempdir / 's05'
        r5.mkdir(parents=True)

        r1 = self.tempdir / 's01-toast'
        r1.mkdir(parents=True)

        r2 = self.tempdir / 's02-grr'
        r2.mkdir(parents=True)

        bad = self.tempdir / 'agkjslakdhjfld'
        bad.mkdir(parents=True)

        res = list(utils.find_tiledirs(self.tempdir))
        self.assertEqual(res, [(1, r1), (2, r2), (5, r5)])

    def test_works_with_str_basedir(self):
        r5 = self.tempdir / 's05'
        r5.mkdir(parents=True)

        r1 = self.tempdir / 's01-toast'
        r1.mkdir(parents=True)

        r2 = self.tempdir / 's02-grr'
        r2.mkdir(parents=True)

        res = list(utils.find_tiledirs(str(self.tempdir)))
        self.assertEqual(res, [(1, r1), (2, r2), (5, r5)])

    def test_can_filter_on_tile_numbers(self):

        r5 = self.tempdir / 's05'
        r5.mkdir(parents=True)

        r1 = self.tempdir / 's01-toast'
        r1.mkdir(parents=True)

        r2 = self.tempdir / 's02-grr'
        r2.mkdir(parents=True)

        res = list(utils.find_tiledirs(self.tempdir, tiles=[2, 5]))
        self.assertEqual(res, [(2, r2), (5, r5)])

    def test_can_filter_on_tile_number_strings(self):

        r5 = self.tempdir / 's05'
        r5.mkdir(parents=True)

        r1 = self.tempdir / 's01-toast'
        r1.mkdir(parents=True)

        r2 = self.tempdir / 's02-grr'
        r2.mkdir(parents=True)

        res = list(utils.find_tiledirs(self.tempdir, tiles=['2', 5]))
        self.assertEqual(res, [(2, r2), (5, r5)])

    def test_can_filter_on_single_tile_number(self):

        r5 = self.tempdir / 's05'
        r5.mkdir(parents=True)

        r1 = self.tempdir / 's01-toast'
        r1.mkdir(parents=True)

        r2 = self.tempdir / 's02-grr'
        r2.mkdir(parents=True)

        res = list(utils.find_tiledirs(self.tempdir, tiles=5))
        self.assertEqual(res, [(5, r5)])

    def test_can_filter_on_single_tile_number_string(self):

        r5 = self.tempdir / 's05'
        r5.mkdir(parents=True)

        r1 = self.tempdir / 's01-toast'
        r1.mkdir(parents=True)

        r2 = self.tempdir / 's02-grr'
        r2.mkdir(parents=True)

        res = list(utils.find_tiledirs(self.tempdir, tiles='2'))
        self.assertEqual(res, [(2, r2)])

    def test_can_filter_on_single_condition(self):

        r5 = self.tempdir / 's05-h7-1'
        r5.mkdir(parents=True)

        r1 = self.tempdir / 's01-h7-2'
        r1.mkdir(parents=True)

        r2 = self.tempdir / 's02-wtc-2'
        r2.mkdir(parents=True)

        res = list(utils.find_tiledirs(self.tempdir, conditions=['h7']))
        self.assertEqual(res, [(1, r1), (5, r5)])

    def test_can_filter_on_multiple_conditions(self):

        r5 = self.tempdir / 's05-h7-1'
        r5.mkdir(parents=True)

        r1 = self.tempdir / 's01-wtb-2'
        r1.mkdir(parents=True)

        r2 = self.tempdir / 's02-wtc-2'
        r2.mkdir(parents=True)

        res = list(utils.find_tiledirs(self.tempdir, conditions=['wtb', 'h7']))
        self.assertEqual(res, [(1, r1), (5, r5)])

    def test_condition_filter_case_insensitive(self):

        r5 = self.tempdir / 's05-H7-1'
        r5.mkdir(parents=True)

        r1 = self.tempdir / 's01-wTb-2'
        r1.mkdir(parents=True)

        r2 = self.tempdir / 's02-wtc-2'
        r2.mkdir(parents=True)

        res = list(utils.find_tiledirs(self.tempdir, conditions=['WTB', 'h7']))
        self.assertEqual(res, [(1, r1), (5, r5)])


class TestPairTiledirs(helpers.FileSystemTestCase):

    def test_pair_zero_dirs(self):

        res = utils.pair_tiledirs()
        exp = []
        self.assertEqual(res, exp)

    def test_pair_one_dir(self):

        r1 = self.tempdir / 'bees'
        t1_1 = r1 / 's01'
        t1_1.mkdir(parents=True)

        res = utils.pair_tiledirs(r1)
        exp = [(t1_1, )]
        self.assertEqual(res, exp)

    def test_pair_two_dirs_one_match(self):

        r1 = self.tempdir / 'bees'
        r2 = self.tempdir / 'toast'

        t1_1 = r1 / 's01'
        t1_1.mkdir(parents=True)

        t2_1 = r2 / 's01-bees'
        t2_1.mkdir(parents=True)

        res = utils.pair_tiledirs(r1, r2)
        exp = [(t1_1, t2_1)]
        self.assertEqual(res, exp)

    def test_pair_two_dirs_two_matches(self):

        r1 = self.tempdir / 'bees'
        r2 = self.tempdir / 'toast'

        t1_1 = r1 / 's01'
        t1_1.mkdir(parents=True)

        t1_2 = r1 / 's02-grr'
        t1_2.mkdir(parents=True)

        t2_1 = r2 / 's01-bees'
        t2_1.mkdir(parents=True)

        t2_2 = r2 / 's02'
        t2_2.mkdir(parents=True)

        res = utils.pair_tiledirs(r1, r2)
        exp = [(t1_1, t2_1), (t1_2, t2_2)]
        self.assertEqual(res, exp)

    def test_pair_three_dirs_two_matches(self):

        r1 = self.tempdir / 'bees'
        r2 = self.tempdir / 'toast'
        r3 = self.tempdir / 'buzz'

        t1_1 = r1 / 's01'
        t1_1.mkdir(parents=True)

        t1_2 = r1 / 's02-grr'
        t1_2.mkdir(parents=True)

        t2_1 = r2 / 's01-bees'
        t2_1.mkdir(parents=True)

        t2_2 = r2 / 's02'
        t2_2.mkdir(parents=True)

        t3_1 = r3 / 's01-bees'
        t3_1.mkdir(parents=True)

        t3_2 = r3 / 's02-grr'
        t3_2.mkdir(parents=True)

        res = utils.pair_tiledirs(r1, r2, r3)
        exp = [(t1_1, t2_1, t3_1), (t1_2, t2_2, t3_2)]
        self.assertEqual(res, exp)

    def test_pair_three_dirs_missing_tiles(self):

        r1 = self.tempdir / 'bees'
        r2 = self.tempdir / 'toast'
        r3 = self.tempdir / 'buzz'

        t1_1 = r1 / 's01'
        t1_1.mkdir(parents=True)

        t2_1 = r2 / 's01-bees'
        t2_1.mkdir(parents=True)

        t2_2 = r2 / 's02'
        t2_2.mkdir(parents=True)

        t3_2 = r3 / 's02-grr'
        t3_2.mkdir(parents=True)

        with self.assertRaises(OSError):
            utils.pair_tiledirs(r1, r2, r3)

    def test_pair_three_dirs_can_allow_unpaired(self):

        r1 = self.tempdir / 'bees'
        r2 = self.tempdir / 'toast'
        r3 = self.tempdir / 'buzz'

        t1_1 = r1 / 's01'
        t1_1.mkdir(parents=True)

        t2_1 = r2 / 's01-bees'
        t2_1.mkdir(parents=True)

        t2_2 = r2 / 's02'
        t2_2.mkdir(parents=True)

        t3_1 = r3 / 's01-grr'
        t3_1.mkdir(parents=True)

        res = utils.pair_tiledirs(r1, r2, r3, check_pairing=False)
        exp = [(t1_1, t2_1, t3_1)]

        self.assertEqual(res, exp)

    def test_pair_two_dirs_missing_tile(self):

        r1 = self.tempdir / 'bees'
        r2 = self.tempdir / 'toast'

        t1_1 = r1 / 's01'
        t1_1.mkdir(parents=True)

        t2_1 = r2 / 's01-bees'
        t2_1.mkdir(parents=True)

        t2_2 = r2 / 's02'
        t2_2.mkdir(parents=True)

        with self.assertRaises(OSError):
            utils.pair_tiledirs(r1, r2)


class TestFindExperimentDirs(helpers.FileSystemTestCase):

    def test_finds_nothing_by_default(self):

        res = utils.find_experiment_dirs()
        self.assertEqual(res, [])

    def test_finds_single_dir_pathlib(self):

        r1 = self.tempdir / 'data/Experiment/2016-01-01'
        res = utils.find_experiment_dirs(r1)

        self.assertEqual(res, [r1])

    def test_finds_single_dir_str(self):

        r1 = self.tempdir / 'data/Experiment/2016-01-01'
        res = utils.find_experiment_dirs(str(r1))

        self.assertEqual(res, [r1])

    def test_finds_multiple_dirs_str(self):

        r1 = self.tempdir / 'data/Experiment/2016-01-01'
        r2 = self.tempdir / 'data/Experiment/2016-02-01'

        res = utils.find_experiment_dirs([str(r1), str(r2)])

        self.assertEqual(res, [r1, r2])

    def test_finds_single_dir_basedir(self):

        basedir = self.tempdir / 'data/Experiment'
        r1 = basedir / '2016-01-01'
        r1.mkdir(parents=True)

        res = utils.find_experiment_dirs(basedir=basedir)

        self.assertEqual(res, [r1])

    def test_finds_multiple_dir_basedir(self):

        basedir = self.tempdir / 'data/Experiment'
        r1 = basedir / '2016-01-01'
        r1.mkdir(parents=True)

        r2 = basedir / '2016-02-03'
        r2.mkdir(parents=True)

        res = utils.find_experiment_dirs(basedir=str(basedir))

        self.assertEqual(res, [r1, r2])

    def test_ignores_files_finds_dirs_basedir(self):

        basedir = self.tempdir / 'data/Experiment'
        r1 = basedir / '2016-01-01'
        r1.parent.mkdir(parents=True)
        with r1.open('wt') as fp:
            fp.write('bees')

        r2 = basedir / '2016-02-03'
        r2.mkdir(parents=True)

        res = utils.find_experiment_dirs(basedir=str(basedir))

        self.assertEqual(res, [r2])

    def test_ignores_dirs_with_weird_names_basedir(self):

        basedir = self.tempdir / 'data/Experiment'
        r1 = basedir / 'bees'
        r1.mkdir(parents=True)

        r2 = basedir / '2016-02-03'
        r2.mkdir(parents=True)

        res = utils.find_experiment_dirs(basedir=str(basedir))

        self.assertEqual(res, [r2])

    def test_ignores_dirs_with_weird_suffices_basedir(self):

        basedir = self.tempdir / 'data/Experiment'
        r1 = basedir / '2016-02-09-bees'
        r1.mkdir(parents=True)

        r2 = basedir / '2016-02-03'
        r2.mkdir(parents=True)

        res = utils.find_experiment_dirs(basedir=str(basedir))

        self.assertEqual(res, [r2])


class TestGetOutfileName(unittest.TestCase):

    def test_flat_name(self):

        infile = pathlib.Path('TL Brightfield/s01/TL Brightfield-s01t025.tif')
        outdir = pathlib.Path('out')

        res = utils.get_outfile_name(infile, outdir, mode='flat')
        exp = outdir / 'TL Brightfield-s01t025.tif'

        self.assertEqual(res, exp)

    def test_nested_name(self):

        infile = pathlib.Path('TL Brightfield/s01/TL Brightfield-s01t025.tif')
        outdir = pathlib.Path('out')

        res = utils.get_outfile_name(infile, outdir, mode='nested')
        exp = outdir / 'TL Brightfield' / 's01' / 'TL Brightfield-s01t025.tif'

        self.assertEqual(res, exp)

    def test_nested_name_with_channel(self):

        infile = pathlib.Path('TL Brightfield/s01-foo/TL Brightfield-s01t025.tif')
        outdir = pathlib.Path('out')

        res = utils.get_outfile_name(infile, outdir, mode='nested')
        exp = outdir / 'TL Brightfield' / 's01-foo' / 'TL Brightfield-s01t025.tif'

        self.assertEqual(res, exp)

    def test_flat_name_with_new_extension(self):

        infile = pathlib.Path('TL Brightfield/s01/TL Brightfield-s01t025.tif')
        outdir = pathlib.Path('out')

        res = utils.get_outfile_name(infile, outdir, mode='flat', ext='npz')
        exp = outdir / 'TL Brightfield-s01t025.npz'

        self.assertEqual(res, exp)

    def test_nested_name_with_new_extension(self):

        infile = pathlib.Path('TL Brightfield/s01/TL Brightfield-s01t025.tif')
        outdir = pathlib.Path('out')

        res = utils.get_outfile_name(infile, outdir, mode='nested', ext='.npz')
        exp = outdir / 'TL Brightfield' / 's01' / 'TL Brightfield-s01t025.npz'

        self.assertEqual(res, exp)


class TestFindRawData(helpers.FileSystemTestCase):

    def test_empty_directory(self):

        tempdir = self.tempdir

        with self.assertRaises(OSError):
            utils.find_raw_data(tempdir)

    def test_finds_rawdata(self):

        tempdir = self.tempdir

        p1 = tempdir / 'RawData'
        p1.mkdir()

        res = utils.find_raw_data(tempdir)
        self.assertEqual(res, p1)

    def test_finds_reformat(self):

        tempdir = self.tempdir

        p1 = tempdir / 'Reformat'
        p1.mkdir()

        res = utils.find_raw_data(tempdir)
        self.assertEqual(res, p1)

    def test_prefers_reformat_to_rawdata(self):

        tempdir = self.tempdir

        p1 = tempdir / 'Reformat'
        p1.mkdir()

        p2 = tempdir / 'RawData'
        p2.mkdir()

        res = utils.find_raw_data(tempdir)
        self.assertEqual(res, p1)

    def test_angry_if_multiple_reformats(self):

        tempdir = self.tempdir

        p1 = tempdir / 'Reformat1'
        p1.mkdir()

        p2 = tempdir / 'Reformat2'
        p2.mkdir()

        with self.assertRaises(OSError):
            utils.find_raw_data(tempdir)

    def test_angry_if_multiple_rawdatas(self):

        tempdir = self.tempdir

        p1 = tempdir / 'RawData-1'
        p1.mkdir()

        p2 = tempdir / 'rawdata'
        p2.mkdir()

        with self.assertRaises(OSError):
            utils.find_raw_data(tempdir)


class TestIsSameChannel(unittest.TestCase):

    def test_two_stupid_channels(self):

        self.assertTrue(utils.is_same_channel('--foo--', 'Foo'))

        self.assertFalse(utils.is_same_channel('--foo--', '--bar--'))

        self.assertFalse(utils.is_same_channel('--foo--', 'phase'))

    def test_two_known_channels_phase(self):

        self.assertTrue(utils.is_same_channel('tl BRIGHTFIELD', 'phase'))

        self.assertTrue(utils.is_same_channel('phase', 'phase'))

        self.assertFalse(utils.is_same_channel('tl BRIGHTFIELD', 'gfp'))

        self.assertFalse(utils.is_same_channel('tl BRIGHTFIELD', 'adksfhkahs'))


class TestGuessChannelDir(helpers.FileSystemTestCase):

    def test_will_guess_stupid_names(self):

        p1 = self.tempdir / 'Ponies'

        with self.assertRaises(OSError):
            utils.guess_channel_dir(self.tempdir, 'ponies')

        p1.mkdir()

        res = utils.guess_channel_dir(self.tempdir, 'ponies')
        exp = ('Ponies', p1)

        self.assertEqual(res, exp)

    def test_guesses_dapi_aliases(self):

        for alias in ['DAPI', 'af405', 'AF350', 'Hoechst']:
            dapidir = self.tempdir / alias

            with self.assertRaises(OSError):
                utils.guess_channel_dir(self.tempdir, 'dapi')

            dapidir.mkdir()

            res = utils.guess_channel_dir(self.tempdir, 'dapi')
            exp = (alias, dapidir)

            self.assertEqual(res, exp)

            dapidir.rmdir()

    def test_guesses_gfp_aliases(self):

        for alias in ['EGFP', 'Alexa Fluor 488', 'AF488']:
            gfpdir = self.tempdir / alias

            with self.assertRaises(OSError):
                utils.guess_channel_dir(self.tempdir, 'gfp')

            gfpdir.mkdir()

            res = utils.guess_channel_dir(self.tempdir, 'gfp')
            exp = (alias, gfpdir)

            self.assertEqual(res, exp)

            gfpdir.rmdir()

    def test_guesses_rfp_aliases(self):

        for alias in ['AF647', 'alexa_fluor_647', 'DSRed']:
            gfpdir = self.tempdir / alias

            with self.assertRaises(OSError):
                utils.guess_channel_dir(self.tempdir, 'rfp')

            gfpdir.mkdir()

            res = utils.guess_channel_dir(self.tempdir, 'rfp')
            exp = (alias, gfpdir)

            self.assertEqual(res, exp)

            gfpdir.rmdir()

    def test_guesses_mkate_aliases(self):

        for alias in ['mKate', 'Alexa Fluor 555', 'Alexa Fluor 568', 'AF555', 'AF568']:
            mkatedir = self.tempdir / alias

            with self.assertRaises(OSError):
                utils.guess_channel_dir(self.tempdir, 'af555')

            mkatedir.mkdir()

            res = utils.guess_channel_dir(self.tempdir, 'af555')
            exp = (alias, mkatedir)

            self.assertEqual(res, exp)

            mkatedir.rmdir()

    def test_guesses_brightfield_aliases(self):

        for alias in ['TL Brightfield', 'Phase']:
            phasedir = self.tempdir / alias

            with self.assertRaises(OSError):
                utils.guess_channel_dir(self.tempdir, 'brightfield')

            phasedir.mkdir()

            res = utils.guess_channel_dir(self.tempdir, 'brightfield')
            exp = (alias, phasedir)

            self.assertEqual(res, exp)

            phasedir.rmdir()


class TestGroupImageFiles(helpers.FileSystemTestCase):

    def test_group_single_tile_single_channel_flat(self):

        tempdir = self.tempdir

        f1 = tempdir / 'TL Brightfield-s04t045.tif'
        f2 = tempdir / 'TL Brightfield-s04t046.tif'
        f1.touch()
        f2.touch()

        res = utils.group_image_files(tempdir, mode='flat')
        exp = utils.TileGroup()
        exp.extend([f1, f2])

        self.assertEqual(res, exp)

    def test_group_single_tile_single_channel_flat_multi(self):

        tempdir = self.tempdir

        f1 = tempdir / 'TL Brightfield-s04t045.tif'
        f2 = tempdir / 'TL Brightfield-s04t046m03.tif'
        f3 = tempdir / 'TL Brightfield-s04t046m01.tif'
        f1.touch()
        f2.touch()
        f3.touch()

        res = utils.group_image_files(tempdir, mode='flat')
        exp = utils.TileGroup()
        exp.extend([f1, f2, f3])

        self.assertEqual(res, exp)

    def test_group_single_tile_multi_channel_flat(self):

        tempdir = self.tempdir

        f1 = tempdir / 'TL Brightfield-s04t045.tif'
        f2 = tempdir / 'mCherry-s04t046.tif'
        f1.touch()
        f2.touch()

        res = utils.group_image_files(tempdir, mode='flat')
        exp = utils.TileGroup()
        exp.extend([f1, f2])

        self.assertEqual(res, exp)

    def test_group_multi_tile_multi_channel_flat(self):

        tempdir = self.tempdir

        f1 = tempdir / 'TL Brightfield-s04t045.tif'
        f2 = tempdir / 'TL Brightfield-s03t045.tif'
        f3 = tempdir / 'mCherry-s04t046.tif'
        f4 = tempdir / 'mCherry-s05t046.tif'
        f1.touch()
        f2.touch()
        f3.touch()
        f4.touch()

        res = utils.group_image_files(tempdir, mode='flat')
        exp = utils.TileGroup()
        exp.extend([f1, f2, f3, f4])

        self.assertEqual(res, exp)

    def test_group_can_select_channel_flat(self):

        tempdir = self.tempdir

        f1 = tempdir / 'TL Brightfield-s04t045.tif'
        f2 = tempdir / 'TL Brightfield-s03t045.tif'
        f3 = tempdir / 'mCherry-s04t046.tif'
        f4 = tempdir / 'mCherry-s05t046.tif'
        f1.touch()
        f2.touch()
        f3.touch()
        f4.touch()

        res = utils.group_image_files(tempdir, mode='flat', channel='brightfield')
        exp = utils.TileGroup()
        exp.extend([f1, f2])

        self.assertEqual(res, exp)

        res = utils.group_image_files(tempdir, mode='flat', channel='mcherry')
        exp = utils.TileGroup()
        exp.extend([f3, f4])

        self.assertEqual(res, exp)

    def test_group_single_tile_single_channel_nested(self):

        tempdir = self.tempdir
        d1 = tempdir / 'TL Brightfield' / 's04'
        d1.mkdir(parents=True)

        f1 = d1 / 'TL Brightfield-s04t045.tif'
        f2 = d1 / 'TL Brightfield-s04t046.tif'
        f1.touch()
        f2.touch()

        res = utils.group_image_files(tempdir, mode='nested')
        exp = utils.TileGroup()
        exp.extend([f1, f2])

        self.assertEqual(res, exp)

    def test_group_single_tile_multi_channel_nested(self):

        tempdir = self.tempdir
        d1 = tempdir / 'TL Brightfield' / 's04'
        d2 = tempdir / 'mCherry' / 's04'
        d1.mkdir(parents=True)
        d2.mkdir(parents=True)

        f1 = d1 / 'TL Brightfield-s04t045.tif'
        f2 = d2 / 'mCherry-s04t046.tif'
        f1.touch()
        f2.touch()

        res = utils.group_image_files(tempdir, mode='nested')
        exp = utils.TileGroup()
        exp.extend([f1, f2])

        self.assertEqual(res, exp)

    def test_group_multi_tile_multi_channel_nested(self):

        tempdir = self.tempdir
        d1 = tempdir / 'TL Brightfield' / 's04'
        d2 = tempdir / 'TL Brightfield' / 's03'
        d3 = tempdir / 'mCherry' / 's04'
        d4 = tempdir / 'mCherry' / 's05'
        d1.mkdir(parents=True)
        d2.mkdir(parents=True)
        d3.mkdir(parents=True)
        d4.mkdir(parents=True)

        f1 = d1 / 'TL Brightfield-s04t045.tif'
        f2 = d2 / 'TL Brightfield-s03t045.tif'
        f3 = d3 / 'mCherry-s04t046.tif'
        f4 = d4 / 'mCherry-s05t046.tif'
        f1.touch()
        f2.touch()
        f3.touch()
        f4.touch()

        res = utils.group_image_files(tempdir, mode='nested')
        exp = utils.TileGroup()
        exp.extend([f1, f2, f3, f4])

        self.assertEqual(res, exp)

    def test_group_can_select_channel_nested(self):

        tempdir = self.tempdir

        d1 = tempdir / 'TL Brightfield' / 's04'
        d2 = tempdir / 'TL Brightfield' / 's03'
        d3 = tempdir / 'mCherry' / 's04'
        d4 = tempdir / 'mCherry' / 's05'
        d1.mkdir(parents=True)
        d2.mkdir(parents=True)
        d3.mkdir(parents=True)
        d4.mkdir(parents=True)

        f1 = d1 / 'TL Brightfield-s04t045.tif'
        f2 = d2 / 'TL Brightfield-s03t045.tif'
        f3 = d3 / 'mCherry-s04t046.tif'
        f4 = d4 / 'mCherry-s05t046.tif'
        f1.touch()
        f2.touch()
        f3.touch()
        f4.touch()

        res = utils.group_image_files(tempdir, mode='nested', channel='brightfield')
        exp = utils.TileGroup()
        exp.extend([f1, f2])

        self.assertEqual(res, exp)

        res = utils.group_image_files(tempdir, mode='nested', channel='mcherry')
        exp = utils.TileGroup()
        exp.extend([f3, f4])

        self.assertEqual(res, exp)


class TestParseTileName(unittest.TestCase):

    def test_parses_tile_with_condition(self):

        name = 's20-G2-1K.2'

        res = utils.parse_tile_name(name)
        exp = {
            'tile': 20,
            'condition': 'G2-1K.2',
        }
        self.assertEqual(res, exp)

    def test_parses_tile_without_condition(self):

        name = 's04'

        res = utils.parse_tile_name(name)
        exp = {
            'tile': 4,
            'condition': None,
        }
        self.assertEqual(res, exp)


class TestParseImageName(unittest.TestCase):

    def test_parses_colony_mask_name(self):

        name = 'colony_mask-s11t021.npz'
        res = utils.parse_image_name(name)

        exp = {
            'multi_tile': 0,
            'slice': 0,
            'tile': 11,
            'channel': 1,
            'timepoint': 21,
            'channel_name': 'colony_mask',
            'key': '11-colony_mask',
            'suffix': '.npz',
        }

        self.assertEqual(res, exp)

    def test_parses_newer_experiment(self):

        name = '2016-08-29-Interleave-107_s15c3_ORG.tif'
        res = utils.parse_image_name(name)

        exp = {
            'multi_tile': 0,
            'slice': 0,
            'tile': 15,
            'channel': 3,
            'timepoint': 107,
            'channel_name': 'EGFP',
            'key': '15-egfp',
            'prefix': '2016-08-29-Interleave',
            'suffix': '.tif',
        }

        self.assertEqual(res, exp)

    def test_parses_multitile_experiment(self):

        name = 'Alexa Fluor 488-s011t045m3.tif'
        res = utils.parse_image_name(name)

        exp = {
            'slice': 0,
            'tile': 11,
            'channel': 1,
            'timepoint': 45,
            'channel_name': 'Alexa Fluor 488',
            'multi_tile': 3,
            'key': '11-alexa_fluor_488',
            'suffix': '.tif',
        }

        self.assertEqual(res, exp)

    def test_parses_another_new_experiment(self):

        name = '2016-10-28-6hr-Image Export-01_s33t68_ORG.tif'

        res = utils.parse_image_name(name)

        exp = {
            'multi_tile': 0,
            'slice': 0,
            'tile': 33,
            'channel': 1,
            'timepoint': 68,
            'channel_name': 'TL Brightfield',
            'key': '33-tl_brightfield',
            'prefix': '2016-10-28-6hr-Image Export-01',
            'suffix': '.tif',
        }

        self.assertEqual(res, exp)

    def test_parses_yet_another_new_experiment(self):

        name = '2017-01-31-6hr-Image Export-01_s28t60_EGFP_ORG.tif'

        res = utils.parse_image_name(name)

        exp = {
            'multi_tile': 0,
            'slice': 0,
            'tile': 28,
            'channel': 3,
            'timepoint': 60,
            'channel_name': 'EGFP',
            'key': '28-egfp',
            'prefix': '2017-01-31-6hr-Image Export-01',
            'suffix': '.tif',
        }

        self.assertEqual(res, exp)

    def test_parses_ants_affine(self):

        name = 'Experiment-07-Image Export-01_s32t53TL Brightfield_1_ORGAffine.txt'
        res = utils.parse_image_name(name)

        exp = {
            'multi_tile': 0,
            'slice': 0,
            'tile': 32,
            'channel': 1,
            'timepoint': 53,
            'channel_name': 'TL Brightfield',
            'key': '32-tl_brightfield',
            'prefix': 'Experiment-07-Image Export-01',
            'suffix': 'Affine.txt',
        }

        self.assertEqual(res, exp)

    def test_parses_ants_inv_warp(self):

        name = 'Experiment-07-Image Export-01_s32t53TL Brightfield_1_ORGInverseWarp.nii.gz'
        res = utils.parse_image_name(name)

        exp = {
            'multi_tile': 0,
            'slice': 0,
            'tile': 32,
            'channel': 1,
            'timepoint': 53,
            'channel_name': 'TL Brightfield',
            'key': '32-tl_brightfield',
            'prefix': 'Experiment-07-Image Export-01',
            'suffix': 'InverseWarp.nii.gz',
        }

        self.assertEqual(res, exp)

    def test_parses_ants_warp(self):

        name = 'Experiment-07-Image Export-01_s32t53TL Brightfield_1_ORGWarp.nii.gz'
        res = utils.parse_image_name(name)

        exp = {
            'multi_tile': 0,
            'slice': 0,
            'tile': 32,
            'channel': 1,
            'timepoint': 53,
            'channel_name': 'TL Brightfield',
            'key': '32-tl_brightfield',
            'prefix': 'Experiment-07-Image Export-01',
            'suffix': 'Warp.nii.gz',
        }

        self.assertEqual(res, exp)

    def test_parses_relabeled_warp_file(self):

        name = 'TL Brightfield-s04t045InverseWarp.nii.gz'
        res = utils.parse_image_name(name)

        exp = {
            'multi_tile': 0,
            'slice': 0,
            'tile': 4,
            'channel': 1,
            'timepoint': 45,
            'channel_name': 'TL Brightfield',
            'key': '4-tl_brightfield',
            'suffix': 'InverseWarp.nii.gz',
        }

        self.assertEqual(res, exp)

    def test_parses_relabeled_tiff(self):

        name = 'mCherry-s04t045.tif'
        res = utils.parse_image_name(name)

        exp = {
            'multi_tile': 0,
            'slice': 0,
            'tile': 4,
            'channel': 2,
            'timepoint': 45,
            'channel_name': 'mCherry',
            'key': '4-mcherry',
            'suffix': '.tif',
        }

        self.assertEqual(res, exp)

    def test_parses_older_experiment(self):

        name = 'Experiment-07-Image Export-01_s13t57TL Brightfield_1_ORG.tif'
        res = utils.parse_image_name(name)

        exp = {
            'multi_tile': 0,
            'slice': 0,
            'tile': 13,
            'channel': 1,
            'timepoint': 57,
            'channel_name': 'TL Brightfield',
            'key': '13-tl_brightfield',
            'suffix': '.tif',
            'prefix': 'Experiment-07-Image Export-01'
        }

        self.assertEqual(res, exp)
