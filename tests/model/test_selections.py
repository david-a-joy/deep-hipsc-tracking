""" Tests for the region selection database """

import unittest

import numpy as np

from deep_hipsc_tracking.model import load_selection_db

# Tests


class TestSelectionDB(unittest.TestCase):

    def test_can_find_on_empty_filepath(self):

        db = load_selection_db('sqlite:///:memory:')

        self.assertFalse(db.has_selections(None))
        self.assertEqual(list(db.find_selections(None)), [])

    def test_add_find_single_region_default_class(self):

        db = load_selection_db('sqlite:///:memory:')

        self.assertFalse(db.has_selections('foo/bar/baz.tif'))

        db.add_selection('foo/bar/baz.tif',
                         x0=0.0, x1=1.0, y0=0.1, y1=0.9)

        self.assertTrue(db.has_selections('foo/bar/baz.tif'))

        res = list(db.find_selections('foo/bar/baz.tif'))

        self.assertEqual(len(res), 1)

        self.assertEqual(res[0].sel_class, 1)
        self.assertEqual(res[0].x0, 0.0)
        self.assertEqual(res[0].y0, 0.1)
        self.assertEqual(res[0].x1, 1.0)
        self.assertEqual(res[0].y1, 0.9)

    def test_add_find_single_point_default_class(self):

        db = load_selection_db('sqlite:///:memory:')

        self.assertFalse(db.has_selections('foo/bar.baz.tif'))
        self.assertFalse(db.has_points('foo/bar/baz.tif'))

        db.add_point('foo/bar/baz.tif', x=0.5, y=0.4)

        self.assertFalse(db.has_selections('foo/bar/baz.tif'))
        self.assertTrue(db.has_points('foo/bar/baz.tif'))

        res = list(db.find_points('foo/bar/baz.tif'))

        self.assertEqual(len(res), 1)

        self.assertEqual(res[0].sel_class, 1)
        self.assertEqual(res[0].x, 0.5)
        self.assertEqual(res[0].y, 0.4)

    def test_add_find_single_region_specific_class(self):

        db = load_selection_db('sqlite:///:memory:')

        self.assertFalse(db.has_selections('foo/bar/baz.tif'))
        self.assertEqual(list(db.find_selections('foo/bar/baz.tif')),
                         [])

        db.add_selection('foo/bar/baz.tif',
                         sel_class=2, x0=0.0, x1=1.0, y0=0.1, y1=0.9)

        self.assertTrue(db.has_selections('foo/bar/baz.tif'))

        res = list(db.find_selections('foo/bar/baz.tif'))

        self.assertEqual(len(res), 1)

        self.assertEqual(res[0].sel_class, 2)
        self.assertEqual(res[0].x0, 0.0)
        self.assertEqual(res[0].y0, 0.1)
        self.assertEqual(res[0].x1, 1.0)
        self.assertEqual(res[0].y1, 0.9)

    def test_add_duplicate_selection(self):

        db = load_selection_db('sqlite:///:memory:')

        db.add_selection('foo/bar/baz.tif',
                         sel_class=2, x0=0.0, x1=1.0, y0=0.1, y1=0.9)

        db.add_selection('foo/bar/baz.tif',
                         sel_class=2, x0=0.0, x1=1.0, y0=0.1, y1=0.9)

        self.assertTrue(db.has_selections('foo/bar/baz.tif'))

        res = list(db.find_selections('foo/bar/baz.tif'))

        self.assertEqual(len(res), 1)

        self.assertEqual(res[0].sel_class, 2)
        self.assertEqual(res[0].x0, 0.0)
        self.assertEqual(res[0].y0, 0.1)
        self.assertEqual(res[0].x1, 1.0)
        self.assertEqual(res[0].y1, 0.9)

    def test_add_several_image_selections(self):

        db = load_selection_db('sqlite:///:memory:')

        db.add_selection('foo/bar/baz.tif',
                         sel_class=2, x0=0.0, x1=1.0, y0=0.1, y1=0.9)

        db.add_selection('grr/arg/baz.tif',
                         sel_class=2, x0=0.0, x1=1.0, y0=0.1, y1=0.9)

        self.assertTrue(db.has_selections('foo/bar/baz.tif'))

        res = list(db.find_selections('foo/bar/baz.tif'))

        self.assertEqual(len(res), 1)

        self.assertEqual(res[0].sel_class, 2)
        self.assertEqual(res[0].x0, 0.0)
        self.assertEqual(res[0].y0, 0.1)
        self.assertEqual(res[0].x1, 1.0)
        self.assertEqual(res[0].y1, 0.9)

        self.assertTrue(db.has_selections('grr/arg/baz.tif'))

        res = list(db.find_selections('grr/arg/baz.tif'))

        self.assertEqual(len(res), 1)

        self.assertEqual(res[0].sel_class, 2)
        self.assertEqual(res[0].x0, 0.0)
        self.assertEqual(res[0].y0, 0.1)
        self.assertEqual(res[0].x1, 1.0)
        self.assertEqual(res[0].y1, 0.9)

    def test_add_zero_size_selection(self):

        db = load_selection_db('sqlite:///:memory:')

        db.add_selection('foo/bar/baz.tif',
                         sel_class=2, x0=0.0, x1=0.0, y0=0.1, y1=0.9)

        db.add_selection('foo/bar/baz.tif',
                         sel_class=2, x0=0.0, x1=0.0, y0=0.4, y1=0.4)

        self.assertFalse(db.has_selections('foo/bar/baz.tif'))

        res = list(db.find_selections('foo/bar/baz.tif'))

        self.assertEqual(len(res), 0)

    def test_add_find_several_selections(self):

        db = load_selection_db('sqlite:///:memory:')

        self.assertFalse(db.has_selections('foo/bar/baz.tif'))
        self.assertEqual(list(db.find_selections('foo/bar/baz.tif')),
                         [])

        db.add_selections('foo/bar/baz.tif',
                          classes=[1, 2, 1, 1],
                          selections=[(0.0, 1.0, 0.1, 0.9),
                                      (0.1, 0.9, 0.0, 1.0),
                                      (0.2, 0.3, 0.6, 0.7),
                                      (0.6, 0.3, 0.2, 0.7)])

        self.assertTrue(db.has_selections('foo/bar/baz.tif'))

        res = list(db.find_selections('foo/bar/baz.tif'))

        self.assertEqual(len(res), 3)

        self.assertEqual(res[0].sel_class, 1)
        self.assertEqual(res[0].x0, 0.0)
        self.assertEqual(res[0].y0, 0.9)
        self.assertEqual(res[0].x1, 0.1)
        self.assertEqual(res[0].y1, 1.0)

        self.assertEqual(res[1].sel_class, 2)
        self.assertEqual(res[1].x0, 0.0)
        self.assertEqual(res[1].y0, 0.9)
        self.assertEqual(res[1].x1, 0.1)
        self.assertEqual(res[1].y1, 1.0)

        self.assertEqual(res[2].sel_class, 1)
        self.assertEqual(res[2].x0, 0.2)
        self.assertEqual(res[2].y0, 0.3)
        self.assertEqual(res[2].x1, 0.6)
        self.assertEqual(res[2].y1, 0.7)

    def test_add_find_set_several_selections(self):

        db = load_selection_db('sqlite:///:memory:')

        self.assertFalse(db.has_selections('foo/bar/baz.tif'))
        self.assertEqual(list(db.find_selections('foo/bar/baz.tif')),
                         [])

        db.add_selections('foo/bar/baz.tif',
                          classes=[1, 2, 1, 1],
                          selections=[(0.0, 1.0, 0.1, 0.9),
                                      (0.1, 0.9, 0.0, 1.0),
                                      (0.2, 0.3, 0.6, 0.7),
                                      (0.6, 0.3, 0.2, 0.7)])

        db.set_selections('foo/bar/baz.tif',
                          classes=[1, 2, 3, 3],
                          selections=[(0.0, 1.0, 0.1, 0.9),
                                      (0.1, 0.2, 0.3, 0.4),
                                      (0.2, 0.3, 0.6, 0.7),
                                      (0.6, 0.3, 0.2, 0.7)])

        self.assertTrue(db.has_selections('foo/bar/baz.tif'))

        res = list(db.find_selections('foo/bar/baz.tif'))

        self.assertEqual(len(res), 3)

        self.assertEqual(res[0].sel_class, 1)
        self.assertEqual(res[0].x0, 0.0)
        self.assertEqual(res[0].y0, 0.9)
        self.assertEqual(res[0].x1, 0.1)
        self.assertEqual(res[0].y1, 1.0)

        self.assertEqual(res[1].sel_class, 2)
        self.assertEqual(res[1].x0, 0.1)
        self.assertEqual(res[1].y0, 0.2)
        self.assertEqual(res[1].x1, 0.3)
        self.assertEqual(res[1].y1, 0.4)

        self.assertEqual(res[2].sel_class, 3)
        self.assertEqual(res[2].x0, 0.2)
        self.assertEqual(res[2].y0, 0.3)
        self.assertEqual(res[2].x1, 0.6)
        self.assertEqual(res[2].y1, 0.7)

    def test_add_find_set_several_points(self):

        db = load_selection_db('sqlite:///:memory:')

        self.assertFalse(db.has_points('foo/bar/baz.tif'))
        self.assertEqual(list(db.find_points('foo/bar/baz.tif')),
                         [])

        db.add_points('foo/bar/baz.tif',
                      classes=[1, 2, 1, 1],
                      points=[(0.0, 1.0),
                              (0.1, 0.9),
                              (0.2, 0.3),
                              (0.6, 0.3)])

        res = list(db.find_points('foo/bar/baz.tif'))

        self.assertEqual(len(res), 4)

        self.assertEqual(res[0].sel_class, 1)
        self.assertEqual(res[0].x, 0.0)
        self.assertEqual(res[0].y, 1.0)

        self.assertEqual(res[1].sel_class, 2)
        self.assertEqual(res[1].x, 0.1)
        self.assertEqual(res[1].y, 0.9)

        self.assertEqual(res[2].sel_class, 1)
        self.assertEqual(res[2].x, 0.2)
        self.assertEqual(res[2].y, 0.3)

        self.assertEqual(res[3].sel_class, 1)
        self.assertEqual(res[3].x, 0.6)
        self.assertEqual(res[3].y, 0.3)

        db.set_points('foo/bar/baz.tif',
                      classes=[1, 2, 3, 3],
                      points=[(0.0, 0.9),
                              (0.0, 0.9),
                              (0.2, 0.3),
                              (0.2, 0.3)])

        self.assertTrue(db.has_points('foo/bar/baz.tif'))

        res = list(db.find_points('foo/bar/baz.tif'))

        self.assertEqual(len(res), 3)

        self.assertEqual(res[0].sel_class, 1)
        self.assertEqual(res[0].x, 0.0)
        self.assertEqual(res[0].y, 0.9)

        self.assertEqual(res[1].sel_class, 2)
        self.assertEqual(res[1].x, 0.0)
        self.assertEqual(res[1].y, 0.9)

        self.assertEqual(res[2].sel_class, 3)
        self.assertEqual(res[2].x, 0.2)
        self.assertEqual(res[2].y, 0.3)

    def test_add_find_clear_several_points(self):

        db = load_selection_db('sqlite:///:memory:')

        self.assertFalse(db.has_points('foo/bar/baz.tif'))
        self.assertEqual(list(db.find_points('foo/bar/baz.tif')),
                         [])

        db.add_points('foo/bar/baz.tif',
                      classes=[1, 2, 1, 1],
                      points=[(0.0, 1.0),
                              (0.1, 0.9),
                              (0.2, 0.3),
                              (0.2, 0.3)])

        self.assertTrue(db.has_points('foo/bar/baz.tif'))

        res = list(db.find_points('foo/bar/baz.tif'))

        self.assertEqual(len(res), 3)

        db.clear_points('foo/bar/baz.tif')

        self.assertFalse(db.has_points('foo/bar/baz.tif'))
        self.assertEqual(list(db.find_points('foo/bar/baz.tif')),
                         [])

    def test_add_find_clear_several_selections(self):

        db = load_selection_db('sqlite:///:memory:')

        self.assertFalse(db.has_selections('foo/bar/baz.tif'))
        self.assertEqual(list(db.find_selections('foo/bar/baz.tif')),
                         [])

        db.add_selections('foo/bar/baz.tif',
                          classes=[1, 2, 1, 1],
                          selections=[(0.0, 1.0, 0.1, 0.9),
                                      (0.1, 0.9, 0.0, 1.0),
                                      (0.2, 0.3, 0.6, 0.7),
                                      (0.6, 0.3, 0.2, 0.7)])

        self.assertTrue(db.has_selections('foo/bar/baz.tif'))

        res = list(db.find_selections('foo/bar/baz.tif'))

        self.assertEqual(len(res), 3)

        db.clear_selections('foo/bar/baz.tif')

        self.assertFalse(db.has_selections('foo/bar/baz.tif'))
        self.assertEqual(list(db.find_selections('foo/bar/baz.tif')),
                         [])

    def test_find_all_selections(self):

        db = load_selection_db('sqlite:///:memory:')

        db.add_selection('foo/bar/baz.tif',
                         sel_class=2, x0=0.0, x1=1.0, y0=0.1, y1=0.9)
        db.add_selection('foo/bar/baz.tif',
                         sel_class=1, x0=0.0, x1=1.0, y0=0.1, y1=0.9)
        db.add_selection('grr/arg/baz.tif',
                         sel_class=2, x0=0.0, x1=1.0, y0=0.1, y1=0.9)

        all_selections = list(db.find_all_selections())
        self.assertEqual(len(all_selections), 3)

        fpath, roi = all_selections[0]
        self.assertEqual(fpath, 'foo/bar/baz.tif')
        self.assertEqual(roi, (2, 0.0, 0.1, 1.0, 0.9))

        fpath, roi = all_selections[1]
        self.assertEqual(fpath, 'foo/bar/baz.tif')
        self.assertEqual(roi, (1, 0.0, 0.1, 1.0, 0.9))

        fpath, roi = all_selections[2]
        self.assertEqual(fpath, 'grr/arg/baz.tif')
        self.assertEqual(roi, (2, 0.0, 0.1, 1.0, 0.9))

    def test_add_find_ellipses(self):

        db = load_selection_db('sqlite:///:memory:')
        db.add_ellipse('foo/bar/1.tif',
                       sel_class=1, cx=1, cy=2, ct=-np.pi, a=1.0, b=2.0)
        db.add_ellipses('foo/bar/1.tif',
                        ellipses=[
                            (1, 2, -np.pi, 1.0, 2.0),
                            (2, 4, -np.pi, 2.0, 4.0),
                        ], classes=[1, 2])

        all_ellipses = list(db.find_all_ellipses())
        self.assertEqual(len(all_ellipses), 2)

        fpath, ellipse = all_ellipses[0]
        self.assertEqual(fpath, 'foo/bar/1.tif')
        self.assertEqual(ellipse, (1, 1.0, 2.0, 3/2*np.pi, 2.0, 1.0))

        fpath, ellipse = all_ellipses[1]
        self.assertEqual(fpath, 'foo/bar/1.tif')
        self.assertEqual(ellipse, (2, 2.0, 4.0, 3/2*np.pi, 4.0, 2.0))

        db.set_ellipses('foo/bar/1.tif',
                        ellipses=[
                            (3, 4, 0, 5.0, 4.0),
                        ], classes=[3])

        all_ellipses = list(db.find_ellipses('foo/bar/1.tif'))
        self.assertEqual(len(all_ellipses), 1)

        ellipse = all_ellipses[0]
        self.assertEqual(ellipse, (3, 3, 4, 0, 5.0, 4.0))

        db.clear_ellipses('foo/bar/1.tif')

        self.assertFalse(db.has_ellipses('foo/bar/1.tif'))

    def test_find_all_points(self):

        db = load_selection_db('sqlite:///:memory:')

        db.add_point('foo/bar/baz.tif',
                     sel_class=2, x=0.0, y=0.1)
        db.add_point('foo/bar/baz.tif',
                     sel_class=1, x=0.0, y=0.1)

        db.add_point('grr/arg/baz.tif',
                     sel_class=2, x=0.0, y=0.1)

        all_points = list(db.find_all_points())
        self.assertEqual(len(all_points), 3)

        fpath, point = all_points[0]
        self.assertEqual(fpath, 'foo/bar/baz.tif')
        self.assertEqual(point, (2, 0.0, 0.1))

        fpath, point = all_points[1]
        self.assertEqual(fpath, 'foo/bar/baz.tif')
        self.assertEqual(point, (1, 0.0, 0.1))

        fpath, point = all_points[2]
        self.assertEqual(fpath, 'grr/arg/baz.tif')
        self.assertEqual(point, (2, 0.0, 0.1))

    def test_has_annotations(self):

        db = load_selection_db('sqlite:///:memory:')

        db.add_point('foo/bar/baz.tif',
                     sel_class=2, x=0.0, y=0.1)
        db.add_point('grr/arg/baz.tif',
                     sel_class=2, x=0.0, y=0.1)

        db.add_selection('foo/bar/bif.tif',
                         sel_class=1, x0=0.0, x1=1.0, y0=0.1, y1=0.9)
        db.add_selection('grr/arg/baz.tif',
                         sel_class=2, x0=0.0, x1=1.0, y0=0.1, y1=0.9)

        self.assertFalse(db.has_annotations('grr/arg/bif.tif'))

        self.assertTrue(db.has_annotations('foo/bar/baz.tif'))
        self.assertTrue(db.has_annotations('foo/bar/bif.tif'))
        self.assertTrue(db.has_annotations('foo/bar/baz.tif'))

    def test_find_all_annotations(self):

        db = load_selection_db('sqlite:///:memory:')

        db.add_point('foo/bar/baz.tif',
                     sel_class=2, x=0.0, y=0.1)
        db.add_point('foo/bar/baz.tif',
                     sel_class=1, x=0.0, y=0.1)
        db.add_point('grr/arg/baz.tif',
                     sel_class=2, x=0.0, y=0.1)
        db.add_selection('foo/bar/baz.tif',
                         sel_class=2, x0=0.0, x1=1.0, y0=0.1, y1=0.9)
        db.add_selection('foo/bar/baz.tif',
                         sel_class=1, x0=0.0, x1=1.0, y0=0.1, y1=0.9)
        db.add_selection('grr/arg/baz.tif',
                         sel_class=2, x0=0.0, x1=1.0, y0=0.1, y1=0.9)

        all_annotations = list(db.find_all_annotations())
        self.assertEqual(len(all_annotations), 6)

        fpath, point = all_annotations[0]
        self.assertEqual(fpath, 'foo/bar/baz.tif')
        self.assertEqual(point, (2, 0.0, 0.1))

        fpath, point = all_annotations[1]
        self.assertEqual(fpath, 'foo/bar/baz.tif')
        self.assertEqual(point, (1, 0.0, 0.1))

        fpath, point = all_annotations[2]
        self.assertEqual(fpath, 'grr/arg/baz.tif')
        self.assertEqual(point, (2, 0.0, 0.1))

        fpath, roi = all_annotations[3]
        self.assertEqual(fpath, 'foo/bar/baz.tif')
        self.assertEqual(roi, (2, 0.0, 0.1, 1.0, 0.9))

        fpath, roi = all_annotations[4]
        self.assertEqual(fpath, 'foo/bar/baz.tif')
        self.assertEqual(roi, (1, 0.0, 0.1, 1.0, 0.9))

        fpath, roi = all_annotations[5]
        self.assertEqual(fpath, 'grr/arg/baz.tif')
        self.assertEqual(roi, (2, 0.0, 0.1, 1.0, 0.9))


if __name__ == '__main__':
    unittest.main()
