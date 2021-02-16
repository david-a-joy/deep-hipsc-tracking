""" Test the styling tools """

# Imports
import unittest

# 3rd party
import numpy as np

# Our imports
from deep_hipsc_tracking.plotting import styling

# Tests


class TestColorwheel(unittest.TestCase):

    def test_from_colors_name(self):

        palette = styling.colorwheel.from_colors(['red', 'green', 'blue'], n_colors=4)

        exp_palette = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.50196, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        np.testing.assert_allclose(np.array(palette.palette), np.array(exp_palette), atol=0.001)

    def test_from_colors_rgb(self):

        palette = styling.colorwheel.from_colors(
            [(1.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0)],
            n_colors=4,
            color_type='float')

        exp_palette = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0, 1.0),
            (1.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        np.testing.assert_allclose(np.array(palette.palette), np.array(exp_palette), atol=0.001)

    def test_from_colors_8bit(self):

        palette = styling.colorwheel.from_colors(
            [(255, 0, 0), (0, 255, 255), (255, 0, 255)],
            n_colors=4,
            color_type='8bit')
        exp_palette = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0, 1.0),
            (1.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        np.testing.assert_allclose(np.array(palette.palette), np.array(exp_palette), atol=0.001)

    def test_from_color_range(self):

        palette = styling.colorwheel.from_color_range(
            '#ff0000', '#0000ff', n_colors=8)

        exp_palette = [
            (1.0, 0.0, 0.0, 1.0),
            (0.8571, 0.0, 0.1428, 1.0),
            (0.7142, 0.0, 0.2857, 1.0),
            (0.5714, 0.0, 0.4285, 1.0),
            (0.4285, 0.0, 0.5714, 1.0),
            (0.2857, 0.0, 0.7142, 1.0),
            (0.1428, 0.0, 0.8571, 1.0),
            (0.0, 0.0, 1.0, 1.0),
        ]
        np.testing.assert_allclose(np.array(palette.palette), np.array(exp_palette), atol=0.001)

    def test_from_color_anchors_two_anchors(self):

        palette = styling.colorwheel.from_color_anchors(
            ('#ff0000', '#0000ff'), (0.0, 1.0), n_colors=8)
        exp_palette = [
            (1.0, 0.0, 0.0, 1.0),
            (0.8571, 0.0, 0.1428, 1.0),
            (0.7142, 0.0, 0.2857, 1.0),
            (0.5714, 0.0, 0.4285, 1.0),
            (0.4285, 0.0, 0.5714, 1.0),
            (0.2857, 0.0, 0.7142, 1.0),
            (0.1428, 0.0, 0.8571, 1.0),
            (0.0, 0.0, 1.0, 1.0),
        ]
        np.testing.assert_allclose(np.array(palette.palette), np.array(exp_palette), atol=0.001)

    def test_from_color_anchors_three_anchors(self):

        palette = styling.colorwheel.from_color_anchors(
            ('#ff0000', '#00ff00', '#0000ff'), (-1.0, 0.0, 1.0), n_colors=8)
        exp_palette = [
            (1.0, 0.0, 0.0, 1.0),
            (0.7142, 0.2857, 0.0, 1.0),
            (0.4285, 0.5714, 0.0, 1.0),
            (0.1428, 0.8571, 0.0, 1.0),
            (0.0, 0.8571, 0.1428, 1.0),
            (0.0, 0.5714, 0.4285, 1.0),
            (0.0, 0.2857, 0.7142, 1.0),
            (0.0, 0.0, 1.0, 1.0),
        ]
        np.testing.assert_allclose(np.array(palette.palette), np.array(exp_palette), atol=0.001)

    def test_from_color_anchors_as_heatmap(self):

        palette = styling.colorwheel.from_color_anchors(
            ('#ff0000', '#00ff00', '#0000ff'), (-1.0, 0.0, 1.0), n_colors=9)

        # Heatmaps use the callable interface
        res = palette(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        exp = [
            (1.0, 0.0, 0.0, 1.0),
            (0.5, 0.5, 0.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 0.5, 0.5, 1.0),
            (0.0, 0.0, 1.0, 1.0),
        ]
        np.testing.assert_allclose(res, np.array(exp), atol=0.001)

    def test_from_color_anchors_as_heatmap_out_of_bounds(self):

        palette = styling.colorwheel.from_color_anchors(
            ('#ff0000', '#00ff00', '#0000ff'), (-1.0, 0.0, 1.0), n_colors=9)

        # Heatmaps use the callable interface
        res = palette(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        exp = [
            (1.0, 0.0, 0.0, 1.0),
            (0.5, 0.5, 0.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 0.5, 0.5, 1.0),
            (0.0, 0.0, 1.0, 1.0),
        ]
        np.testing.assert_allclose(res, np.array(exp), atol=0.001)


class TestSetPlotStyle(unittest.TestCase):

    def test_can_get_current_plot_style(self):

        res = styling.set_plot_style.get_active_style()
        self.assertIsNone(res)

        with styling.set_plot_style('light'):
            res = styling.set_plot_style.get_active_style()
            self.assertEqual(res, 'light')

            with styling.set_plot_style('dark'):
                res = styling.set_plot_style.get_active_style()
                self.assertEqual(res, 'dark')

            res = styling.set_plot_style.get_active_style()
            self.assertEqual(res, 'light')

        res = styling.set_plot_style.get_active_style()
        self.assertIsNone(res)
