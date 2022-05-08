""" Styling tools for plots

* :py:class:`~set_plot_style`: The plot style context manager
* :py:class:`~colorwheel`: Infinitely iterable color wheel

"""

# Imports
import itertools
import pathlib
from contextlib import ContextDecorator
from typing import Tuple, List, Optional, Dict

THISDIR = pathlib.Path(__file__).resolve().parent

# 3rd party imports
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from matplotlib.axes import Axes

import seaborn as sns

# Our own imports
from .consts import (
    RC_PARAMS_MINIMAL, RC_PARAMS_DARK, RC_PARAMS_LIGHT,
    RC_PARAMS_POSTER, RC_PARAMS_DARK_POSTER, RC_PARAMS_FIGURE,
    COLOR_PALETTE,
)

# Decorators


class set_plot_style(ContextDecorator):
    """ Context manager for styling matplotlib plots

    Basic usage as a context manager

    .. code-block:: python

        with set_plot_style('dark') as style:
            # In here, plots are 'dark' styled
            fig, ax = plt.subplots(1, 1)
            ax.plot([1, 2, 3], [1, 2, 3])
            # Save the plot with correct background colors
            style.savefig('some_fig.png')

    Can also be used as a decorator

    .. code-block:: python

        @set_plot_style('dark')
        def plot_something():
            # In here, plots are 'dark' styled
            fig, ax = plt.subplots(1, 1)
            ax.plot([1, 2, 3], [1, 2, 3])
            plt.show()

    For more complex use, see the
    `Matplotlib rcParam <http://matplotlib.org/users/customizing.html>`_
    docs which list all the parameters that can be tweaked.

    :param str style:
        One of 'dark', 'minimal', 'poster', 'dark_poster', 'default'
    """

    _active_styles = []

    def __init__(self, style: str = 'dark'):
        style = style.lower().strip()
        self.stylename = style
        if style == 'dark':
            self.params = RC_PARAMS_DARK
            self.savefig_params = {'facecolor': 'k',
                                   'edgecolor': 'k'}
        elif style == 'dark_poster':
            self.params = RC_PARAMS_DARK_POSTER
            self.savefig_params = {'facecolor': 'k',
                                   'edgecolor': 'k'}
        elif style == 'poster':
            self.params = RC_PARAMS_POSTER
            self.savefig_params = {'facecolor': 'w',
                                   'edgecolor': 'w'}
        elif style == 'light':
            self.params = RC_PARAMS_LIGHT
            self.savefig_params = {'facecolor': 'w',
                                   'edgecolor': 'w'}
        elif style == 'figure':
            self.params = RC_PARAMS_FIGURE
            self.savefig_params = {'facecolor': 'w',
                                   'edgecolor': 'w'}
        elif style == 'minimal':
            self.params = RC_PARAMS_MINIMAL
            self.savefig_params = {}
        elif style == 'default':
            self.params = {}
            self.savefig_params = {}
        else:
            raise KeyError(f'Unknown plot style: "{style}"')

    @property
    def axis_color(self) -> str:
        """ Get the color for axis edges in the current theme

        :returns:
            The edge color or None
        """
        if self.stylename.startswith('dark'):
            default = 'white'
        else:
            default = 'black'
        return self.params.get('axes.edgecolor', default)

    @property
    def edgecolor(self) -> str:
        """ Get the color for edges in the current theme

        :returns:
            The edge color or None
        """
        if self.stylename.startswith('dark'):
            default = 'white'
        else:
            default = 'black'
        return self.params.get('edgecolor', default)

    @classmethod
    def get_active_style(cls) -> Optional[str]:
        """ Get the currently active style, or None if nothing is active

        :returns:
            The current style or None
        """
        if cls._active_styles:
            return cls._active_styles[-1]
        return None

    def twinx(self, ax: Optional[Axes] = None) -> Axes:
        """ Create a second axis sharing the x axis

        :param Axes ax:
            The axis instance to set to off
        :returns:
            A second axis with a shared x coordinate
        """
        if ax is None:
            ax = plt.gca()
        ax2 = ax.twinx()

        # Fix up the defaults to make sense
        ax2.spines['right'].set_visible(True)
        ax2.tick_params(axis='y',
                        labelcolor=self.axis_color,
                        color=self.axis_color,
                        left=True)
        return ax2

    def set_axis_off(self, ax: Optional[Axes] = None):
        """ Remove labels and ticks from the axis

        :param Axes ax:
            The axis instance to set to off
        """
        if ax is None:
            ax = plt.gca()

        # Blank all the things
        ax.set_xticks([])
        ax.set_yticks([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for side in ('top', 'bottom', 'left', 'right'):
            if side in ax.spines:
                ax.spines[side].set_visible(False)
        ax.set_axis_off()

    def set_image_axis_lims(self,
                            img: np.ndarray,
                            ax: Optional[Axes] = None,
                            axis_off: bool = True):
        """ Set limits for the axis based on the images

        :param ndarray img:
            The image to use for the limits
        :param Axes ax:
            The axis to set limits on
        """
        if ax is None:
            ax = plt.gca()
        rows, cols = img.shape[:2]
        ax.set_xlim([0, cols])
        ax.set_ylim([rows, 0])

        if axis_off:
            self.set_axis_off(ax)

    def rotate_xticklabels(self,
                           ax: Axes,
                           rotation: float,
                           horizontalalignment: str = 'center',
                           verticalalignment: str = 'center',
                           rotation_mode: str = 'default'):
        """ Rotate the x ticklabels

        :param Axes ax:
            The axis instance to rotate xticklabels on
        :param float rotation:
            Rotation of the text (in degrees)
        :param str rotation_mode:
            Either "default" or "anchor"
        """
        for tick in ax.get_xticklabels():
            plt.setp(tick,
                     rotation=rotation,
                     horizontalalignment=horizontalalignment,
                     verticalalignment=verticalalignment,
                     rotation_mode=rotation_mode)

    def rotate_yticklabels(self,
                           ax: Axes,
                           rotation: float,
                           horizontalalignment: str = 'center',
                           verticalalignment: str = 'center',
                           rotation_mode: str = 'default'):
        """ Rotate the y ticklabels

        :param Axes ax:
            The axis instance to rotate xticklabels on
        :param float rotation:
            Rotation of the text (in degrees)
        :param str rotation_mode:
            Either "default" or "anchor"
        """
        #FIXME: Not sure if this is right...
        for tick in ax.get_yticklabels():
            plt.setp(tick,
                     rotation=rotation,
                     horizontalalignment=horizontalalignment,
                     verticalalignment=verticalalignment,
                     rotation_mode=rotation_mode)

    def show(self,
             outfile: Optional[pathlib.Path] = None,
             transparent: bool = True,
             tight_layout: bool = True,
             close: bool = True,
             fig: Optional = None,
             dpi: Optional[int] = None,
             suffixes: Optional[List[str]] = None):
        """ Act like matplotlib's show, but also save the file if passed

        :param Path outfile:
            If not None, save to this file instead of plotting
        :param bool transparent:
            If True, save with a transparent background if possible
        :param bool tight_layout:
            If True, try and squish the layout before saving
        :param bool close:
            If True, close the figure after saving it
        :param Figure fig:
            If not None, the figure to save
        """
        # Save the plot with multiple suffixes
        if suffixes is None:
            suffixes = []
        elif isinstance(suffixes, str):
            suffixes = [suffixes]
        suffixes = ['.' + str(s).lstrip('.') for s in suffixes
                    if s is not None and len(s) > 0]

        # Apply a tight layout before showing the plot
        if tight_layout:
            plt.tight_layout()

        if outfile is None:
            plt.show()
        else:
            # Save the plot a bunch of different ways
            if len(suffixes) == 0:
                suffixes = [outfile.suffix]
            for suffix in suffixes:
                final_outfile = outfile.parent / f'{outfile.stem}{suffix}'
                print(f'Writing {final_outfile}')
                self.savefig(final_outfile, transparent=transparent, fig=fig, dpi=dpi)
            # Close the figure unless it was specifically forced open
            if close:
                plt.close()

    def update(self, params: Dict):
        """ Update the matplotlib rc.params

        :param dict params:
            rcparams to fiddle with
        """
        self.params.update(params)

    def savefig(self,
                savefile: pathlib.Path,
                fig: Optional = None,
                **kwargs):
        """ Save the figure, with proper background colors

        :param Path savefile:
            The file to save
        :param fig:
            The figure or plt.gcf()
        :param \\*\\*kwargs:
            The keyword arguments to pass to fig.savefig
        """
        if fig is None:
            fig = plt.gcf()

        savefile = pathlib.Path(savefile)
        savefile.parent.mkdir(exist_ok=True, parents=True)

        savefig_params = dict(self.savefig_params)
        savefig_params.update(kwargs)
        fig.savefig(str(savefile), **kwargs)

    def load(self):
        """ Load the style """
        self._style = plt.rc_context(self.params)
        self._style.__enter__()
        self._active_styles.append(self.stylename)

    def unload(self, *args, **kwargs):
        """ Unload the active style """
        self._style.__exit__(*args, **kwargs)
        self._active_styles.pop()

    def __enter__(self) -> 'set_plot_style':
        self.load()
        return self

    def __exit__(self, *args, **kwargs):
        self.unload(*args, **kwargs)

# Classes


class colorwheel(mplcolors.Colormap):
    """ Generate colors like a matplotlib color cycle

    .. code-block:: python

        palette = colorwheel(palette='some seaborn palette', n_colors=5)
        for item, color in zip(items, colors):
            # In here, the colors will cycle over and over for each item

        # Access by index
        color = palette[10]

    :param str palette:
        A palette that can be recognized by seaborn
    :param int n_colors:
        The number of colors to generate
    """

    def __init__(self,
                 palette: str = COLOR_PALETTE,
                 n_colors: int = 10):
        if isinstance(palette, colorwheel):
            palette = palette.palette
        self.palette = palette
        self.n_colors = n_colors
        self.N = self.n_colors

        self._idx = 0
        self._color_table = None

    @classmethod
    def from_colors(cls,
                    colors: List[str],
                    n_colors: Optional[int] = None,
                    color_type: str = 'name') -> 'colorwheel':
        """ Make a palette from a list of colors

        :param str colors:
            A list of matplotlib colors to use
        :param int n_colors:
            If not None, the number of colors in the cycle
        :param str color_type:
            Color type, one of ('name', '8bit', 'float')
        :returns:
            A colorwheel object with this palette
        """
        if n_colors is None:
            n_colors = len(colors)
        palette = []
        for _, color in zip(range(n_colors), itertools.cycle(colors)):
            if color_type == 'name':
                norm_color = mplcolors.to_rgba(color)
            elif color_type == '8bit':
                norm_color = tuple(float(c)/255.0 for c in color)
            elif color_type == 'float':
                norm_color = tuple(float(c) for c in color)
            else:
                raise KeyError(f'Unknown color_type "{color_type}"')

            # Make the color RGBA and 0.0 <= c <= 1.0
            if len(norm_color) == 3:
                norm_color = norm_color + (1.0, )
            if len(norm_color) != 4:
                raise ValueError(f'Got invalid color spec {color}')
            if min(norm_color) < 0.0 or max(norm_color) > 1.0:
                raise ValueError(f'Got invalid color range {color}')
            palette.append(norm_color)
        return cls(palette, n_colors=n_colors)

    @classmethod
    def from_color_range(cls,
                         color_start: str,
                         color_end: str,
                         n_colors: int) -> 'colorwheel':
        """ Make a color range

        :param str color_start:
            The start for the color range (i == 0)
        :param str color_end:
            The end for the color range (i == n_colors-1)
        :param int n_colors:
            The number of colors in the range
        :returns:
            A colorwheel object with this palette
        """
        palette = []
        color_start = mplcolors.to_rgba(color_start)
        color_end = mplcolors.to_rgba(color_end)

        red_color = np.linspace(color_start[0], color_end[0], n_colors)
        green_color = np.linspace(color_start[1], color_end[1], n_colors)
        blue_color = np.linspace(color_start[2], color_end[2], n_colors)

        for r, g, b in zip(red_color, green_color, blue_color):
            palette.append((r, g, b, 1.0))
        return cls(palette, n_colors=n_colors)

    @classmethod
    def from_color_anchors(cls,
                           colors: List[str],
                           anchors: List[float],
                           n_colors: int) -> 'colorwheel':
        """ Make a palette with linear ramps between the colors

        :param list[str] colors:
            The set points for the colors
        :param list[float] anchors:
            The points where those colors are mapped
        :param int n_colors:
            The number of colors in the range
        :returns:
            A colorwheel object with this palette
        """
        colors = [mplcolors.to_rgba(c) for c in colors]
        if len(colors) < 2:
            return cls(palette=[colors[0]]*n_colors, n_colors=n_colors)

        anchor_start = np.min(anchors)
        anchor_end = np.max(anchors)

        inds = np.argsort(anchors)
        colors = np.array(colors)[inds, :]
        anchors = np.array(anchors)[inds]
        anchor_inds = np.arange(anchors.shape[0])

        assert colors.shape[0] == anchors.shape[0]
        assert colors.shape[1] == 4

        ranges = np.linspace(anchor_start, anchor_end, n_colors)
        palette = []

        for anchor_pt in ranges:
            left_idx = (anchor_inds[anchors <= anchor_pt])[-1]
            right_idx = left_idx + 1
            if right_idx >= len(anchors):
                color = colors[-1, :]
            else:
                left_anchor = anchors[left_idx]
                right_anchor = anchors[right_idx]
                right_pct = (anchor_pt - left_anchor) / (right_anchor - left_anchor)
                left_pct = 1.0 - right_pct
                left_color = colors[left_idx, :]
                right_color = colors[right_idx, :]

                color = left_color*left_pct + right_color*right_pct
            palette.append(tuple(color))
        return cls(palette, n_colors=n_colors)

    def __call__(self, X: float, alpha=None, bytes=False):
        """ Pretend to be a colormap """
        if isinstance(X, int):
            return self.color_table[self._idx % self.n_colors]
        elif isinstance(X, float):
            n_color = X * self.n_colors
            n_color_low = int(np.floor(n_color))
            n_color_high = int(np.ceil(n_color))
            frac = n_color - n_color_low

            color_left = self.color_table[n_color_low % self.n_colors]
            color_right = self.color_table[n_color_high % self.n_colors]

            return color_left*frac + color_right*(1.0 - frac)
        elif isinstance(X, np.ndarray):
            max_color = self.n_colors
            if X.dtype in (float, np.float32, np.float64):
                X[X < 0.0] = 0.0
                X[X > 1.0] = 1.0

                n_color = X * (max_color - 1)
                n_color_low = np.floor(n_color).astype(int)
                n_color_high = np.ceil(n_color).astype(int)

                n_color_low[n_color_low < 0] = 0
                n_color_high[n_color_high < 0] = 0
                n_color_low[n_color_low >= max_color] = max_color - 1
                n_color_high[n_color_high >= max_color] = max_color - 1

                frac = (n_color - n_color_low)[:, np.newaxis]

                colors_left = np.array([self.color_table[i] for i in n_color_low])
                colors_right = np.array([self.color_table[i] for i in n_color_high])
                colors = (colors_left*frac + colors_right*(1.0 - frac))
                alpha = np.ones((colors.shape[0], 1), dtype=np.float64)
                colors = np.concatenate([colors, alpha], axis=1)
                return colors
            else:
                raise TypeError(f'Unknown array type {X.dtype}')
        raise TypeError(f'Unknown type {type(X)}')

    # Dynamic color palettes
    # These aren't as good as the ones that come with matplotlib
    # They are so todd can be happy

    def wheel_grey(self):
        return [
            (141/255, 141/255, 141/255),
        ]

    def wheel_bluegrey3(self):
        return [
            (0x04/255, 0x04/255, 0x07/255, 1.0),
            (0xb0/255, 0xb0/255, 0xb3/255, 1.0),
            (0x00/255, 0x00/255, 0xff/255, 1.0),
        ]

    def wheel_bluegreywhite3(self):
        return [
            (0xaa/255, 0xaa/255, 0xaa/255, 1.0),
            (0x00/255, 0xff/255, 0xff/255, 1.0),
            (0x00/255, 0x00/255, 0xff/255, 1.0),
        ]

    def wheel_bluegrey4(self):
        return [
            (0xa2/255, 0xa5/255, 0xa7/255, 1.0),
            (0x5c/255, 0xca/255, 0xe7/255, 1.0),
            (0x04/255, 0x07/255, 0x07/255, 1.0),
            (0x3e/255, 0x5b/255, 0xa9/255, 1.0),
        ]

    def wheel_blackwhite(self) -> List[Tuple]:
        """ Colors from black to white in a linear ramp """
        colors = np.linspace(0, 1, self.n_colors)
        return [(c, c, c, 1.0) for c in colors]

    def wheel_greys_light_dark(self) -> List[Tuple]:
        """ Greys from light to dark in a linear ramp """
        colors = np.linspace(0.75, 0.25, self.n_colors)
        return [(c, c, c, 1.0) for c in colors]

    def wheel_greyblack(self) -> List[Tuple]:
        """ Colors from grey to black in a linear ramp """
        colors = np.linspace(0.75, 0, self.n_colors)
        return [(c, c, c, 1.0) for c in colors]

    def wheel_greywhite(self) -> List[Tuple]:
        """ Colors from grey to white in a linear ramp """
        colors = np.linspace(0.25, 1, self.n_colors)
        return [(c, c, c, 1.0) for c in colors]

    def wheel_lightgreywhite(self) -> List[Tuple]:
        """ Colors from grey to white in a linear ramp """
        colors = np.linspace(0.608, 1, self.n_colors)
        return [(c, c, c, 1.0) for c in colors]

    def wheel_whitelightgrey(self) -> List[Tuple]:
        """ Colors from grey to white in a linear ramp """
        colors = np.linspace(1.0, 0.608, self.n_colors)
        return [(c, c, c, 1.0) for c in colors]

    def wheel_redgrey(self) -> List[Tuple]:
        """ Grey to red color space """
        red = np.linspace(155/255, 228/255, self.n_colors)
        green = np.linspace(155/255, 26/255, self.n_colors)
        blue = np.linspace(155/255, 28/255, self.n_colors)
        return [(r, g, b, 1.0) for r, g, b in zip(red, green, blue)]

    def wheel_bluegrey(self) -> List[Tuple]:
        """ Grey to blue color space """
        # red = np.linspace(155/255, 60/255, self.n_colors)
        # green = np.linspace(155/255, 174/255, self.n_colors)
        # blue = np.linspace(155/255, 238/255, self.n_colors)
        red = np.linspace(155/255, 70/255, self.n_colors)
        green = np.linspace(155/255, 130/255, self.n_colors)
        blue = np.linspace(155/255, 180/255, self.n_colors)
        return [(r, g, b, 1.0) for r, g, b in zip(red, green, blue)]

    def wheel_illustrator_bwr(self) -> List[Tuple]:
        """ Load the stupid illustrator BWR version """
        red = []
        green = []
        blue = []
        with (THISDIR / 'illustrator_bwr.txt').open('rt') as fp:
            for line in fp:
                rgb = line.strip()[1:]
                assert len(rgb) == 6
                r, g, b = rgb[:2], rgb[2:4], rgb[4:6]
                red.append(int(r, 16)/256)
                green.append(int(g, 16)/256)
                blue.append(int(b, 16)/256)
        x = np.linspace(0, 1, len(red))
        xf = np.linspace(0, 1, self.n_colors)
        red_f = np.interp(xf, x, red, left=red[0], right=red[-1])
        green_f = np.interp(xf, x, green, left=green[0], right=green[-1])
        blue_f = np.interp(xf, x, blue, left=blue[0], right=blue[-1])
        return [(r, g, b, 1.0) for r, g, b in zip(red_f, green_f, blue_f)]

    @property
    def color_table(self):
        if self._color_table is not None:
            return self._color_table

        # Magic color palettes
        palette = self.palette
        if isinstance(palette, str):
            if palette.startswith('wheel_'):
                palette = getattr(self, palette)()
            elif palette.startswith('color_'):
                color = palette.split('_', 1)[1]
                color = mplcolors.to_rgba(color)
                palette = [color for _ in range(self.n_colors)]
            else:
                palette = palette
        else:
            palette = self.palette

        # Memorize the color table then output it
        self._color_table = sns.color_palette(palette=palette, n_colors=self.n_colors)
        return self._color_table

    def __len__(self):
        return len(self.color_table)

    def __getitem__(self, idx):
        return self.color_table[idx % len(self.color_table)]

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        color = self.color_table[self._idx]
        self._idx = (self._idx + 1) % len(self.color_table)
        return color

    next = __next__
