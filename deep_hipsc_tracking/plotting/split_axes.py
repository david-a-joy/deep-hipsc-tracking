""" Split Axes object for making split grid plots

Make a single split barplot over the y-axis:

.. code-block:: python

    df = pd.DataFrame({
        'cat': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
        'val': [0.0, 1.1, 0.0, 1.1, 1.2, 5.0, 5.1, 5.2, 5.2, 5.1],
    })

    with SplitAxes(ylimits=[(4.5, 6.0), (0, 0.5)]) as axes:
        sns.barplot(data=df, x='cat', y='val', ax=axes)
    plt.show()

Main class:

* :py:class:`SplitAxes`: Create a figure with a set of axes with breaks at fixed places

Utilities:

* :py:class:`MultiAxesProxy`: Weird proxy object to broadcast actions over all the child axes

API Documentation
-----------------

"""


# Imports
import copy

# 3rd party
import matplotlib.pyplot as plt


# Classes


class MultiAxesProxy(object):
    """ Broadcast axis properties over all axes objects

    .. warning:: This is a massive hack and might explode if shaken vigorously

    :param list[Axes] axes:
        The list of axes objects (or sub-objects) to broadcast over
    :param str attr:
        The attribute currently being broadcast

    Currently will broadcast over method calls and attribute access. Doesn't
    handle returns or setting attributes, but so far most matplotlib calls don't care
    """

    def __init__(self, axes, attr):
        self._axes = axes
        self._attr = attr

    def __call__(self, *args, **kwargs):
        for ax in self._axes:
            ret = getattr(ax, self._attr)(*args, **kwargs)
        return ret

    def __getattr__(self, attr):
        if attr.startswith('_'):
            return super(self).__getattr__(attr)
        subaxes = [getattr(ax, self._attr) for ax in self._axes]
        return MultiAxesProxy(subaxes, attr)

    def __repr__(self):
        return f'MultiAxesProxy({self._attr})'

    __str__ = __repr__


class SplitAxes(object):
    """ Matplotlib axes splitting object

    Acts as a split along the x and/or y axis

    :param list[tuple[float]] xlimits:
        If not None, the list of (left, right) splits for the x axis
    :param list[tuple[float]] ylimits:
        If not None, the list of (bottom, top) splits for the y axis
    :param tuple[float] figsize:
        If not None, the figure size to use
    :param float diagonal_linewidth:
        The linewidth for the diagonal line breaks
    :param str diagonal_color:
        The color for the diagonal line breaks (defaults to plt.rc('xtick.color'))

    Example: Split a barplot to show outliers more clearly:

    .. code-block:: python

        df = pd.DataFrame({
            'cat': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
            'val': [0.0, 1.1, 0.0, 1.1, 1.2, 5.0, 5.1, 5.2, 5.2, 5.1],
        })

        with SplitAxes(ylimits=[(4.5, 6.0), (1.0, 2.0), (0, 0.5)]) as axes:
            sns.barplot(data=df, x='cat', y='val', ax=axes)
        plt.show()

    This will make a seaborn barplot split over 3 ypanels

    Example: Split a scatterplot into windows to see different sections of gaussians:

    .. code-block:: python

        blue_loc_x = np.random.normal(loc=1.0, scale=1.0, size=(1000, ))
        blue_loc_y = np.random.normal(loc=1.0, scale=0.5, size=(1000, ))

        red_loc_x = np.random.normal(loc=2, scale=2, size=(1000, ))
        red_loc_y = np.random.normal(loc=2, scale=1.5, size=(1000, ))

        with SplitAxes(ylimits=[(4.5, 6.0), (1.0, 2.0), (0, 0.5)],
                       xlimits=[(0.5, 1.5), (3.0, 5.0)]) as axes:
            axes.plot(blue_loc_x, blue_loc_y, '.b')
            axes.plot(red_loc_x, red_loc_y, '.r')
        plt.show()

    This will make a scatterplot with a 2 row x 3 column grid of axes. It's probably
    not the best way to show this particular dataset...

    """

    def __init__(self, xlimits=None, ylimits=None, figsize=None,
                 diagonal_linewidth=0.01,
                 diagonal_color=None):
        if xlimits is None:
            xlimits = []
        self.xlimits = list(sorted(xlimits))
        if ylimits is None:
            ylimits = []
        self.ylimits = list(sorted(ylimits, reverse=True))

        self.figsize = figsize
        self.diagonal_linewidth = diagonal_linewidth
        if diagonal_color is None:
            diagonal_color = plt.rcParams.get('xtick.color', 'k')
        self.diagonal_color = diagonal_color

        self._fig = None
        self._axes = []

        self._height_ratios = None
        self._width_ratios = None
        self._height_coords = None
        self._width_coords = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args, **kwargs):
        self._close()

    def __iter__(self):
        return iter(self._axes.ravel())

    def __getattr__(self, attr):
        if attr.startswith('_'):
            return super(self).__getattr__(attr)
        # Special unproxied attributes
        if attr in ('lines', 'patches'):
            return getattr(self._axes[0, 0], attr)
        # Try and proxy stuff
        return MultiAxesProxy(self._axes.ravel(), attr)

    def add_patch(self, patch):
        """ Add a patch to the axes

        :param Patch patch:
            The patch object to add to all the axes
        """
        for ax in self._axes.ravel():
            ax.add_patch(copy.copy(patch))

    def legend(self):
        """ Set the legend only on one axis """

        # Set the legend on the top right axis
        ax = self._axes[0, -1]
        ax.legend()

    def set_xlabel(self, label: str):
        """ Set the x-label only on the correct axis

        Work out which sub-axis is most central in x, bottom in y

        :param str label:
            Label to set on the x-axis
        """
        num_rows = max([1, len(self.ylimits)])
        num_cols = max([1, len(self.xlimits)])

        for i in range(num_rows):
            for j in range(num_cols):
                ax = self._axes[i, j]
                norm_left, norm_right = self._width_coords[j], self._width_coords[j+1]
                is_center_x = norm_left <= 0.5 < norm_right

                # Figure out which sub-axes should be labeled or not
                if i == 0 and num_rows > 1:
                    # Top row
                    ax.set_xlabel('')
                elif i < num_rows - 1 and num_rows > 1:
                    # Middle row
                    ax.set_xlabel('')
                else:
                    # Bottom row
                    if is_center_x:
                        #print('Setting xlabel to {}'.format(label))
                        ax.set_xlabel(label)
                    else:
                        ax.set_xlabel('')

    def set_ylabel(self, label: str):
        """ Set the y-label only on the correct axis

        Work out which sub-axis is most central in y, bottom in x

        :param str label:
            Label to set on the y-axis
        """
        num_rows = max([1, len(self.ylimits)])
        num_cols = max([1, len(self.xlimits)])

        for i in range(num_rows):
            norm_bottom, norm_top = self._height_coords[i+1], self._height_coords[i]
            is_center_y = norm_bottom <= 0.5 < norm_top
            for j in range(num_cols):
                ax = self._axes[i, j]
                if j == 0:
                    # Left column
                    if is_center_y:
                        #print('Setting ylabel to {}'.format(label))
                        ax.set_ylabel(label)
                    else:
                        ax.set_ylabel('')
                elif j < num_cols - 1:
                    # middle column
                    ax.set_ylabel('')
                else:
                    # Right column
                    ax.set_ylabel('')

    def set_title(self, title: str):
        """ Set the title only on the correct axis

        Work out which sub-axis is most central in x, top in y

        :param str title:
            Title to set on the combined axis
        """
        num_rows = max([1, len(self.ylimits)])
        num_cols = max([1, len(self.xlimits)])

        for i in range(num_rows):
            for j in range(num_cols):
                ax = self._axes[i, j]
                norm_left, norm_right = self._width_coords[j], self._width_coords[j+1]
                is_center_x = norm_left <= 0.5 < norm_right

                # Figure out which sub-axes should be labeled or not
                if i == 0:
                    # Top row
                    if is_center_x:
                        #print('Setting title to {}'.format(title))
                        ax.set_title(title)
                    else:
                        ax.set_title('')
                elif i < num_rows - 1:
                    # Middle row
                    ax.set_title('')
                else:
                    # Bottom row
                    ax.set_title('')

    # Formatters that ensure the axes layouts are correct after plotting

    def _close(self):
        """ Apply the limits and properly link things """

        num_rows = max([1, len(self.ylimits)])
        num_cols = max([1, len(self.xlimits)])

        # Set all the limits properly for axes
        for i in range(num_rows):
            if self.ylimits:
                bottom, top = self.ylimits[i]
            else:
                bottom = top = None
            norm_bottom, norm_top = self._height_coords[i+1], self._height_coords[i]
            is_center_y = norm_bottom <= 0.5 < norm_top

            for j in range(num_cols):
                if self.xlimits:
                    left, right = self.xlimits[j]
                else:
                    left = right = None
                norm_left, norm_right = self._width_coords[j], self._width_coords[j+1]
                is_center_x = norm_left <= 0.5 < norm_right

                ax = self._axes[i, j]

                # Actually set the limits for this axis window
                if bottom is not None and top is not None:
                    ax.set_ylim(bottom, top)

                if left is not None and right is not None:
                    ax.set_xlim(left, right)

                # Figure out which sub-axes should be ticked or not
                if i == 0 and num_rows > 1:
                    # Top row
                    ax.spines['bottom'].set_visible(False)
                    ax.set_xlabel('')
                    ax.set_xticks([])
                elif i < num_rows - 1 and num_rows > 1:
                    # Middle row
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.set_xlabel('')
                    ax.set_xticks([])
                else:
                    # Bottom row
                    ax.spines['top'].set_visible(False)
                    ax.xaxis.tick_bottom()
                    if not is_center_x:
                        ax.set_xlabel('')

                if j == 0:
                    # Left column
                    ax.yaxis.tick_left()
                    ax.spines['right'].set_visible(False)
                    if not is_center_y:
                        ax.set_ylabel('')
                elif j < num_cols - 1:
                    # middle column
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.set_ylabel('')
                    ax.set_yticks([])
                else:
                    # Right column
                    ax.spines['left'].set_visible(False)
                    ax.set_ylabel('')
                    ax.set_yticks([])

                # Add split marks to the spines
                if self.diagonal_linewidth is not None:
                    kwargs = {
                        'transform': ax.transAxes,
                        'color': self.diagonal_color,
                        'clip_on': False,
                    }
                    d = self.diagonal_linewidth

                    # Scale the ticks for the axis aspect ratio
                    xmin, xmax = self._width_coords[j], self._width_coords[j+1]
                    ymin, ymax = self._height_coords[i+1], self._height_coords[i]

                    xscale = xmax - xmin
                    yscale = ymax - ymin

                    dx = d/xscale
                    dy = d/yscale

                    # Ticks on the x-axis, so only use the bottom row
                    if i == num_rows - 1 and num_cols > 1:
                        xticks = ax.xaxis.get_major_ticks()
                        if xticks == []:
                            continue
                        if j == 0:
                            # Left column, so only slash the right side
                            ax.plot((1-dx, 1+dx), (-dy, +dy), **kwargs)
                            xticks[-1].tick1line.set_visible(False)
                        elif j < num_cols - 1:
                            # middle column, slash both sides
                            ax.plot((-dx, +dx), (-dy, +dy), **kwargs)
                            ax.plot((1-dx, 1+dx), (-dy, +dy), **kwargs)
                            xticks[0].tick1line.set_visible(False)
                            xticks[-1].tick1line.set_visible(False)
                        else:
                            # Right column, slash the left side
                            ax.plot((-dx, +dx), (-dy, +dy), **kwargs)
                            xticks[0].tick1line.set_visible(False)

                    # Ticks on the y-axis, only use the left column
                    if j == 0 and num_rows > 1:
                        yticks = ax.yaxis.get_major_ticks()
                        if yticks == []:
                            continue
                        if i == 0:
                            # Top row, only slash the bottom side
                            ax.plot((-dx, +dx), (-dy, +dy), **kwargs)
                            yticks[0].tick1line.set_visible(False)
                        elif i < num_rows - 1:
                            # middle row, slash both sides
                            ax.plot((-dx, +dx), (-dy, +dy), **kwargs)
                            ax.plot((-dx, +dx), (1-dy, 1+dy), **kwargs)
                            yticks[0].tick1line.set_visible(False)
                            yticks[-1].tick1line.set_visible(False)
                        else:
                            # bottom column, slash the top side
                            ax.plot((-dx, +dx), (1-dy, 1+dy), **kwargs)
                            yticks[-1].tick1line.set_visible(False)

    def _open(self):
        """ Initialize the figure with the correct axes """

        num_rows = max([1, len(self.ylimits)])
        num_cols = max([1, len(self.xlimits)])

        # Work out the ratios
        ylengths = [y1 - y0 for y0, y1 in self.ylimits]
        yspan = sum(ylengths)
        xlengths = [x1 - x0 for x0, x1 in self.xlimits]
        xspan = sum(xlengths)

        height_ratios = [y/min(ylengths) for y in ylengths]
        if not height_ratios:
            height_ratios = [1]
        width_ratios = [x/min(xlengths) for x in xlengths]
        if not width_ratios:
            width_ratios = [1]

        # Use these ratios to calculate normalized axis position
        self._width_ratios = width_ratios
        self._height_ratios = height_ratios

        # Starts from the left, steps to the right
        width_coords = [0]
        for i, step in enumerate(xlengths):
            width_coords.append(width_coords[i] + step/xspan)
        if len(width_coords) == num_cols:
            width_coords.append(1)
        assert round(width_coords[-1], 1) == 1
        self._width_coords = width_coords

        # Height starts from the top and goes to the bottom
        height_coords = [1]
        for i, step in enumerate(ylengths):
            height_coords.append(height_coords[i] - step/yspan)
        if len(height_coords) == num_rows:
            height_coords.append(0)
        assert round(height_coords[-1], 1) == 0
        self._height_coords = height_coords

        # Generate a subplot with the correct scale parameters
        self._fig, self._axes = plt.subplots(
            num_rows, num_cols,
            squeeze=False,
            figsize=self.figsize,
            gridspec_kw={
                'height_ratios': height_ratios,
                'width_ratios': width_ratios,
                'wspace': 0.1,
                'hspace': 0.1,
            },
        )
        # Set up shared axes for x and y
        for i in range(num_rows):
            for j in range(num_cols):
                # Figure out which x and y axis to link to
                x_ax = self._axes[num_rows-1, j]
                y_ax = self._axes[i, 0]
                ax = self._axes[i, j]

                # If we're not the bottom row, sharex
                if i != num_rows-1:
                    ax.get_shared_x_axes().join(ax, x_ax)
                # If we're not the left column, sharey
                if j != 0:
                    ax.get_shared_y_axes().join(ax, y_ax)
        return self
