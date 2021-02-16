""" Constants for default plotting values

Mostly a set of tailored RC params for matplotlib to make the plots look good:

Scaling fonts and linewidths for presentations/figures/posters

* RC_PARAMS_FONTSIZE_NORMAL: Normal size for powerpoint slides
* RC_PARAMS_FONTSIZE_POSTER: 1.5x size for posters
* RC_PARAMS_LINEWIDTH_NORMAL: Normal size for powerpoint slides
* RC_PARAMS_LINEWIDTH_POSTER: 1.5x size for posters

Configure font families to match between mathtext, regular text

* RC_PARAMS_FONT: Normal fonts for slides/posters
* RC_PARAMS_FONT_FIGURE: Fancy fonts for paper figures

Core stylesheets:

* RC_PARAMS_LIGHT: The stylesheet used by ``set_plot_style('light')``
* RC_PARAMS_FIGURE: The stylesheet used by ``set_plot_style('figure')``
* RC_PARAMS_POSTER: The stylesheet used by ``set_plot_style('poster')``

Dark mode stylesheets:

* RC_PARAMS_DARK: The stylesheet used by ``set_plot_style('dark')``
* RC_PARAMS_DARK_POSTER: The stylesheet used by ``set_plot_style('dark_poster')``

Minimal parameter reset:

* RC_PARAMS_MINIMAL: (Deprecated) the stylesheet used by ``set_plot_style('minimal')``

"""

# Constants

SUFFIX = '.png'  # File type to save the plots

PLOT_STYLE = 'light'  # Style of the plots

PALETTE = 'Set1'  # Palette to use for color cycles

BARPLOT_ORIENT = 'vertical'  # Should bar plots be vertical or horizontal

FIGSIZE = (8, 8)  # Size of the figure in inches (width, height)

LINEWIDTH = 5  # Points, size of the line in figures
MARKERSIZE = 3  # Point sizes for dots in figures

COLOR_PALETTE = 'deep'  # colorwheel default palette

# Font selections by platform, because fonts are annoying
MATHTEXT = 'Arial'  # Use Arial for slides
MATHTEXT_FIGURE = 'Helvetica'  # Use Helvetica in figures

# Line control RC params
RC_PARAMS_LINEWIDTH_NORMAL = {
    'axes.linewidth': '2',
    'lines.linewidth': '5',
    'lines.markersize': '10',
    'xtick.major.size': '5',
    'ytick.major.size': '5',
    'xtick.major.width': '1.5',
    'ytick.major.width': '1.5',
    'xtick.minor.size': '3',
    'ytick.minor.size': '3',
    'xtick.minor.width': '1.5',
    'xtick.minor.width': '1.5',
}
RC_PARAMS_LINEWIDTH_POSTER = {
    k: '{:0.0f}'.format(float(v) * 1.5)
    for k, v in RC_PARAMS_LINEWIDTH_NORMAL.items()
}
RC_PARAMS_LINE = {
    'grid.linestyle': '-',
    'lines.dash_capstyle': 'butt',
    'lines.dash_joinstyle': 'miter',
    'lines.solid_capstyle': 'projecting',
    'lines.solid_joinstyle': 'miter',
}

# Font control RC params
RC_PARAMS_FONTSIZE_NORMAL = {
    'figure.titlesize': '32',
    'axes.titlesize': '24',
    'axes.labelsize': '20',
    'xtick.labelsize': '20',
    'ytick.labelsize': '20',
    'legend.fontsize': '20',
}
RC_PARAMS_FONTSIZE_POSTER = {
    k: '{:0.0f}'.format(float(v) * 1.5)
    for k, v in RC_PARAMS_FONTSIZE_NORMAL.items()
}
RC_PARAMS_FONT = {
    'font.family': 'sans-serif',
    'font.sans-serif': f'{MATHTEXT}, Arial, Liberation Sans, Bitstream Vera Sans, DejaVu Sans, sans-serif',
    'text.usetex': 'False',
    'mathtext.default': 'regular',
    'mathtext.fontset': 'custom',
    'mathtext.it': f'{MATHTEXT}:italic',
    'mathtext.rm': f'{MATHTEXT}',
    'mathtext.tt': f'{MATHTEXT}',
    'mathtext.bf': f'{MATHTEXT}:bold',
    'mathtext.cal': f'{MATHTEXT}',
    'mathtext.sf': f'{MATHTEXT}',
    # 'mathtext.fallback_to_cm': 'True',
    'mathtext.fallback': 'cm',  # or 'stix' or 'stixsans'
    'svg.fonttype': 'none',  # or 'path' to render as paths
}
RC_PARAMS_FONT_FIGURE = {
    'font.family': 'sans-serif',
    'font.sans-serif': f'{MATHTEXT_FIGURE}, Helvetica, sans-serif',
    'text.usetex': 'False',
    'mathtext.default': 'regular',
    'mathtext.fontset': 'custom',
    'mathtext.it': f'{MATHTEXT_FIGURE}:italic',
    'mathtext.rm': f'{MATHTEXT_FIGURE}',
    'mathtext.tt': f'{MATHTEXT_FIGURE}',
    'mathtext.bf': f'{MATHTEXT_FIGURE}:bold',
    'mathtext.cal': f'{MATHTEXT_FIGURE}',
    'mathtext.sf': f'{MATHTEXT_FIGURE}',
    # 'mathtext.fallback_to_cm': 'True',
    'mathtext.fallback': 'cm',  # or 'stix' or 'stixsans'
    'svg.fonttype': 'none',  # or 'path' to render as paths
}

# Parameters for face/border/line colors in dark mode
RC_PARAMS_DARK = {
    'figure.facecolor': 'black',
    'figure.edgecolor': 'white',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'axes.facecolor': 'black',
    'axes.edgecolor': 'white',
    'axes.axisbelow': 'True',
    'axes.grid': 'False',
    'axes.spines.left': 'True',
    'axes.spines.bottom': 'True',
    'axes.spines.top': 'False',
    'axes.spines.right': 'False',
    'axes.axisbelow': 'True',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'legend.frameon': 'False',
    'legend.numpoints': '1',
    'legend.scatterpoints': '1',
    'image.cmap': 'Greys',
    'hatch.color': 'white',
    'grid.color': 'white',
    **RC_PARAMS_FONT,
    **RC_PARAMS_FONTSIZE_NORMAL,
    **RC_PARAMS_LINE,
    **RC_PARAMS_LINEWIDTH_NORMAL,
}

# Parameters for face/border/line colors in light mode
RC_PARAMS_LIGHT = {
    'figure.facecolor': 'white',
    'figure.edgecolor': 'black',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.spines.left': 'True',
    'axes.spines.bottom': 'True',
    'axes.spines.top': 'False',
    'axes.spines.right': 'False',
    'axes.axisbelow': 'True',
    'axes.grid': 'False',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.top': 'False',
    'xtick.bottom': 'True',
    'ytick.left': 'True',
    'ytick.right': 'False',
    'legend.frameon': 'False',
    'legend.numpoints': '1',
    'legend.scatterpoints': '1',
    'image.cmap': 'Greys',
    'grid.linestyle': '-',
    'hatch.color': 'black',
    'grid.color': 'black',
    **RC_PARAMS_FONT,
    **RC_PARAMS_FONTSIZE_NORMAL,
    **RC_PARAMS_LINE,
    **RC_PARAMS_LINEWIDTH_NORMAL,
}

# Parameters for face/border/line colors in figure mode
RC_PARAMS_FIGURE = {
    'figure.facecolor': 'white',
    'figure.edgecolor': 'black',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.spines.left': 'True',
    'axes.spines.bottom': 'True',
    'axes.spines.top': 'False',
    'axes.spines.right': 'False',
    'axes.axisbelow': 'True',
    'axes.grid': 'False',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.top': 'False',
    'xtick.bottom': 'True',
    'ytick.left': 'True',
    'ytick.right': 'False',
    'legend.frameon': 'False',
    'legend.numpoints': '1',
    'legend.scatterpoints': '1',
    'image.cmap': 'Greys',
    'grid.linestyle': '-',
    'hatch.color': 'black',
    'grid.color': 'black',
    **RC_PARAMS_FONT_FIGURE,
    **RC_PARAMS_FONTSIZE_POSTER,
    **RC_PARAMS_LINE,
    **RC_PARAMS_LINEWIDTH_POSTER,
}

# Overrides for light and dark poster mode
RC_PARAMS_POSTER = {
    **RC_PARAMS_LIGHT,
    **RC_PARAMS_FONTSIZE_POSTER,
    **RC_PARAMS_LINEWIDTH_POSTER,
}
RC_PARAMS_DARK_POSTER = {
    **RC_PARAMS_DARK,
    **RC_PARAMS_FONTSIZE_POSTER,
    **RC_PARAMS_LINEWIDTH_POSTER,
}

# Minimal parameters for backwards compatibility
RC_PARAMS_MINIMAL = {
    'figure.titlesize': '32',
    'axes.titlesize': '24',
    'axes.labelsize': '20',
    'xtick.labelsize': '20',
    'ytick.labelsize': '20',
    'legend.fontsize': '20',
}
