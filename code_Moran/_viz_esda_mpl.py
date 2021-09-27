import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import numpy as np
from libpysal.weights.contiguity import Queen
from libpysal.weights.spatial_lag import lag_spatial
import seaborn as sbn
from esda.moran import (Moran_Local, Moran_Local_BV,
                        Moran, Moran_BV)
import warnings
from spreg import OLS

from matplotlib import patches, colors

from _viz_utils import splot_colors
# (mask_local_auto, moran_hot_cold_spots,
#                          splot_colors)

"""
Lightweight visualizations for esda using Matplotlib and Geopandas

TODO
* geopandas plotting, change round shapes in legends to boxes
* prototype moran_facet using `seaborn.FacetGrid`
"""



def _create_moran_fig_ax(ax, figsize, aspect_equal):
    """
    Creates matplotlib figure and axes instances
    for plotting moran visualizations. Adds common viz design.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    ax.spines['left'].set_position(('axes', -0.05))
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('axes', -0.05))
    ax.spines['top'].set_color('none')
    if aspect_equal is True:
        ax.set_aspect('equal')
    return fig, ax


def moran_scatterplot(moran, zstandard=True, p=None,
                      aspect_equal=True, ax=None,
                      scatter_kwds=None, fitline_kwds=None):
    """
    Moran Scatterplot
    
    Parameters
    ----------
    moran : esda.moran instance
        Values of Moran's I Global, Bivariate and Local
        Autocorrelation Statistics
    zstandard : bool, optional
        If True, Moran Scatterplot will show z-standardized attribute and
        spatial lag values. Default =True.
    p : float, optional
        If given, the p-value threshold for significance
        for Local Autocorrelation analysis. Points will be colored by
        significance. By default it will not be colored.
        Default =None.
    aspect_equal : bool, optional
        If True, Axes will show the same aspect or visual proportions
        for Moran Scatterplot.
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Moran scatterplot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted
    
    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import (Moran, Moran_BV,
    ...                         Moran_Local, Moran_Local_BV)
    >>> from splot.esda import moran_scatterplot
    
    Load data and calculate weights
    
    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate esda.moran Objects
    
    >>> moran = Moran(y, w)
    >>> moran_bv = Moran_BV(y, x, w)
    >>> moran_loc = Moran_Local(y, w)
    >>> moran_loc_bv = Moran_Local_BV(y, x, w)
    
    Plot
    
    >>> fig, axs = plt.subplots(2, 2, figsize=(10,10),
    ...                         subplot_kw={'aspect': 'equal'})
    >>> moran_scatterplot(moran, p=0.05, ax=axs[0,0])
    >>> moran_scatterplot(moran_loc, p=0.05, ax=axs[1,0])
    >>> moran_scatterplot(moran_bv, p=0.05, ax=axs[0,1])
    >>> moran_scatterplot(moran_loc_bv, p=0.05, ax=axs[1,1])
    >>> plt.show()
    
    """
    if isinstance(moran, Moran):
        if p is not None:
            warnings.warn('`p` is only used for plotting `esda.moran.Moran_Local`\n'
                          'or `Moran_Local_BV` objects')
        fig, ax = _moran_global_scatterplot(moran=moran, zstandard=zstandard,
                                            ax=ax, aspect_equal=aspect_equal,
                                            scatter_kwds=scatter_kwds,
                                            fitline_kwds=fitline_kwds)
    elif isinstance(moran, Moran_BV):
        if p is not None:
            warnings.warn('`p` is only used for plotting `esda.moran.Moran_Local`\n'
                          'or `Moran_Local_BV` objects')
        fig, ax = _moran_bv_scatterplot(moran_bv=moran, ax=ax,
                                        aspect_equal=aspect_equal,
                                        scatter_kwds=scatter_kwds,
                                        fitline_kwds=fitline_kwds)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return fig, ax


def _moran_global_scatterplot(moran, zstandard=True,
                              aspect_equal=True, ax=None,
                              scatter_kwds=None, fitline_kwds=None):
    """
    Global Moran's I Scatterplot.

    Parameters
    ----------
    moran : esda.moran.Moran instance
        Values of Moran's I Global Autocorrelation Statistics
    zstandard : bool, optional
        If True, Moran Scatterplot will show z-standardized attribute and
        spatial lag values. Default =True.
    aspect_equal : bool, optional
        If True, Axes will show the same aspect or visual proportions.
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Moran scatterplot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran
    >>> from splot.esda import moran_scatterplot
    
    Load data and calculate weights
    
    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate Global Moran
    
    >>> moran = Moran(y, w)
    
    plot
    
    >>> moran_scatterplot(moran)
    >>> plt.show()
    
    customize plot
    
    >>> fig, ax = moran_scatterplot(moran, zstandard=False,
    ...                             fitline_kwds=dict(color='#4393c3'))
    >>> ax.set_xlabel('Donations')
    >>> plt.show()
    
    """
    # to set default as an empty dictionary that is later filled with defaults
    if scatter_kwds is None:
        scatter_kwds = dict()
    if fitline_kwds is None:
        fitline_kwds = dict()

    # define customization defaults
    scatter_kwds.setdefault('alpha', 0.6)
    scatter_kwds.setdefault('color', splot_colors['moran_base'])
    scatter_kwds.setdefault('s', 40)
    
    fitline_kwds.setdefault('alpha', 0.9)
    fitline_kwds.setdefault('color', splot_colors['moran_fit'])
    
    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize=(7, 7),
                                   aspect_equal=aspect_equal)
    
    # set labels
    ax.set_xlabel('Attribute')
    ax.set_ylabel('Spatial Lag')
    ax.set_title('Moran Scatterplot' +
                 ' (' + str(round(moran.I, 2)) + ')')

    # plot and set standards
    if zstandard is True:
        lag = lag_spatial(moran.w, moran.z)
        fit = OLS(moran.z[:, None], lag[:, None])
        # plot
        ax.scatter(moran.z, lag, **scatter_kwds)
        ax.plot(lag, fit.predy, **fitline_kwds)
        # v- and hlines
        ax.axvline(0, alpha=0.5, color='k', linestyle='--')
        ax.axhline(0, alpha=0.5, color='k', linestyle='--')
    else:
        lag = lag_spatial(moran.w, moran.y)
        b, a = np.polyfit(moran.y, lag, 1)
        # plot
        ax.scatter(moran.y, lag, **scatter_kwds)
        ax.plot(moran.y, a + b*moran.y, **fitline_kwds)
        # dashed vert at mean of the attribute
        ax.vlines(moran.y.mean(), lag.min(), lag.max(), alpha=0.5,
                  linestyle='--')
        # dashed horizontal at mean of lagged attribute
        ax.hlines(lag.mean(), moran.y.min(), moran.y.max(), alpha=0.5,
                  linestyle='--')
    return fig, ax


def plot_moran_simulation(moran, aspect_equal=True,
                          ax=None, fitline_kwds=None,
                          **kwargs):
    """
    Global Moran's I simulated reference distribution.

    Parameters
    ----------
    moran : esda.moran.Moran instance
        Values of Moran's I Global Autocorrelation Statistics
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the
        vertical moran fitline. Default =None.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the figure,
        passed to seaborn.kdeplot.

    Returns
    -------
    fig : Matplotlib Figure instance
        Simulated reference distribution figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran
    >>> from splot.esda import plot_moran_simulation
    
    Load data and calculate weights
    
    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate Global Moran
    
    >>> moran = Moran(y, w)
    
    plot
    
    >>> plot_moran_simulation(moran)
    >>> plt.show()
    
    customize plot
    
    >>> plot_moran_simulation(moran, fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()
    
    """
    # to set default as an empty dictionary that is later filled with defaults
    if fitline_kwds is None:
        fitline_kwds = dict()

    figsize = kwargs.pop('figsize', (7, 7))
    
    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize,
                                   aspect_equal=aspect_equal)

    # plot distribution
    shade = kwargs.pop('shade', True)
    color = kwargs.pop('color', splot_colors['moran_base'])
    sbn.kdeplot(moran.sim, shade=shade, color=color, ax=ax, **kwargs)

    # customize plot
    fitline_kwds.setdefault('color', splot_colors['moran_fit'])
    ax.vlines(moran.I, 0, 1, **fitline_kwds)
    ax.vlines(moran.EI, 0, 1)
    ax.set_title('Reference Distribution')
    ax.set_xlabel('Moran I: ' + str(round(moran.I, 2)))
    return fig, ax


def plot_moran(moran, zstandard=True, aspect_equal=True,
               scatter_kwds=None, fitline_kwds=None, **kwargs):
    """
    Global Moran's I simulated reference distribution and scatterplot.

    Parameters
    ----------
    moran : esda.moran.Moran instance
        Values of Moran's I Global Autocorrelation Statistics
    zstandard : bool, optional
        If True, Moran Scatterplot will show z-standardized attribute and
        spatial lag values. Default =True.
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline
        and vertical fitline. Default =None.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the figure,
        passed to seaborne.kdeplot.

    Returns
    -------
    fig : Matplotlib Figure instance
        Moran scatterplot and reference distribution figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran
    >>> from splot.esda import plot_moran
    
    Load data and calculate weights
    
    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate Global Moran
    
    >>> moran = Moran(y, w)
    
    plot
    
    >>> plot_moran(moran)
    >>> plt.show()
    
    customize plot
    
    >>> plot_moran(moran, zstandard=False,
    ...            fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()
    
    """
    figsize = kwargs.pop('figsize', (10, 4))
    fig, axs = plt.subplots(1, 2, figsize=figsize,
                            subplot_kw={'aspect': 'equal'})
    plot_moran_simulation(moran, ax=axs[0], fitline_kwds=fitline_kwds, **kwargs)
    moran_scatterplot(moran, zstandard=zstandard, ax=axs[1],
                      scatter_kwds=scatter_kwds, fitline_kwds=fitline_kwds)
    axs[0].set(aspect="auto")
    if aspect_equal is True:
        axs[1].set_aspect("equal", "datalim")
    else: 
        axs[1].set_aspect("auto")
    return fig, axs


def _moran_bv_scatterplot(moran_bv, ax=None, aspect_equal=True,
                          scatter_kwds=None, fitline_kwds=None):
    """
    Bivariate Moran Scatterplot.

    Parameters
    ----------
    moran_bv : esda.moran.Moran_BV instance
        Values of Bivariate Moran's I Autocorrelation Statistics
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Bivariate moran scatterplot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran_BV
    >>> from splot.esda import moran_scatterplot
    
    Load data and calculate weights
    
    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate Bivariate Moran
    
    >>> moran_bv = Moran_BV(x, y, w)
    
    plot
    
    >>> moran_scatterplot(moran_bv)
    >>> plt.show()
    
    customize plot
    
    >>> moran_scatterplot(moran_bv,
    ...                      fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()
    
    """
    # to set default as an empty dictionary that is later filled with defaults
    if scatter_kwds is None:
        scatter_kwds = dict()
    if fitline_kwds is None:
        fitline_kwds = dict()

    # define customization
    scatter_kwds.setdefault('alpha', 0.6)
    scatter_kwds.setdefault('color', splot_colors['moran_base'])
    scatter_kwds.setdefault('s', 40)
    
    fitline_kwds.setdefault('alpha', 0.9)
    fitline_kwds.setdefault('color', splot_colors['moran_fit'])

    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize=(7,7),
                                   aspect_equal=aspect_equal)
    
    # set labels
    ax.set_xlabel('Attribute X')
    ax.set_ylabel('Spatial Lag of Y')
    ax.set_title('Bivariate Moran Scatterplot' +
                 ' (' + str(round(moran_bv.I, 2)) + ')')

    # plot and set standards
    lag = lag_spatial(moran_bv.w, moran_bv.zy)
    fit = OLS(moran_bv.zx[:, None], lag[:, None])
    # plot
    ax.scatter(moran_bv.zx, lag, **scatter_kwds)
    ax.plot(lag, fit.predy, **fitline_kwds)
    # v- and hlines
    ax.axvline(0, alpha=0.5, color='k', linestyle='--')
    ax.axhline(0, alpha=0.5, color='k', linestyle='--')
    return fig, ax


def plot_moran_bv_simulation(moran_bv, ax=None, aspect_equal=True,
                             fitline_kwds=None, **kwargs):
    """
    Bivariate Moran's I simulated reference distribution.

    Parameters
    ----------
    moran_bv : esda.moran.Moran_BV instance
        Values of Bivariate Moran's I Autocorrelation Statistics
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the
        vertical moran fitline. Default =None.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the figure,
        passed to seaborne.kdeplot.

    Returns
    -------
    fig : Matplotlib Figure instance
        Bivariate moran reference distribution figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran_BV
    >>> from splot.esda import plot_moran_bv_simulation
    
    Load data and calculate weights
    
    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate Bivariate Moran
    
    >>> moran_bv = Moran_BV(x, y, w)
    
    plot
    
    >>> plot_moran_bv_simulation(moran_bv)
    >>> plt.show()
    
    customize plot
    
    >>> plot_moran_bv_simulation(moran_bv,
    ...                          fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()
    
    """
    # to set default as an empty dictionary that is later filled with defaults
    if fitline_kwds is None:
        fitline_kwds = dict()

    figsize = kwargs.pop('figsize', (7, 7))

    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize,
                                   aspect_equal=aspect_equal)

    # plot distribution
    shade = kwargs.pop('shade', True)
    color = kwargs.pop('color', splot_colors['moran_base'])
    sbn.kdeplot(moran_bv.sim, shade=shade, color=color, ax=ax, **kwargs)

    # customize plot
    fitline_kwds.setdefault('color', splot_colors['moran_fit'])
    ax.vlines(moran_bv.I, 0, 1, **fitline_kwds)
    ax.vlines(moran_bv.EI_sim, 0, 1)
    ax.set_title('Reference Distribution')
    ax.set_xlabel('Bivariate Moran I: ' + str(round(moran_bv.I, 2)))
    return fig, ax


def plot_moran_bv(moran_bv, aspect_equal=True,
                  scatter_kwds=None, fitline_kwds=None, **kwargs):
    """
    Bivariate Moran's I simulated reference distribution and scatterplot.

    Parameters
    ----------
    moran_bv : esda.moran.Moran_BV instance
        Values of Bivariate Moran's I Autocorrelation Statistics
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline
        and vertical fitline. Default =None.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the figure,
        passed to seaborne.kdeplot.

    Returns
    -------
    fig : Matplotlib Figure instance
        Bivariate moran scatterplot and reference distribution figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran_BV
    >>> from splot.esda import plot_moran_bv
    
    Load data and calculate weights
    
    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate Bivariate Moran
    
    >>> moran_bv = Moran_BV(x, y, w)
    
    plot
    
    >>> plot_moran_bv(moran_bv)
    >>> plt.show()
    
    customize plot
    
    >>> plot_moran_bv(moran_bv, fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()
    
    """
    figsize = kwargs.pop('figsize', (10, 4))
    fig, axs = plt.subplots(1, 2, figsize=figsize,
                            subplot_kw={'aspect': 'equal'})
    plot_moran_bv_simulation(moran_bv, ax=axs[0], fitline_kwds=fitline_kwds,
                             **kwargs)
    moran_scatterplot(moran_bv, ax=axs[1], aspect_equal=aspect_equal,
                      scatter_kwds=scatter_kwds, fitline_kwds=fitline_kwds)
    axs[0].set(aspect="auto")
    if aspect_equal is True:
        axs[1].set_aspect("equal", "datalim")
    else:
        axs[1].set(aspect="auto")
    return fig, axs

