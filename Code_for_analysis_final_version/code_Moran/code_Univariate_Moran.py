from _viz_esda_mpl import plt, Moran, plot_moran
import libpysal
import numpy as np
import pandas as pd
import geopandas as gpd
import math

np.random.seed(10)

if __name__ == '__main__':

    w = libpysal.io.open(r'../data_for_paper_final_version/Spatial weight matrix excluding diagonal.gal').read()
    sf = gpd.read_file(r'../data_for_paper_final_version/Grids with hash rate and energy(201806-201905)/Grids with hash rate and energy(201806-201905).shp')

    # source code for esda.Moran
    # https://pysal.org/esda/_modules/esda/moran.html#Moran
    mi = Moran(sf['HrateZS'],  w, transformation='r', permutations=999)
    print("Univariate Moran Hashrate :", mi.I, mi.p_sim, mi.z_sim, mi.EI_sim, mi.VI_sim, mi.seI_sim)
    fig, ax = plot_moran(mi, figsize=(10, 4), scatter_kwds=dict(marker='o', s=10),
                         fitline_kwds=dict(color='b', linewidth=3))
    ticks = np.arange(-3, 8, 1)
    plt.xticks(ticks)
    plt.yticks(ticks)
    ax[1].set_title('Moran’s I: ' + str(round(mi.I, 2)))
    ax[1].set_ylabel('Spatial lag of hash rate')
    ax[1].set_xlabel('Hash rate')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    plt.show()
