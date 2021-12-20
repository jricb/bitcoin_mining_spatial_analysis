from _viz_esda_mpl import plot_moran_bv, plt, Moran_BV, Moran
import libpysal
import numpy as np
import pandas as pd
import geopandas as gpd
import math

np.random.seed(10)

if __name__ == '__main__':

    w = libpysal.io.open(r'../data_for_paper_final_version/Spatial weight matrix including diagonal.gal').read()
    sf = gpd.read_file(r'../data_for_paper_final_version/Grids with hash rate and energy(201806-201905)/Grids with hash rate and energy(201806-201905).shp')
    cap_fossil_lg = [math.log10(i + 1) for i in sf['Cap.Fos']]
    cap_renewable_lg = [math.log10(i + 1) for i in sf['Cap.Ren']]
    cap_all_lg = [math.log10(i + 1) for i in sf['Cap.All']]
    h_rate_lg = [math.log10(i + 1) for i in sf['HrateAvg']]

    #source code for esda.Moran_BV
    #https://pysal.org/esda/_modules/esda/moran.html#Moran_BV
    mbi_fos = Moran_BV(h_rate_lg, cap_fossil_lg, w, transformation='r', permutations=999)
    mbi_ren = Moran_BV(h_rate_lg, cap_renewable_lg, w, transformation='r', permutations=999)
    mbi_all = Moran_BV(h_rate_lg, cap_all_lg, w, transformation='r', permutations=999)
    print("Bivariate Moran All :", mbi_all.I, mbi_all.p_sim, mbi_all.z_sim, mbi_all.EI_sim, mbi_all.VI_sim,
          mbi_all.seI_sim)
    print("Bivariate Moran Fos :", mbi_fos.I, mbi_fos.p_sim, mbi_fos.z_sim, mbi_fos.EI_sim, mbi_fos.VI_sim,
          mbi_fos.seI_sim)
    print("Bivariate Moran Ren :", mbi_ren.I, mbi_ren.p_sim, mbi_ren.z_sim, mbi_ren.EI_sim, mbi_ren.VI_sim,
          mbi_ren.seI_sim)

    fig, ax = plot_moran_bv(mbi_all, figsize=(10, 4), scatter_kwds=dict(marker='o', s=10),
                            fitline_kwds=dict(color='b', linewidth=3))
    ticks = np.arange(-3, 8, 1)
    plt.xticks(ticks)
    plt.yticks(ticks)
    ax[1].set_title('Moran’s I: ' + str(round(mbi_all.I, 2)))
    ax[1].set_xlabel('Hash rate')
    ax[1].set_ylabel('Spatial lag of capacity')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    plt.show()

    fig, ax = plot_moran_bv(mbi_fos, figsize=(10, 4), scatter_kwds=dict(marker='o', s=10),
                            fitline_kwds=dict(color='b', linewidth=3))
    ticks = np.arange(-3, 8, 1)
    plt.xticks(ticks)
    plt.yticks(ticks)
    ax[1].set_title('Moran’s I: ' + str(round(mbi_fos.I, 2)))
    ax[1].set_xlabel('Hash rate')
    ax[1].set_ylabel('Spatial lag of capacity')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    plt.show()

    fig, ax = plot_moran_bv(mbi_ren, figsize=(10, 4), scatter_kwds=dict(marker='o', s=10),
                            fitline_kwds=dict(color='b', linewidth=3))
    ticks = np.arange(-3, 8, 1)
    plt.xticks(ticks)
    plt.yticks(ticks)
    ax[1].set_title('Moran’s I: ' + str(round(mbi_ren.I, 2)))
    ax[1].set_xlabel('Hash rate')
    ax[1].set_ylabel('Spatial lag of capacity')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    plt.show()