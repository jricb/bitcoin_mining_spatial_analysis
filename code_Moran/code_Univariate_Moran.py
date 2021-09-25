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

    mi = Moran(sf['HrateZS'],  w, transformation='r', permutations=999, two_tailed=True)
    print("Univariate Moran Hashrate :", mi.I, mi.p_sim, mi.z_sim, mi.EI_sim, mi.VI_sim, mi.seI_sim)
    fig, ax = plot_moran(mi, figsize=(12, 6), scatter_kwds=dict(marker='o', s=10),
                         fitline_kwds=dict(color='b', linewidth=3))

    ax[0].set_xticks([i / 10.0 for i in np.arange(0, 8, 1)])
    ax[1].set_xticks(np.arange(-1, 8, 1))
    ax[1].set_xticks(np.arange(-1, 8, 1))
    ax[1].set_title('Moran’s I: ' + str(round(mi.I, 2)))
    ax[1].set_ylabel('Spatial lag of hash rate')
    ax[1].set_xlabel('Hash rate')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    plt.show()

'''
source code for esda.Moran
https://pysal.org/esda/_modules/esda/moran.html#Moran
class Moran(object):
    def __init__(
            self, y, w, transformation="r", permutations=PERMUTATIONS, two_tailed=True):
        y = np.asarray(y).flatten()
        self.y = y
        w.transform = transformation
        self.w = w
        self.permutations = permutations
        self.__moments()
        self.I = self.__calc(self.z)
        self.z_norm = (self.I - self.EI) / self.seI_norm
        self.z_rand = (self.I - self.EI) / self.seI_rand

        if self.z_norm > 0:
            self.p_norm = 1 - stats.norm.cdf(self.z_norm)
            self.p_rand = 1 - stats.norm.cdf(self.z_rand)
        else:
            self.p_norm = stats.norm.cdf(self.z_norm)
            self.p_rand = stats.norm.cdf(self.z_rand)

        if two_tailed:
            self.p_norm *= 2.0
            self.p_rand *= 2.0

        if permutations:
            sim = [
                self.__calc(np.random.permutation(self.z)) for i in range(permutations)
            ]
            self.sim = sim = np.array(sim)
            above = sim >= self.I
            larger = above.sum()
            if (self.permutations - larger) < larger:
                larger = self.permutations - larger
            self.p_sim = (larger + 1.0) / (permutations + 1.0)
            self.EI_sim = sim.sum() / permutations
            self.seI_sim = np.array(sim).std()
            self.VI_sim = self.seI_sim ** 2
            self.z_sim = (self.I - self.EI_sim) / self.seI_sim
            if self.z_sim > 0:
                self.p_z_sim = 1 - stats.norm.cdf(self.z_sim)
            else:
                self.p_z_sim = stats.norm.cdf(self.z_sim)

        # provide .z attribute that is znormalized
        sy = y.std()
        self.z /= sy


    def __moments(self):
        self.n = len(self.y)
        y = self.y
        z = y - y.mean()
        self.z = z
        self.z2ss = (z * z).sum()
        self.EI = -1.0 / (self.n - 1)
        n = self.n
        n2 = n * n
        s1 = self.w.s1
        s0 = self.w.s0
        s2 = self.w.s2
        s02 = s0 * s0
        v_num = n2 * s1 - n * s2 + 3 * s02
        v_den = (n - 1) * (n + 1) * s02
        self.VI_norm = v_num / v_den - (1.0 / (n - 1)) ** 2
        self.seI_norm = self.VI_norm ** (1 / 2.0)

        # variance under randomization
        xd4 = z ** 4
        xd2 = z ** 2
        k_num = xd4.sum() / n
        k_den = (xd2.sum() / n) ** 2
        k = k_num / k_den
        EI = self.EI
        A = n * ((n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02)
        B = k * ((n2 - n) * s1 - 2 * n * s2 + 6 * s02)
        VIR = (A - B) / ((n - 1) * (n - 2) * (n - 3) * s02) - EI * EI
        self.VI_rand = VIR
        self.seI_rand = VIR ** (1 / 2.0)

    def __calc(self, z):
        zl = slag(self.w, z)
        inum = (z * zl).sum()
        return self.n / self.w.s0 * inum / self.z2ss

    @property
    def _statistic(self):
        """More consistent hidden attribute to access ESDA statistics"""
        return self.I

    @classmethod
    def by_col(
        cls, df, cols, w=None, inplace=False, pvalue="sim", outvals=None, **stat_kws
    ):
        
        return _univariate_handler(
            df,
            cols,
            w=w,
            inplace=inplace,
            pvalue=pvalue,
            outvals=outvals,
            stat=cls,
            swapname=cls.__name__.lower(),
            **stat_kws
        )

'''