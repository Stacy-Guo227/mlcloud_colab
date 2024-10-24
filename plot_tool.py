import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmaps

class plotTools:
    def __init__(self, time, height):
        self.Time = time
        self.Height = height
    def hovmollar(self, var, cmap=cm.turbo, vrange=None, title='', subtitle='', unit='', xlabel='LST', ylabel='', \
                             xlim=None, ylim=None, xticks=[None], yticks=[None], \
                             plot_blh=False, blh=[], blh_label=[], blh_color=None, blh_legend_loc='upper left', \
                             ):
        if vrange==None:
            plt.pcolormesh(self.Time, self.Height, var.T, cmap = cmap)
        else:
            plt.pcolormesh(self.Time, self.Height, var.T, cmap = cmap, vmin = vrange[0], vmax = vrange[1])
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(unit, fontsize = 12)
        cbar.ax.tick_params(labelsize = 12)
        if plot_blh:          self.plot_BLH(blh, blh_label, color=blh_color, legend_loc=blh_legend_loc)
        plt.title(title, loc = 'left', weight = 'heavy', fontsize = 14)
        plt.title(subtitle, loc = 'right', fontsize = 10)
        plt.xlabel(xlabel, fontsize = 12)
        plt.ylabel(ylabel, fontsize = 12)
        if xlim != None:      plt.xlim(xlim[0], xlim[1])
        if ylim != None:      plt.ylim(ylim[0], ylim[1])
        if xticks[0] != None:
            plt.xticks(xticks)
        else:
            plt.xticks(np.arange(self.Time[1]//3+self.Time[1], self.Time[-1]+1, 3))
        if yticks[0] != None: plt.yticks(yticks)
        plt.tick_params(labelsize = 12)
        plt.grid(lw = 0.5, ls = ':', c = 'grey')

    def plot_BLH(self, blh, label, color=None, legend_loc='upper left'):
        for i in range(len(blh)):
            if color==None:
                plt.scatter(self.Time, blh[i], s = 5, zorder = 10, label = label)
            else:
                plt.scatter(self.Time, blh[i], color = color[i], s = 5, zorder = 10, label = label)
        if len(label) != 0: plt.legend(loc = legend_loc)
