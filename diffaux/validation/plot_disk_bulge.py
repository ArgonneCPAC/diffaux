import os
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import matplotlib.cm as cm
from itertools import zip_longest
from diffaux.validation.plot_utilities import get_nrow_ncol
plotdir = '/Users/kovacs/cosmology/DiskBulgePlots'


def plot_qs_nocuts(qs, zvalues, redshifts,  dz=0.1,
                  qlabels=['Bulge', 'Disk', 'Total'], normalize=False, lbox=100,
                  colors = ['r', 'blue', 'black'], xlabel='$\\log_{10}(M^*/M_\\odot)$',
                  pltname='N_vs_logM_{}.png', yscale='log', xscale='log', bins=50, xname='disk_bulge_total',
                  plotdir = os.path.join(plotdir, 'DiskBulge_Histograms'), lgnd_title=''):
    
    nrow, ncol = get_nrow_ncol(len(zvalues))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow))
    bin_widths = bins[1:] - bins[:-1]

    for ax, z in zip_longest(ax_all.flat, zvalues):
        zmask = (z-dz <= redshifts) & (z+dz >= redshifts)
        zlabel = '${:.1f} \\leq z \\leq {:.1f}$'.format(max(0., z-dz), z+dz)
        # apply row mask to arrays
        qs_z = []
        for q in qs:
            q_z = jnp.ravel(q[:, zmask])
            qs_z.append(q_z)
            
        for q_z, qlabel, color in zip(qs_z, qlabels, colors):
            N,_,_ = ax.hist(q_z, bins=bins, color=color, label=qlabel, histtype='step')

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        #ax.set_ylabel('$dN/dlog_{10}(M^*/M_\\odot)$')
        ax.set_ylabel('$N$')
        ax.legend(loc='best', title=zlabel)

    fn = os.path.join(plotdir, pltname.format(xname))
    plt.tight_layout()
    plt.savefig(fn)
    print('Saving {}'.format(fn))

    return


def plot_histories(qs, t_table, labels, ylabels, plot_label=None,
                   color_array=None, row_mask=None, lgnd_label='#{}', 
                   pltname='History_{}_step_{}.png', yscale='', reverse=False,
                   check_step=5000, xlimlo=0.5, xlimhi=14,
                   step=300, plotdir = os.path.join(plotdir, 'DiskBulge_Histories')):
    
    nrow, ncol = get_nrow_ncol(len(qs))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow))
    colors=cm.coolwarm(np.linspace(0, 1, len(qs[0]))) #expect all q to be same length
    #print(len(colors))
    indices = np.linspace(0, len(qs[0])-1, len(qs[0]), dtype=int)
    if not plot_label: #assume plot_label, color_array, indx_array supplied together
        color_array = indices
    if row_mask is None:
        row_mask = np.ones(len(qs[0]), dtype=bool)
    sort_array = np.argsort(color_array[row_mask])
    if reverse:
        sort_array =  sort_array[::-1]
        print(len(color_array[row_mask]), color_array[row_mask][sort_array][::check_step])

    for ax, q, ylabel in zip_longest(ax_all.flat, qs, ylabels):
        if ylabel is None:
            ax.set_visible(False)
            continue
        if len(color_array[row_mask]) != len(q[row_mask]):
            print('oops: array mismatch')
        for n, (h, c, i) in enumerate(zip(q[row_mask][sort_array][::step],
                                          color_array[row_mask][sort_array][::step],
                                          indices[row_mask][::step])):
            if n==0:
                __=ax.plot(t_table, h, color=colors[i], label=lgnd_label.format(c))
            elif n==int(len(q)/step):
                __=ax.plot(t_table, h, color=colors[i], label=lgnd_label.format(c))
            else:
                __=ax.plot(t_table, h, color=colors[i])

        ax.set_xlim(xlimlo, xlimhi)
        ax.set_ylabel(ylabel)
        if yscale=='log' and ('SMH' in ylabel or 'SFH' in ylabel or 'SFR' in ylabel):
            ax.set_yscale('log')
        ax.set_xlabel('$t$ (Gyr)')
        ax.legend(loc='best')
    fig.suptitle('Sample Histories')
    
    xname = '_'.join(labels)
    if plot_label:
        xname = '_'.join([xname, plot_label])
    if yscale=='log':
        xname = '_'.join([xname, yscale])
    fn = os.path.join(plotdir, pltname.format(xname, step))
    plt.tight_layout()
    plt.savefig(fn)
    print('Saving {}'.format(fn))

    return


def plot_q_with_cuts(q, zvalues, redshifts, cut_array, cuts, dz=0.1,
                     cut_labels=['{{}} $\\leq$ {:.0f}', '{{}} $\\geq$ {:.0f}'],
                     colors = ['r', 'blue'], xlabel='B/T', cut_name='$\\log_{10}(sSFR/yr)$',
                     pltname='BT_cut_on_{}.png', yscale='log', xscale='', bins=50, xname='log_sSFR',
                     plotdir = os.path.join(plotdir, 'DiskBulge_Histograms'), lgnd_title=''):
    
    nrow, ncol = get_nrow_ncol(len(zvalues))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow))

    for ax, z in zip_longest(ax_all.flat, zvalues):
        zmask = (z-dz <= redshifts) & (z+dz >= redshifts)
        zlabel = '${:.1f} \\leq z \\leq {:.1f}$'.format(max(0., z-dz), z+dz)
        # apply row mask to arrays
        q_z = q[:, zmask]
        cut_array_z = cut_array[:, zmask]
        #print(cut_array_z.shape, q_z.shape)
        for n, (cut, cut_label, color) in enumerate(zip(cuts, cut_labels, colors)):
            cut_mask = (cut_array_z <= cut) if n==0 else (cut_array_z >= cut)
            label = cut_label.format(cut).format(cut_name)
            #print(np.count_nonzero(cut_mask), q_z[cut_mask].shape)
            ax.hist(q_z[cut_mask], bins=bins, color=color, label=label, alpha=0.4,
                    density=True)
        if yscale=='log':
            ax.set_yscale('log')
        if xscale=='log':
            ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('PDF')
        ax.legend(loc='best', title=zlabel+lgnd_title)
    
    fn = os.path.join(plotdir, pltname.format(xname))
    plt.tight_layout()
    print('Saving {}'.format(fn))
    plt.savefig(fn)

    return


def plot_q1_q2(q1, q2, zvalues, redshifts, cut_array, cut_lo, cut_hi, dz=0.1,
           cut_label='{:.1f} $\\leq$ {{}} $\\leq$ {:.1f}', qlabels=['Bulge', 'Disk'],
           cut_at_z0=True,
           colors = ['r', 'blue'], xlabel='sSFR $(yr^{-1})$', cut_name='$\\log_{10}(M^*_{z=0}/M_\\odot)$',
           pltname='sSFR_cut_on_{}.png', yscale='log', xscale='log', bins=50, xname='log_M0_{:.1f}_{:.1f}',
           plotdir = os.path.join(plotdir, 'DiskBulge_Histograms'), lgnd_title=''):
    
    nrow, ncol = get_nrow_ncol(len(zvalues))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow))

    clabel = cut_label.format(cut_lo, cut_hi).format(cut_name)
    for ax, z in zip_longest(ax_all.flat, zvalues):
        zmask = (z-dz <= redshifts) & (z+dz >= redshifts)
        zlabel = '${:.1f} \\leq z \\leq {:.1f}$'.format(max(0., z-dz), z+dz)
        # apply row mask to arrays
        q1_z = q1[:, zmask]
        q2_z = q2[:, zmask]
        cut_array_z = cut_array[:, zmask]
        #print(q1_z.shape, cut_array_z.shape)
        if cut_at_z0: #cut on value at z=0
            cut_mask = (cut_array_z[:, -1] >= cut_lo) & (cut_array_z[:, -1] < cut_hi)
            #now broadcast back to 2-d mask
            cut_mask = np.broadcast_to(cut_mask, (np.count_nonzero(zmask), len(cut_mask))).T
        else: #cut on values in redshift range
            cut_mask = (cut_array_z >= cut_lo) & (cut_array_z < cut_hi)
        print(z,  np.count_nonzero(cut_mask))
        for q_z, qlabel, color in zip([q1_z[cut_mask], q2_z[cut_mask]], qlabels, colors):
            ax.hist(q_z, bins=bins, color=color, label=qlabel, alpha=0.4, density=True)

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_xlim(max(np.min(bins), min(np.min(q1_z[cut_mask]), np.min(q1_z[cut_mask]))*0.5),
                    min(np.max(bins), max(np.max(q1_z[cut_mask]), np.max(q1_z[cut_mask])))*2.)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('PDF')
        ax.legend(loc='best', title='\n'.join([zlabel, clabel]))

    fn = os.path.join(plotdir, pltname.format(xname.format(cut_lo, cut_hi)))
    plt.tight_layout()
    plt.savefig(fn)
    print('Saving {}'.format(fn))

    return
    

import matplotlib.cm as cm
from diffaux.validation.plot_utilities import get_subsample
def plot_q1_vs_q2(qx, qy, zvalues, redshifts, color_array, dz=0.1,
                  ymin=-14, ymax=-9, cbar_title="B/T",
                  ylabel='$\\log_{10}(sSFR/yr)$', xlabel='$\\log_{10}(M^*/M_\\odot)$', 
                  cmap='jet', N=1000,
                  pltname='sSFR_vs_Mstar_{}.png', yscale='linear', xscale='linear', xname='',
                  plotdir = os.path.join(plotdir, 'DiskBulge_Scatter'), title=''):
    
    nrow, ncol = get_nrow_ncol(len(zvalues))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow))

    #clabel = cut_label.format(cut_lo, cut_hi).format(cut_name)
    for ax, z in zip_longest(ax_all.flat, zvalues):
        zmask = (z-dz <= redshifts) & (z+dz >= redshifts)
        zlabel = '${:.1f} \\leq z \\leq {:.1f}$'.format(max(0., z-dz), z+dz)
        # apply row mask to arrays
        qx_z = jnp.ravel(qx[:, zmask])
        qy_z = jnp.ravel(qy[:, zmask])
        color_array_z = jnp.ravel(color_array[:, zmask])
        #subsample
        Nobj, idx = get_subsample(len(qx_z), N)
        print(np.min(qx_z[idx]), np.max(qx_z[idx]), np.min(qy_z[idx]), np.max(qy_z[idx]))
        im = ax.scatter(qx_z[idx], qy_z[idx], c=color_array_z[idx], cmap=cmap)
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        qymin = jnp.min(qy_z[idx])
        ymin = ymin if qymin < ymin else 1.1*qymin
        qymax = jnp.max(qy_z[idx])
        ymax = 0.92*qymax if qymax > ymax else ymax
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis='y', labelrotation=90)
        ax.set_ylabel(ylabel, labelpad=0.0)
        ax.text(.05, .05, zlabel, transform=ax.transAxes)

    fn = os.path.join(plotdir, pltname.format(xname))
    cbar = fig.colorbar(im, ax=ax_all, location='right', fraction=0.1)
    #cbar.ax.set_title(cbar_title, rotation=90)
    cbar.set_label(cbar_title, rotation=90, labelpad=3.0, y=0.5, fontsize=14)
    fig.suptitle(title, y=0.97)
    #plt.tight_layout()
    plt.savefig(fn)
    print('Saving {}'.format(fn))

    return




