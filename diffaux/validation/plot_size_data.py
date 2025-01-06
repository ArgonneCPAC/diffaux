from .plot_utilities import get_nrow_ncol, save_fig, fix_plotid
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc('text.latex', preamble=r'\usepackage{bm}')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

PLOT_DRN = './SizePlots'


def plot_Re_vs_z_Mstar_data(data, author, validation_info, plotdir=PLOT_DRN,
                            pltname='log10Re_vs_Mstar_{}.png', samples=None,
                            fig=None, ax_all=None,
                            summary_fig=None, summary_ax=None, save_summary=False, summary_only=False):

    if 'z-values' not in validation_info.keys():
        zkeys = [k for k in data.keys() if 'z' in k]
    else:
        zcol = validation_info['z-values']
        zvalues = np.unique(data[zcol])
        zkeys = ['z = {:.2f}'.format(z) for z in zvalues]

    nrow, ncol = get_nrow_ncol(len(zkeys))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(ncol * 7, nrow * 5))

    for ax, zlabel in zip(ax_all.flat, zkeys):
        samples = validation_info[author]['samples'] if samples is None else samples
        for sample in samples:
            idx = validation_info[author]['samples'].index(sample)
            color = validation_info[author]['colors'][idx]
            for ycol, yerrp, yerrn in zip(validation_info['y-values'],
                                          validation_info['y-errors+'],
                                          validation_info['y-errors-'],
                                          ):  # so far these are 1-entry lists
                xcol = validation_info['x-values'].format(sample)
                ycol = ycol.format(sample)
                v = data[zlabel] if 'z-values' not in validation_info.keys() else data

                if xcol in v.keys() and ycol in v.keys():
                    if 'z-values' not in validation_info.keys():
                        mask = v[xcol] > 0.
                    else:
                        zval_mask = (
                            data[zcol] == zvalues[zkeys.index(zlabel)])
                        mask = (v[xcol] > 0.) & zval_mask

                    if np.count_nonzero(mask) > 0:
                        ax.plot(
                            v[xcol][mask],
                            v[ycol][mask],
                            color=color,
                            label=sample)
                        y_upper = v[ycol][mask] + v[yerrp.format(sample)][mask]
                        y_lower = v[ycol][mask] - v[yerrn.format(sample)][mask]
                        ax.fill_between(v[xcol][mask], y_lower, y_upper,
                                        facecolor=color, alpha=0.2)

        ax.set_xlabel(validation_info['xlabel'])
        ax.set_ylabel(validation_info['ylabel'])
        ax.legend(loc='best')
        ax.set_title(zlabel)

    # not used for now but save code
    if 'M*_colnames' in validation_info[author].keys():
        if 'M*_lo' in validation_info[author]['M*_colnames'] and 'M*_hi' in validation_info[author]['M*_colnames']:
            idx_lo = validation_info[author]['M*_colnames'].index('M*_lo')
            idx_hi = validation_info[author]['M*_colnames'].index('M*_hi')
            Mlabels = []
            for Mlo, Mhi in zip(data[validation_info[author]['M*_colnames'][idx_lo]],
                                data[validation_info[author]['M*_colnames'][idx_hi]]):
                Mlabels.append(
                    '${:.1f} \\leq \\log10(M^*/M_\\odot) < {:.1f}$'.format(Mlo, Mhi))

    fig.tight_layout()
    fig.suptitle(validation_info[author]['suptitle'], y=1.01)

    pltname = pltname.format(author)
    save_fig(fig, plotdir, pltname)

    return summary_fig, summary_ax


def plot_Re_vs_z_Mstar_fits(data, author, validation_info, plotdir=PLOT_DRN,
                            pltname='Re_vs_z_Mstar_{}.png', fontsize=14,
                            summary_fig=None, summary_ax=None, save_summary=False, summary_only=False):

    # fit coefficient plot: Re = B + (1 + z)^-β
    # fit coefficient plot: Re = A*(M/M_p)^𝛂

    nrow, ncol = get_nrow_ncol(len(validation_info['y-values']))
    if not summary_only:
        fig, ax_all = plt.subplots(nrow, ncol, figsize=(ncol * 7, nrow * 5))
    else:
        ax_all = np.zeros(nrow * ncol)
    if summary_fig is None:
        summary_fig, summary_ax = plt.subplots(
            nrow, ncol, figsize=(ncol * 7, nrow * 5))
    for ax, sumax, yvalue, yerr_p, yerr_m, xlabel, ylabel, title in zip(ax_all.flat,
                                                                        summary_ax.flat,
                                                                        validation_info['y-values'],
                                                                        validation_info['y-errors+'],
                                                                        validation_info['y-errors-'],
                                                                        validation_info['xlabels'],
                                                                        validation_info['ylabels'],
                                                                        validation_info['titles'],
                                                                        ):
        for sample, color in zip(
                validation_info[author]['samples'], validation_info[author]['colors']):
            ycol = yvalue.format(sample)
            lower_error = np.abs(data[yerr_p.format(sample)])
            upper_error = np.abs(data[yerr_m.format(sample)])
            xcol = validation_info['x-values'].format(sample)
            # mask for missing values
            mask = np.abs(data[ycol]) > 0.
            # print(asymmetric_error)
            if np.count_nonzero(mask) > 0:
                asymmetric_error = [lower_error[mask], upper_error[mask]]
                if not summary_only:
                    ax.errorbar(data[xcol][mask], data[ycol][mask], yerr=asymmetric_error, label=sample, color=color,
                                marker=validation_info[author]['marker'], linestyle='')
                if 'All' not in sample:
                    sum_label = '{} ({} ${}\\mu m$)'.format(sample, validation_info[author]['short_title'],
                                                            validation_info[author]['wavelength'])
                    if 'M_p_label' in validation_info[author].keys():
                        M_p_txt = validation_info[author]['M_p_label']
                        if ycol + M_p_txt in data.keys():  # check if column exits
                            ycol = ycol + M_p_txt
                            asymmetric_error = [np.abs(data[yerr_p.format(sample) + M_p_txt][mask]),
                                                np.abs(data[yerr_m.format(sample) + M_p_txt][mask])]
                            print(
                                '..Note: Plotting {} for {} in summary'.format(
                                    ycol, author))

                    sumax.errorbar(data[xcol][mask], data[ycol][mask], yerr=asymmetric_error, label=sum_label, color=color,
                                   marker=validation_info[author]['marker'], linestyle='')

        if 'M_p' in validation_info[author].keys():
            title = title.format(validation_info[author]['M_p'])
        if not summary_only:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold")
            legend_properties = {'weight': 'bold'}
            ax.legend(loc='best', fontsize='small', prop=legend_properties)
            fig.suptitle('{} (${}\\mu m$)'.format(validation_info[author]['suptitle'],
                                                  validation_info[author]['wavelength']), y=1.01,
                         fontweight="bold")

        if save_summary:
            sumax.set_xlabel(xlabel, fontsize=fontsize)
            sumax.set_ylabel(ylabel, fontsize=fontsize)
            sumax.set_title('{}'.format(title), fontweight='bold')
            # legend_properties = {'weight':'bold'}
            # , prop=legend_properties)
            sumax.legend(loc='best', ncols=2, fontsize='x-small')

    if not summary_only:
        save_fig(fig, plotdir, pltname.format(author))
    if save_summary:
        save_fig(summary_fig, plotdir, pltname.format('summary'))

    return summary_fig, summary_ax


def plot_size_data(data, validation_info, authors, info_keys=[], plotdir=PLOT_DRN,
                   summary_only=False, plttype='.png', fits={}, xpltname='',):
    sum_fig = None
    sum_ax = None
    info_key = data.keys() if not info_keys else info_keys
    for author in authors:
        for key in info_keys:
            plotter = validation_info[key]['plotter']
            pltname = '_'.join([key, '{}', xpltname]
                               ) if xpltname else '_'.join([key, '{}'])
            pltname = pltname + plttype
            if key in data.keys() and author in data[key].keys():
                save_summary = (author == authors[-1])
                sum_fig, sum_ax = plotter(data[key][author], author, validation_info[key], plotdir=plotdir,
                                          pltname='{}_{{}}'.format(key), summary_only=summary_only,
                                          summary_fig=sum_fig, summary_ax=sum_ax, save_summary=save_summary)

    return
