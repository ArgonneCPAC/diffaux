from itertools import zip_longest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

from ..size_modeling.fit_size_data import _sigmoid, get_color_mask, median_size_vs_z
from .plot_utilities import get_nrow_ncol, save_fig

matplotlib.rc("text.latex", preamble=r"\usepackage{bm}")
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

PLOT_DRN = "./SizePlots"


def plot_generated_sizes(
    Re,
    R_med,
    color_gal,
    log_Mstar,
    redshift,
    samples,
    authors,
    data,
    val_info,
    z_lo=0.0,
    z_hi=3.0,
    Nz=7,
    logM_lo=9.0,
    logM_hi=12.0,
    NM=6,
    plotdir=PLOT_DRN,
    fontsize=12,
    pltname="GalaxySizes_vs_Mstar_zbins_{}.png",
):
    # logM_bins = np.linspace(np.floor(log_Mstar), np.ceil(log_Mstar)
    z_bins = np.linspace(z_lo, z_hi, Nz)
    nrow, ncol = get_nrow_ncol(len(z_bins))
    pt_colors = ("royalblue", "tomato")  # for simple plot
    med_colors = ("blue", "red")

    fig, ax_all = plt.subplots(nrow, ncol, figsize=(ncol * 7, nrow * 5))
    for ax, z_min, z_max in zip_longest(ax_all.flat, z_bins[0:-1], z_bins[1:]):
        if z_min is None:
            ax.set_visible(False)
            continue
        zmask = (redshift >= z_min) & (redshift < z_max)
        for sample, pcolor, med_color in zip(samples, pt_colors, med_colors):
            cmask = get_color_mask(color_gal, sample)
            mask = zmask & cmask
            ztitle = f"${z_min:.1f} \\leq z < {z_max:.1f}$"
            ax.scatter(log_Mstar[mask], Re[mask], color=pcolor, alpha=0.4, label=sample)
            # profile plot of Mstar vs R_med

            result = binned_statistic(
                log_Mstar[mask],
                [R_med[mask], R_med[mask] ** 2],
                bins=NM,
                range=(logM_lo, logM_hi),
                statistic="mean",
            )
            means, means2 = result.statistic
            bin_edges = result.bin_edges
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            ax.plot(
                bin_centers, means, color=med_color, label=f"{sample} $\\bm{{R_e^{{med}} }}$", linewidth=3
            )
            for author in authors:
                if ztitle not in data[author] and "z-values" not in val_info:
                    print(f"..Skipping bin {ztitle} for {author}: no data")
                    continue
                if ztitle in data[author]:  # choose sub-dict
                    v = data[author][ztitle]
                else:  # create mask if no subdict
                    v = data[author]
                    zcol = val_info["z-values"]
                    zval_mask = (v[zcol] >= z_min) & (v[zcol] < z_max)
                idx = val_info[author]["samples"].index(sample)
                dcolor = val_info[author]["colors"][idx]
                for ycol, yerrp, yerrn in zip(
                    val_info["y-values"],
                    val_info["y-errors+"],
                    val_info["y-errors-"],
                ):  # so far these are 1-entry lists
                    xcol = val_info["x-values"].format(sample)
                    ycol = ycol.format(sample)
                    mask = v[xcol] > 0.0
                    # print(np.count_nonzero(zval_mask), np.count_nonzero(mask))
                    if ztitle not in data[author]:
                        mask = mask & zval_mask
                    if np.count_nonzero(mask) > 0:
                        label = "{} {} ({} ${}\\mu m$)".format(
                            sample,
                            val_info["lgnd_label"],
                            val_info[author]["short_title"],
                            val_info[author]["wavelength"],
                        )
                        # label = '{} {}'.format(val_info['lgnd_label'], sample)
                        if ztitle in data[author]:
                            ax.plot(v[xcol][mask], v[ycol][mask], color=dcolor, label=label)
                            y_upper = v[ycol][mask] + v[yerrp.format(sample)][mask]
                            y_lower = v[ycol][mask] - v[yerrn.format(sample)][mask]
                            ax.fill_between(v[xcol][mask], y_lower, y_upper, facecolor=dcolor, alpha=0.2)
                        else:
                            ax.errorbar(
                                v[xcol][mask],
                                v[ycol][mask],
                                yerr=v[yerrp.format(sample)][mask],
                                color=dcolor,
                                label=label,
                                linestyle="",
                                marker=val_info[author]["marker"],
                            )

            ax.legend(loc="best")
            ax.set_title(ztitle)
            ax.set_xlabel(val_info["xlabel"])
            ax.set_ylabel(val_info["ylabel"])
    short_titles = [val_info[author]["short_title"] for author in authors]
    author_list = "_".join(authors)
    title_list = "+".join(short_titles)
    fig.suptitle(f"Generated Sizes and {title_list} Data", y=0.92, fontweight="bold")
    save_fig(fig, plotdir, pltname.format(author_list))

    return


def plot_fits(
    fits,
    data_vectors,
    samples,
    val_info,
    data=None,
    authors=None,
    func=_sigmoid,
    title="",
    dYp="y-errors+",
    dYn="y-errors-",
    plotdir=PLOT_DRN,
    fontsize=12,
    pltname="Fit_Re_vs_z{}.png",
    fit_label="\\log(M^*/M_\\odot)",
):
    parameters = val_info["y-values"]
    nrow = len(samples)
    ncol = len(parameters)
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(ncol * 7, nrow * 5))
    colors = ("blue", "red")  # for simple plot
    for sample, ax_row, color in zip(samples, ax_all, colors):
        M = data_vectors[val_info["x-values"] + f"_{sample}"]
        for par, ax, dYp, dYn, xlabel, ylabel in zip(
            parameters,
            ax_row,
            val_info["y-errors+"],
            val_info["y-errors-"],
            val_info["xlabels"],
            val_info["ylabels"],
        ):
            fit_key = par.format(sample)
            y = data_vectors[fit_key]
            dy = data_vectors[dYp.format(sample)]
            if not authors:
                ax.errorbar(M, y, yerr=dy, label=fit_key, color=color, marker="o", linestyle="")
            elif data:  # fancy plot with full legend
                for author in authors:
                    idx = val_info[author]["samples"].index(sample)
                    mcolor = val_info[author]["colors"][idx]
                    xcol = val_info["x-values"]
                    yerr = np.fmax(data[author][dYp.format(sample)], data[author][dYn.format(sample)])
                    # mask for missing values
                    yvalues = data[author][fit_key]
                    mask = np.abs(yvalues) > 0.0
                    marker = val_info[author]["marker"]
                    sum_label = "{} ({}) ${}\\mu m$".format(
                        sample, val_info[author]["short_title"], val_info[author]["wavelength"]
                    )
                    ax.errorbar(
                        data[author][xcol][mask],
                        yvalues[mask],
                        yerr=yerr[mask],
                        label=sum_label,
                        color=mcolor,
                        marker=marker,
                        linestyle="",
                    )
            else:
                print(" Missing data dict")
            popt = fits[fit_key]["popt"]
            Msort = np.sort(M)
            label = "$\\bm{{{:.2g}+({:.2g}-{:.2g})/(1+\\exp(-{:.2g}*({} -{:.3g})))}}$".format(
                popt[2], popt[3], popt[2], popt[1], fit_label, popt[0]
            )
            ax.plot(Msort, func(Msort, *popt), color="black", label=label)
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.set_title(sample)
            legend_properties = {"weight": "bold"}
            ax.legend(loc="best", fontsize="small", prop=legend_properties)

    # fig.suptitle('{} ({})'.format())
    save_fig(fig, plotdir, pltname.format(title))
    return


def plot_re_median(
    fit_parameters,
    samples,
    lM_lo=9.0,
    lM_hi=12.0,
    Nm=12,
    z_lo=0.0,
    z_hi=3.0,
    Nz=12,
    fontsize=14,
    pltname="Re_median_vs_Mstar_ztrends.png",
    plotdir=PLOT_DRN,
    size_func=median_size_vs_z,
    fit_func=_sigmoid,
):
    nrow = 1
    ncol = len(samples)
    logMs = np.linspace(lM_lo, lM_hi, Nm + 1)
    zs = np.linspace(z_lo, z_hi, Nz + 1)
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(ncol * 7, nrow * 5))
    for ax, sample in zip(ax_all.flat, samples):
        parameters = [par for par in fit_parameters._fields if sample in par]
        func_pars = [getattr(fit_parameters, par) for par in parameters]
        func_values = [fit_func(logMs, *fpar) for fpar in func_pars]
        # print(func_values)
        for z in zs:
            Re = size_func(z, *func_values)
            ax.plot(logMs, Re, label=f"z = {z:.2f}")

        ax.set_xlabel("$\\bm{\\log_{10}(M^*/M_\\odot)}$")
        ax.set_ylabel("$\\bm{R_{e}^{med} (\\mathrm{kpc})}$")
        ax.legend(loc="best")
        ax.set_title(sample)

    save_fig(fig, plotdir, pltname)

    return
