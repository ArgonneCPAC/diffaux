import os
from itertools import zip_longest

import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

from ..disk_bulge_modeling import disk_bulge_kernels as dbk
from .plot_utilities import get_nrow_ncol, get_subsample

plotdir = "/Users/kovacs/cosmology/DiskBulgePlots"


def plot_qs_nocuts(
    qs,
    zvalues,
    redshifts,
    dz=0.1,
    qlabels=("Bulge", "Disk", "Total"),
    normalize=False,
    lbox=100,
    colors=("r", "blue", "black"),
    xlabel="$\\log_{10}(M^*/M_\\odot)$",
    pltname="N_vs_logM_{}.png",
    yscale="log",
    xscale="log",
    bins=50,
    xname="disk_bulge_total",
    plotdir="./",
    plotsubdir="DiskBulge_Histograms",
    lgnd_title="",
):
    plotdir = os.path.join(plotdir, plotsubdir)
    nrow, ncol = get_nrow_ncol(len(zvalues))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))
    # bin_widths = bins[1:] - bins[:-1]

    for ax, z in zip_longest(ax_all.flat, zvalues):
        zmask = (z - dz <= redshifts) & (z + dz >= redshifts)
        zlabel = f"${max(0., z-dz):.1f} \\leq z \\leq {z+dz:.1f}$"
        # apply row mask to arrays
        qs_z = []
        for q in qs:
            q_z = jnp.ravel(q[:, zmask])
            qs_z.append(q_z)

        for q_z, qlabel, color in zip(qs_z, qlabels, colors):
            N, _, _ = ax.hist(q_z, bins=bins, color=color, label=qlabel, histtype="step")

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        # ax.set_ylabel('$dN/dlog_{10}(M^*/M_\\odot)$')
        ax.set_ylabel("$N$")
        ax.legend(loc="best", title=zlabel)

    fn = os.path.join(plotdir, pltname.format(xname))
    plt.tight_layout()
    plt.savefig(fn)
    print(f"Saving {fn}")

    return


def plot_histories(
    qs,
    t_table,
    labels,
    ylabels,
    plot_label=None,
    color_array=None,
    row_mask=None,
    lgnd_label="#{}",
    pltname="History_{}_step_{}.png",
    yscale="",
    reverse=False,
    check_step=5000,
    xlimlo=0.5,
    xlimhi=14,
    step=300,
    plotdir="./",
    plotsubdir="DiskBulge_Histories",
):
    plotdir = os.path.join(plotdir, plotsubdir)
    nrow, ncol = get_nrow_ncol(len(qs))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))
    # expect all q to be same length
    colors = cm.coolwarm(np.linspace(0, 1, len(qs[0])))
    # print(len(colors))
    indices = np.linspace(0, len(qs[0]) - 1, len(qs[0]), dtype=int)
    if not plot_label:  # assume plot_label, color_array, indx_array supplied together
        color_array = indices
    if row_mask is None:
        row_mask = np.ones(len(qs[0]), dtype=bool)
    sort_array = np.argsort(color_array[row_mask])
    if reverse:
        sort_array = sort_array[::-1]
        print(len(color_array[row_mask]), color_array[row_mask][sort_array][::check_step])

    for ax, q, ylabel in zip_longest(ax_all.flat, qs, ylabels):
        if ylabel is None:
            ax.set_visible(False)
            continue
        if len(color_array[row_mask]) != len(q[row_mask]):
            print("oops: array mismatch")
        for n, (h, c, i) in enumerate(
            zip(
                q[row_mask][sort_array][::step],
                color_array[row_mask][sort_array][::step],
                indices[row_mask][::step],
            )
        ):
            if n == 0:
                __ = ax.plot(t_table, h, color=colors[i], label=lgnd_label.format(c))
            elif n == int(len(q) / step):
                __ = ax.plot(t_table, h, color=colors[i], label=lgnd_label.format(c))
            else:
                __ = ax.plot(t_table, h, color=colors[i])

        ax.set_xlim(xlimlo, xlimhi)
        ax.set_ylabel(ylabel)
        if yscale == "log" and ("SMH" in ylabel or "SFH" in ylabel or "SFR" in ylabel):
            ax.set_yscale("log")
        ax.set_xlabel("$t$ (Gyr)")
        ax.legend(loc="best")
    fig.suptitle("Sample Histories")

    xname = "_".join(labels)
    if plot_label:
        xname = "_".join([xname, plot_label])
    if yscale == "log":
        xname = "_".join([xname, yscale])
    fn = os.path.join(plotdir, pltname.format(xname, step))
    plt.tight_layout()
    plt.savefig(fn)
    print(f"Saving {fn}")

    return


def plot_q_with_cuts(
    q,
    zvalues,
    redshifts,
    cut_array,
    cuts,
    dz=0.1,
    cut_labels=("{{}} $\\leq$ {:.0f}", "{{}} $\\geq$ {:.0f}"),
    colors=("r", "blue"),
    xlabel="B/T",
    cut_name="$\\log_{10}(sSFR/yr^{-1})$",
    pltname="BT_cut_on_{}.png",
    yscale="log",
    xscale="",
    bins=50,
    xname="log_sSFR",
    plotdir="./",
    plotsubdir="DiskBulge_Histograms",
    lgnd_title="",
):
    plotdir = os.path.join(plotdir, plotsubdir)
    nrow, ncol = get_nrow_ncol(len(zvalues))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))

    for ax, z in zip_longest(ax_all.flat, zvalues):
        zmask = (z - dz <= redshifts) & (z + dz >= redshifts)
        zlabel = f"${max(0., z-dz):.1f} \\leq z \\leq {z+dz:.1f}$"
        # apply row mask to arrays
        q_z = q[:, zmask]
        cut_array_z = cut_array[:, zmask]
        # print(cut_array_z.shape, q_z.shape)
        for n, (cut, cut_label, color) in enumerate(zip(cuts, cut_labels, colors)):
            cut_mask = (cut_array_z <= cut) if n == 0 else (cut_array_z >= cut)
            label = cut_label.format(cut).format(cut_name)
            # print(np.count_nonzero(cut_mask), q_z[cut_mask].shape)
            ax.hist(q_z[cut_mask], bins=bins, color=color, label=label, alpha=0.4, density=True)
        if yscale == "log":
            ax.set_yscale("log")
        if xscale == "log":
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PDF")
        ax.legend(loc="best", title=zlabel + lgnd_title)

    fn = os.path.join(plotdir, pltname.format(xname))
    plt.tight_layout()
    print(f"Saving {fn}")
    plt.savefig(fn)

    return


def plot_q1_q2(
    q1,
    q2,
    zvalues,
    redshifts,
    cut_array,
    cut_lo,
    cut_hi,
    dz=0.1,
    cut_label="{:.1f} $\\leq$ {{}} $\\leq$ {:.1f}",
    qlabels=("Bulge", "Disk"),
    cut_at_z0=True,
    cut_name="$\\log_{10}(M^*_{z=0}/M_\\odot)$",
    colors=("r", "blue"),
    xlabel="sSFR $(yr^{-1})$",
    xname="log_M0_{:.1f}_{:.1f}",
    pltname="sSFR_cut_on_{}.png",
    yscale="log",
    xscale="log",
    bins=50,
    plotdir="./",
    plotsubdir="DiskBulge_Histograms",
    lgnd_title="",
):
    plotdir = os.path.join(plotdir, plotsubdir)
    nrow, ncol = get_nrow_ncol(len(zvalues))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))

    clabel = cut_label.format(cut_lo, cut_hi).format(cut_name)
    for ax, z in zip_longest(ax_all.flat, zvalues):
        zmask = (z - dz <= redshifts) & (z + dz >= redshifts)
        zlabel = f"${max(0., z-dz):.1f} \\leq z \\leq {z+dz:.1f}$"
        # apply row mask to arrays
        q1_z = q1[:, zmask]
        q2_z = q2[:, zmask]
        cut_array_z = cut_array[:, zmask]
        # print(q1_z.shape, cut_array_z.shape)
        if cut_at_z0:  # cut on value at z=0
            cut_mask = (cut_array_z[:, -1] >= cut_lo) & (cut_array_z[:, -1] < cut_hi)
            # now broadcast back to 2-d mask
            cut_mask = np.broadcast_to(cut_mask, (np.count_nonzero(zmask), len(cut_mask))).T
        else:  # cut on values in redshift range
            cut_mask = (cut_array_z >= cut_lo) & (cut_array_z < cut_hi)
        print(z, np.count_nonzero(cut_mask))
        for q_z, qlabel, color in zip([q1_z[cut_mask], q2_z[cut_mask]], qlabels, colors):
            ax.hist(q_z, bins=bins, color=color, label=qlabel, alpha=0.4, density=True)

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_xlim(
            max(np.min(bins), min(np.min(q1_z[cut_mask]), np.min(q1_z[cut_mask])) * 0.5),
            min(np.max(bins), max(np.max(q1_z[cut_mask]), np.max(q1_z[cut_mask]))) * 2.0,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PDF")
        ax.legend(loc="best", title="\n".join([zlabel, clabel]))

    fn = os.path.join(plotdir, pltname.format(xname.format(cut_lo, cut_hi)))
    plt.tight_layout()
    plt.savefig(fn)
    print(f"Saving {fn}")

    return


def plot_q1_vs_q2(
    qx,
    qy,
    zvalues,
    redshifts,
    color_array,
    dz=0.1,
    ymin=-14,
    ymax=-9,
    xmin=7,
    xmax=12,
    cbar_title="B/T",
    ylabel="$\\log_{10}(sSFR/yr^{-1})$",
    xlabel="$\\log_{10}(M^*/M_\\odot)$",
    cmap="jet",
    N=1000,
    pltname="sSFR_vs_Mstar_{}.png",
    yscale="linear",
    xscale="linear",
    xname="",
    plotdir="./",
    plotsubdir="DiskBulge_Scatter",
    title="",
    label_x=0.05,
    label_y=0.05,
):
    plotdir = os.path.join(plotdir, plotsubdir)
    nrow, ncol = get_nrow_ncol(len(zvalues))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))

    # clabel = cut_label.format(cut_lo, cut_hi).format(cut_name)
    for ax, z in zip_longest(ax_all.flat, zvalues):
        zmask = (z - dz <= redshifts) & (z + dz >= redshifts)
        zlabel = f"${max(0., z-dz):.1f} \\leq z \\leq {z+dz:.1f}$"
        # apply row mask to arrays
        qx_z = jnp.ravel(qx[:, zmask])
        qy_z = jnp.ravel(qy[:, zmask])
        color_array_z = jnp.ravel(color_array[:, zmask])
        # subsample
        Nobj, idx = get_subsample(len(qx_z), N)
        print(np.min(qx_z[idx]), np.max(qx_z[idx]), np.min(qy_z[idx]), np.max(qy_z[idx]))
        im = ax.scatter(qx_z[idx], qy_z[idx], c=color_array_z[idx], cmap=cmap)
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        qymin = jnp.min(qy_z[idx])
        ymin = ymin if qymin < ymin else 1.1 * qymin
        qymax = jnp.max(qy_z[idx])
        ymax = 0.92 * qymax if qymax > ymax else ymax
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="y", labelrotation=90)
        ax.set_ylabel(ylabel, labelpad=0.0)
        ax.text(label_x, label_y, zlabel, transform=ax.transAxes)

    fn = os.path.join(plotdir, pltname.format(xname))
    cbar = fig.colorbar(im, ax=ax_all, location="right", fraction=0.1)
    # cbar.ax.set_title(cbar_title, rotation=90)
    cbar.set_label(cbar_title, rotation=90, labelpad=3.0, y=0.5, fontsize=14)
    fig.suptitle(title, y=0.97)
    # plt.tight_layout()
    plt.savefig(fn)
    print(f"Saving {fn}")

    return


def plot_q_vs_q_x_at_z_scatter(
    q,
    q_x,
    ylabel,
    color_array,
    zindexes,
    zs,
    cbar_title="$\\epsilon_{bulge}$",
    xlabel="tcrit_bulge",
    cmap="jet",
    N=1000,
    yscale="linear",
    xscale="linear",
    pltname="{}_{}.png",
    bins=50,
    xname="",
    title="",
    plotdir="./",
    plotsubdir="DiskBulge_Scatter",
    lgnd_title="",
):
    plotdir = os.path.join(plotdir, plotsubdir)
    nrow, ncol = get_nrow_ncol(len(zs))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), sharex=True, sharey=True)

    for ax, zidx, z in zip_longest(ax_all.flat, zindexes, zs):
        zlabel = f"$z = {z:.2f}$"
        color_array_z = color_array[:, zidx]
        # subsample
        Nobj, idx = get_subsample(len(q_x), N)
        # print(np.min(q_x[idx]), np.max(q_x[idx]), np.min(q[idx]), np.max(q[idx]))
        im = ax.scatter(q_x[idx], q[idx], c=color_array_z[idx], cmap=cmap)
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        # ax.set_ylabel('$dN/dlog_{10}(M^*/M_\\odot)$')
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", title=zlabel)

    fn = os.path.join(plotdir, pltname.format(ylabel, xname))
    cbar = fig.colorbar(im, ax=ax_all, location="right", fraction=0.1)
    # cbar.ax.set_title(cbar_title, rotation=90)
    cbar.set_label(cbar_title, rotation=90, labelpad=3.0, y=0.5, fontsize=14)
    fig.suptitle(title, y=0.97)
    # plt.tight_layout()
    plt.savefig(fn)
    print(f"Saving {fn}")

    return


def plot_q_vs_qx_at_z_profile(
    qs,
    qx,
    ylabels,
    zindexes,
    zs,
    yscale="linear",
    xscale="linear",
    pltname="{}_{}.png",
    bins=50,
    xlabel="$\\log_{10}(M^*/M_\\odot)$",
    xname="",
    title="",
    plotdir="./",
    Ylabel="$\\langle {} \\rangle$",
    plotsubdir="DiskBulge_Profiles",
    lgnd_title="",
    colors=("b", "g", "c", "r", "darkorange", "m"),
    errors=(True, True, True, True, False, False),
    qs_depends_z=False,
):
    plotdir = os.path.join(plotdir, plotsubdir)
    nrow, ncol = get_nrow_ncol(len(qs))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))

    for ax, q, ylabel, error in zip_longest(ax_all.flat, qs, ylabels, errors):
        if q is None:
            continue

        for zidx, z, c in zip(zindexes, zs, colors):
            zlabel = f"$z = {z:.2f}$"
            xvals = qx[:, zidx]  # x values at z
            xmeans, _, _ = binned_statistic(xvals, xvals, bins=bins)
            yvals = q[:, zidx] if qs_depends_z else q
            ymeans, _, _ = binned_statistic(xvals, yvals, bins=bins)
            std, _, _ = binned_statistic(xvals, yvals, bins=bins, statistic="std")
            ax.plot(xmeans, ymeans, label=zlabel, color=c)
            if error:
                ax.fill_between(xmeans, ymeans - std, ymeans + std, alpha=0.3, color=c)

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(Ylabel.format(ylabel))
        ax.legend(loc="best")

    fn = os.path.join(plotdir, pltname.format(xname))
    fig.suptitle(title, y=0.97)
    # plt.tight_layout()
    plt.savefig(fn)
    print(f"Saving {fn}")

    return


def plot_eff_sigmoids(
    tarr,
    f_tcrit,
    f_early,
    f_late,
    Dt,
    k_inv=6.0,
    yscale="linear",
    xscale="linear",
    ylabel="$\\epsilon_{bulge}$",
    xlabel="t",
    pltname="eff_bulge.png",
    xname="",
    title="",
    plotdir="./",
    plotsubdir="DiskBulge_Eff",
    lgnd_title="",
):
    plotdir = os.path.join(plotdir, plotsubdir)

    nrow, ncol = get_nrow_ncol(len(f_early))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), sharex=True, sharey=True)

    for ax, early, late in zip(ax_all.flat, f_early, f_late):
        title = f"$f_{{early}} = {early:.2f}, f_{{late}} = {late:.2f}$"
        for thalf in f_tcrit:
            for dt in Dt:
                tw_h = dt / k_inv
                yvals = dbk._tw_sigmoid(tarr, thalf, tw_h, early, late)
                ax.plot(tarr, yvals, label=f"$t_c$ = {thalf:.1f}, $dt$ = {dt:.1f}")

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # ax.set_ylim()
        ax.legend(loc="best", ncol=2, fontsize=8)

    fn = os.path.join(plotdir, pltname.format(xname))
    plt.savefig(fn)
    print(f"Saving {fn}")

    return


def plot_q_profile(ax_all, xs, qs, binz, label, error=True):
    for ax, x, q, bins in zip(ax_all.flat, xs, qs, binz):
        xmeans, _, _ = binned_statistic(x, x, bins=bins)
        # print(xmeans)
        ymeans, _, _ = binned_statistic(x, q, bins=bins)
        std, _, _ = binned_statistic(x, q, bins=bins, statistic="std")
        ax.plot(xmeans, ymeans, label=label)
        if error:
            ax.fill_between(xmeans, ymeans - std, ymeans + std, alpha=0.3)

    return


def plot_qs_zprofiles(
    ax_all,
    xs,
    qs,
    binz,
    zindexes,
    zs,
    x_depends_z=True,
    colors=("b", "g", "c", "r", "darkorange", "m"),
    error=True,
):
    for zidx, z, c in zip(zindexes, zs, colors):
        zlabel = f"$z = {z:.2f}$"
        xvals = [x[:, zidx] for x in xs] if x_depends_z else [x for x in xs]  # x values at z
        yvals = [q[:, zidx] for q in qs]

        plot_q_profile(ax_all, xvals, yvals, binz, zlabel, error=error)


# profiles for list of quantities with list of x dependence at chosen z values
def plot_qs_profiles_for_zvals(
    xs,
    qs,
    binz,
    labels,
    zindexes,
    zs,
    xlabels,
    colors=("b", "g", "c", "r", "darkorange", "m"),
    plotdir="./",
    plotsubdir="profiles",
    error=True,
    pltname="qs_z_{}.png",
    title="",
    xname="",
):
    plotdir = os.path.join(plotdir, plotsubdir)
    nrow, ncol = get_nrow_ncol(len(labels))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))

    plot_qs_zprofiles(ax_all, xs, qs, binz, zindexes, zs, colors=colors, error=error)

    for ax, label, xlabel in zip(ax_all.flat, labels, xlabels):
        ax.legend(loc="best", title=label)
        ax.set_xlabel(xlabel)

    fn = os.path.join(plotdir, pltname.format(xname))
    fig.suptitle(title, y=0.97)
    # plt.tight_layout()
    plt.savefig(fn)
    print(f"Saving {fn}")


# comparison profiles for lists of quantities with lists of x dependence
def plot_qs_profiles(
    xs,
    fs,
    labels,
    lxs,
    lys,
    bins,
    plotdir="./",
    pltname="generate_fbulge_{}.png",
    plotsubdir="profiles",
    title="",
    xname="",
):
    plotd = os.path.join(plotdir, plotsubdir)
    nrow, ncol = get_nrow_ncol(len(xs[0]))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))

    for x, f, label in zip(xs, fs, labels):  # loop over curves for each plot
        plot_q_profile(ax_all, x, f, bins, label, error=True)  # loop over plots

    for ax, lx, ly in zip(ax_all.flat, lxs, lys):
        ax.legend(loc="best")
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)

    fn = os.path.join(plotd, pltname.format(xname))
    fig.suptitle(title, y=0.97)
    plt.savefig(fn)
    print(f"Saving {fn}")

    return


def plot_sigmoid_2d(
    x,
    x0,
    y,
    y0,
    kpairs,
    ymin,
    ymax,
    ylabel="$\\log_{10}(M^*/M_\\odot)$",
    xlabel="$\\log_{10}(sSFR/yr^{-1})$",
    edgecolor="royalblue",
    ytit=0.95,
    ysuptit=0.98,
    alpha=0.5,
    cmap="coolwarm",
    contour=False,
    cmapc="coolwarm_r",
    plotdir="./",
    plotsubdir="sigmoids",
    xname="",
    title="",
    pltname="sigmoid_2d_{}.png",
):
    plotdir = os.path.join(plotdir, plotsubdir)
    nrow, ncol = get_nrow_ncol(len(kpairs))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), subplot_kw={"projection": "3d"})
    for ax, (kx, ky) in zip(ax_all.flat, kpairs):
        z = dbk._sigmoid_2d(x, x0, y, y0, kx, ky, ymin, ymax)
        label = f"$kx = {kx:.1f}, ky={ky:.1f}$"
        ax.plot_surface(
            x,
            y,
            z,
            edgecolor=edgecolor,
            cmap=cmap,
            linewidth=0,
            antialiased=False,
            # rstride=rstride, cstride=cstride,
            alpha=alpha,
        )

        ax.set_title(label, y=ytit)
        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph
        x_min = np.min(x)
        y_max = np.max(y)
        if contour:
            ax.contourf(x, y, z, zdir="x", offset=x_min, cmap=cmapc)
            ax.contourf(x, y, z, zdir="y", offset=y_max, cmap=cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # ax.set(xlim=(-40, 40), ylim=(-40, 40))
    #   xlabel='X', ylabel='Y', zlabel='Z')

    fn = os.path.join(plotdir, pltname.format(xname))
    fig.suptitle(title, y=ysuptit)
    plt.tight_layout()
    plt.savefig(fn)
    print(f"Saving {fn}")

    return


def plot_q_vs_xs_color_scatter(
    q,
    xs,
    color_arrays,
    qlabel,
    clabels=("$\\log_{10}(sSFR/yr^{-1})$", "$\\log_{10}(M^*/M_\\odot)$"),
    xlabels=("$\\log_{10}(M^*/M_\\odot)$", "$\\log_{10}(sSFR/yr^{-1})$"),
    N=2000,
    wspace=0.5,
    pltname="{}_vs_sSFR_Mstar_{}.png",
    xname="",
    cmaps=("jet_r", "jet"),
    plotdir="./",
    plotsubdir="Fbulge",
    title="",
):
    plotdir = os.path.join(plotdir, plotsubdir)
    nrow, ncol = get_nrow_ncol(len(xs))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))

    for ax, x, c, xlabel, clabel, cmap in zip(ax_all.flat, xs, color_arrays, xlabels, clabels, cmaps):
        Nobj, idx = get_subsample(len(q), N)
        # print(np.min(q_x[idx]), np.max(q_x[idx]), np.min(q[idx]), np.max(q[idx]))
        im = ax.scatter(x[idx], q[idx], c=c[idx], cmap=cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(qlabel)
        cbar = fig.colorbar(im, ax=ax, location="right", fraction=0.1)
        cbar.set_label(clabel, rotation=90, labelpad=3.0, y=0.5, fontsize=14)

    fn = os.path.join(plotdir, pltname.format(qlabel, xname))
    plt.subplots_adjust(wspace=wspace)
    fig.suptitle(title, y=0.97)
    # plt.tight_layout()
    plt.savefig(fn)
    print(f"Saving {fn}")

    return


def plot_comparison_profiles_at_zvalues(
    xlist,
    qlist,
    binz,
    labels,
    zindexes,
    zs,
    xlabel,
    ylabel,
    plotdir=plotdir,
    title="",
    xname="",
    error=True,
    pltname="check_eff_bulge_{}.png",
    plotsubdir="Fbulge",
):
    plotdir = os.path.join(plotdir, plotsubdir)
    nrow, ncol = get_nrow_ncol(len(zs))
    fig, ax_all = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))

    for x, q, bins, label in zip(xlist, qlist, binz, labels):
        xvals = [x[:, zidx] for zidx in zindexes]
        yvals = [q[:, zidx] for zidx in zindexes]
        plot_q_profile(ax_all, xvals, yvals, bins, label, error=error)

    zlabels = [f"$z = {z:.2f}$" for z in zs]
    for ax, zlabel in zip(ax_all.flat, zlabels):
        ax.legend(loc="best", title=zlabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fn = os.path.join(plotdir, pltname.format(xname))
    fig.suptitle(title, y=0.97)
    # plt.tight_layout()
    plt.savefig(fn)
    print(f"Saving {fn}")

    return
