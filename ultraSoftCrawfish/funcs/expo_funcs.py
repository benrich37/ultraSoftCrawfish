import matplotlib.pyplot as plt
from ultraSoftCrawfish.funcs.pcohp_funcs import get_pcohp, get_ipcohp, get_ipcoop, get_pcoop
from ultraSoftCrawfish.helpers.rs_helpers import opj
from os.path import basename


def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])


def expo_directionality(idcs0, idcs1, label0, label1, data, suptitle, Erange=None, showsum=False):
    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(7,10))
    lfig = plt.figure()
    c1 = "black"
    c2 = "orange"
    c3 = "green"
    Erange, pcohp = get_pcohp(idcs0, idcs1, data=data, Erange=Erange)
    Erange, dpcohp0 = get_pcohp(idcs0, idcs1, data=data, Erange=Erange, directional=True, iso_acceptance=False)
    Erange, dpcohp1 = get_pcohp(idcs1, idcs0, data=data, Erange=Erange, directional=True, iso_acceptance=False)
    Es, ipcohp = get_ipcohp(idcs0, idcs1, data=data, as_array=True)
    Es, dipcohp0 = get_ipcohp(idcs0, idcs1, data=data, as_array=True, directional=True, iso_acceptance=False)
    Es, dipcohp1 = get_ipcohp(idcs1, idcs0, data=data, as_array=True, directional=True, iso_acceptance=False)
    l1, = ax[0].plot(Erange, pcohp, c=c1, label=f"pCOHP {label0} " + r"$\rightarrow$" + f"{label1}")
    l2, = ax[0].plot(Erange, dpcohp0, c=c2, label=f"dpCOHP {label0}")
    l3, = ax[0].plot(Erange, dpcohp1, c=c3, label=f"dpCOHP {label1}")
    if showsum:
        ax[0].plot(Erange, (dpcohp1+dpcohp0), c="pink", label=f"dpCOHP {label0} + dpCOHP {label1}")
    l4, = ax[0].plot([], [], c=c1, linestyle="dashed", label=r"$\int$ pCOHP")
    lfig.legend([l1, l2, l3, l4], [
        f"pCOHP {label0} " + r"$\rightarrow$" + f"{label1}", f"dpCOHP {label0}", f"dpCOHP {label1}", r"$\int$ pCOHP"
    ], loc="center")
    tax = ax[0].twinx()
    tax.plot(Es, ipcohp, c=c1, linestyle="dashed")
    tax.plot(Es, dipcohp0, c=c2, linestyle="dashed")
    tax.plot(Es, dipcohp1, c=c3, linestyle="dashed")
    if showsum:
        tax.plot(Es, (dipcohp1+dipcohp0), c="pink", linestyle="dashed")
    tax.set_ylabel("ipCOHP (Ha)")
    align_yaxis(ax[0], tax)
    ##

    Erange, pcoop = get_pcoop(idcs0, idcs1, data=data, Erange=Erange)
    Erange, dpcoop0 = get_pcoop(idcs0, idcs1, data=data, Erange=Erange, directional=True, iso_acceptance=False)
    Erange, dpcoop1 = get_pcoop(idcs1, idcs0, data=data, Erange=Erange, directional=True, iso_acceptance=False)
    Es, ipcoop = get_ipcoop(idcs0, idcs1, data=data, as_array=True)
    Es, dipcoop0 = get_ipcoop(idcs0, idcs1, data=data, as_array=True, directional=True, iso_acceptance=False)
    Es, dipcoop1 = get_ipcoop(idcs1, idcs0, data=data, as_array=True, directional=True, iso_acceptance=False)
    ax[2].plot(Erange, pcoop, c=c1)
    ax[2].plot(Erange, dpcoop0, c=c2)
    ax[2].plot(Erange, dpcoop1, c=c3)
    if showsum:
        ax[2].plot(Erange, (dpcoop1+dpcoop0), c="pink", label=f"dpCOOP {label1}+dpCOOP {label0}")
    tax = ax[2].twinx()
    tax.plot(Es, ipcoop, c=c1, linestyle="dashed")
    tax.plot(Es, dipcoop0, c=c2, linestyle="dashed")
    tax.plot(Es, dipcoop1, c=c3, linestyle="dashed")
    if showsum:
        tax.plot(Es, (dipcoop1+dipcoop0), c="pink", linestyle="dashed")
    tax.set_ylabel("ipCOOP (a.u.)")
    align_yaxis(ax[2], tax)
    ########################
    Erange, pcohp = get_pcohp(idcs0, idcs1, data=data, Erange=Erange)
    Erange, dpcohp0 = get_pcohp(idcs0, idcs1, data=data, Erange=Erange, directional=True, iso_acceptance=True)
    Erange, dpcohp1 = get_pcohp(idcs1, idcs0, data=data, Erange=Erange, directional=True, iso_acceptance=True)
    Es, ipcohp = get_ipcohp(idcs0, idcs1, data=data, as_array=True)
    Es, dipcohp0 = get_ipcohp(idcs0, idcs1, data=data, as_array=True, directional=True, iso_acceptance=True)
    Es, dipcohp1 = get_ipcohp(idcs1, idcs0, data=data, as_array=True, directional=True, iso_acceptance=True)
    ax[1].plot(Erange, pcohp, c=c1, label=f"pCOHP {label0} " + r"$\rightarrow$" + f"{label1}")
    ax[1].plot(Erange, dpcohp0, c=c2)
    ax[1].plot(Erange, dpcohp1, c=c3)
    if showsum:
        ax[1].plot(Erange, (dpcohp1+dpcohp0), c="pink",)
    ax[1].plot([], [], c=c1, linestyle="dashed")
    tax = ax[1].twinx()
    tax.plot(Es, ipcohp, c=c1, linestyle="dashed")
    tax.plot(Es, dipcohp0, c=c2, linestyle="dashed")
    tax.plot(Es, dipcohp1, c=c3, linestyle="dashed")
    if showsum:
        tax.plot(Es, (dipcohp1+dipcohp0), c="pink", linestyle="dashed")
    tax.set_ylabel("ipCOHP (Ha)")
    align_yaxis(ax[1], tax)
    ##
    Erange, pcoop = get_pcoop(idcs0, idcs1, data=data, Erange=Erange)
    Erange, dpcoop0 = get_pcoop(idcs0, idcs1, data=data, Erange=Erange, directional=True, iso_acceptance=True)
    Erange, dpcoop1 = get_pcoop(idcs1, idcs0, data=data, Erange=Erange, directional=True, iso_acceptance=True)
    Es, ipcoop = get_ipcoop(idcs0, idcs1, data=data, as_array=True)
    Es, dipcoop0 = get_ipcoop(idcs0, idcs1, data=data, as_array=True, directional=True, iso_acceptance=True)
    Es, dipcoop1 = get_ipcoop(idcs1, idcs0, data=data, as_array=True, directional=True, iso_acceptance=True)
    ax[3].plot(Erange, pcoop, c=c1)
    ax[3].plot(Erange, dpcoop0, c=c2)
    ax[3].plot(Erange, dpcoop1, c=c3)
    if showsum:
        ax[3].plot(Erange, (dpcoop1+dpcoop0), c="pink")
    ax[3].plot([], [], c=c1, linestyle="dashed")
    tax = ax[3].twinx()
    tax.plot(Es, ipcoop, c=c1, linestyle="dashed")
    tax.plot(Es, dipcoop0, c=c2, linestyle="dashed")
    tax.plot(Es, dipcoop1, c=c3, linestyle="dashed")
    if showsum:
        tax.plot(Es, (dipcoop1+dipcoop0), c="pink", linestyle="dashed")

    align_yaxis(ax[3], tax)
    tax.set_ylabel("ipCOOP (a.u.)")

    ###################
    for i in range(4):
        ax[i].axvline(x=data.get_mu(), c="red", label=r"$\mu_{el}$")
    #ax[0].legend(loc="center left")
        # ax[i].legend(loc="right")
    ax[0].set_title(r"pCOHP ($\tilde{P}$)")
    ax[2].set_title(r"pCOOP ($\tilde{P}$)")
    ax[2].set_xlabel("E (Ha)")
    ax[0].set_ylabel("pCOHP (Ha)")
    ax[2].set_ylabel("pCOOP (a.u.)")
    #
    ax[1].set_title(r"pCOHP (A)")
    ax[3].set_title(r"pCOOP (A)")
    ax[3].set_xlabel("E (Ha)")
    ax[1].set_ylabel("pCOHP (Ha)")
    ax[3].set_ylabel("pCOOP (a.u.)")
    fig.suptitle(suptitle)
    calc_path = data.root
    savename = f"all_{basename(calc_path)}dir_anl_" + str(idcs0) + "_" + str(idcs1)
    fig.tight_layout()
    fig.savefig(opj(calc_path, f"{savename}.png"))
    lfig.savefig(opj(calc_path, f"legend-{savename}.png"))