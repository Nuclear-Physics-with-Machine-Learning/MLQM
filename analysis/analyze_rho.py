import sys, os
from matplotlib import pyplot as plt
import numpy
import pathlib

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:blue", "tab:orange"]) 

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from scipy import interpolate


def parse_file(file_path):

    with open(file_path, 'r') as f:
        lines = f.readlines()

    if "GFMC" in file_path:
        lines = lines[2:]

    r    = []
    rho  = []
    drho = []

    for l in lines:
        tokens = l.rstrip('\n').split()
        r   .append(float(tokens[0]))
        rho .append(float(tokens[1]))
        drho.append(float(tokens[2]))

    return numpy.asarray(r), numpy.asarray(rho), numpy.asarray(drho)


for nucleus in ['he4', 'h3', 'h2']:
    r_ann, rho_ann, drho_ann    = parse_file(f'logs/rho_{nucleus}_ANN.dat')
    r_gfmc, rho_gfmc, drho_gfmc = parse_file(f'logs/rho_{nucleus}_GFMC.dat')

    x_max = 4.0

    ann_mask = r_ann <= x_max
    gfmc_mask = r_gfmc <= x_max


    tck,u     = interpolate.splprep( [r_gfmc[gfmc_mask], rho_gfmc[gfmc_mask]],s = 0 )
    xnew,ynew = interpolate.splev( numpy.linspace( 0, 1, 500 ), tck,der = 0)


    plt.figure(figsize=(16,9))

    # Create a spline for the upper and lower uncertainty window
    tck_upper, _  = interpolate.splprep( [r_gfmc[gfmc_mask], rho_gfmc[gfmc_mask] + drho_gfmc[gfmc_mask]],s = 0 )
    tck_lower, _  = interpolate.splprep( [r_gfmc[gfmc_mask], rho_gfmc[gfmc_mask] - drho_gfmc[gfmc_mask]],s = 0 )
    xnew,ynew_upper = interpolate.splev( numpy.linspace( 0, 1, 500 ), tck_upper,der = 0)
    xnew,ynew_lower = interpolate.splev( numpy.linspace( 0, 1, 500 ), tck_lower,der = 0)
    plt.fill_between(xnew, 
        ynew_lower,
        ynew_upper,
        alpha=0.75)
    plt.plot(xnew, ynew, label="GFMC", lw=3)
    plt.errorbar(r_ann[ann_mask], rho_ann[ann_mask], 
        yerr = drho_ann[ann_mask], label="ANN",
        ls="none",
        marker='o',
        ms = 7,
        # color='orange'
        )
    # plt.fill_between(r_ann[ann_mask], 
    #     rho_ann[ann_mask] - drho_ann[ann_mask], 
    #     rho_ann[ann_mask] + drho_ann[ann_mask], 
    #     label="ANN",
    #     # ls="none",
    #     # marker='s',
    #     color='orange',
    #     alpha=0.5)
    plt.xlim([0.0, x_max])


    # plt.yscale('log')
    plt.grid(True)
    plt.legend(fontsize=25)

    plt.xlabel("r (fm)", fontsize=35)
    plt.ylabel(r"$\rho_N~($fm${}^{-3})$", fontsize=35)

    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(30) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(30) 
    plt.tight_layout()

    ax1 = plt.gca()


    x_max = 4.0
    x_min = 2.0

    ann_mask = numpy.logical_and(r_ann >= x_min, r_ann <= x_max)
    gfmc_mask = numpy.logical_and(r_gfmc >= x_min, r_gfmc <= x_max)

    # bottom, left, width, height
    ax2 = plt.axes([0,2.,2.,0.2])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax1, [0.35,0.25,0.5,0.5])
    ax2.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.
    mark_inset(ax1, ax2, loc1=3, loc2=1, fc="black", alpha=0.5, ec='black')

    ax2.fill_between(r_gfmc[gfmc_mask], 
        rho_gfmc[gfmc_mask] - drho_gfmc[gfmc_mask], 
        rho_gfmc[gfmc_mask] + drho_gfmc[gfmc_mask],
        alpha=0.75)
    ax2.plot(r_gfmc[gfmc_mask], rho_gfmc[gfmc_mask],lw=3)
    ax2.errorbar(r_ann[ann_mask], rho_ann[ann_mask], 
        yerr = drho_ann[ann_mask], 
        # label="Error x10",
        ls="none",
        marker='o',
        ms=7,
        # color='orange'
        )
    # ax2.fill_between(r_ann[ann_mask], 
    #     rho_ann[ann_mask] - drho_ann[ann_mask], 
    #     rho_ann[ann_mask] + drho_ann[ann_mask], 
    #     label="ANN",
    #     # ls="none",
    #     # marker='s',
    #     color='orange',
    #     alpha=0.5)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(25) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(25) 
    plt.yscale('log')
    plt.xlim([x_min, x_max])
    plt.ylim([1e-5, 2e-2])
    # plt.legend(fontsize=25)
    ax2.grid(True)

    plt.savefig(f"rho_{nucleus}.pdf")



    plt.show()

    # break


#     # specify integer or one of preset strings, e.g.
#     #tick.label.set_fontsize('x-small') 
#     # tick.label.set_rotation('vertical')

# plt.savefig("energy_convergence.pdf")
# plt.show()
