import sys, os
from matplotlib import pyplot as plt
import numpy
import pathlib

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:blue", "tab:orange"]) 

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

def moving_average(a, n=3) :
    ret = numpy.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

log_4_file = pathlib.Path("logs/pionless_4_nucleus_2.log")
log_6_file = pathlib.Path("logs/pionless_6_nucleus_2.log")

def parse_log_file(file_path):

    with open(file_path, 'r') as f:
        lines = f.readlines()


    runs = []
    for l in lines:
        if "INFO step =" in l:
            if 'energy_jf' in l: continue
            runs.append(l)

    step   = []
    energy = []
    err    = []
    for l in runs:
        tokens = l.rstrip('\n').split(' ')
        # 2020-07-04 12:50:07,259 INFO step = 8, energy = 2.710, err = 0.043
        step.append(int(tokens[5].rstrip(',')))
        energy.append(float(tokens[8].rstrip(',')))
        err.append(float(tokens[11].rstrip(',')))

    step = list(range(len(energy)))
    energy = numpy.asarray(energy)
    err = numpy.asarray(err)

    return numpy.asarray(step), numpy.asarray(energy), numpy.asarray(err)


step_6, energy_6, err_6 =  parse_log_file(log_6_file)
step_4, energy_4, err_4 =  parse_log_file(log_4_file)

# mask_fake = step_6 > 1300
# print(numpy.sum(mask_fake))
# fake = numpy.arange(numpy.sum(mask_fake))
# print(fake)
# energy_6[mask_fake] = energy_6[mask_fake] + 0.001 * fake

# print(len(step_4))

# n=2
# print(len(energy))
# moving_energy = moving_average(energy,2*n + 1)
# print(len(moving_energy))

# print(energy_4)
# print(energy_6)

plt.figure(figsize=(16,9))
# plt.plot(step[n:-n], moving_energy, label="Pionless 4He")
plt.plot(step_4, energy_4, label=r"$\Lambda = 4~$fm${}^{-1}$")
plt.fill_between(step_4, energy_4-err_4, energy_4+err_4, alpha=0.5)

plt.plot(step_6, energy_6, label=r"$\Lambda = 6~$fm${}^{-1}$")
plt.fill_between(step_6, energy_6-err_6, energy_6+err_6, alpha=0.5)
plt.grid(True)
plt.xlabel(r"SR Iteration", fontsize=35)
plt.ylabel(r"Energy [MeV]", fontsize=35)

# Add a line for GFMC solution at 2.224
plt.plot(step_4, [-2.224,]*len(step_4), color="black", ls="--")
plt.legend(fontsize=35)

# matplotlib.pyplot.annotate("-2.224",[100, -2.224 + 0.1], fontsize=25)

for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(30) 
for tick in plt.gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(30) 
    # specify integer or one of preset strings, e.g.
    #tick.label.set_fontsize('x-small') 
    # tick.label.set_rotation('vertical')

plt.tight_layout()

x_max = 1750

plt.xlim([0,x_max])
plt.ylim([-2.5,3.8])
ax1 = plt.gca()

x_min = 1200


print(step_4)

step_4_mask = numpy.logical_and(step_4 >= x_min, step_4 <= x_max)
step_6_mask = numpy.logical_and(step_6 >= x_min, step_6 <= x_max)

# bottom, left, width, height
ax2 = plt.axes([-2,1800.,200,1.0])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.6,0.25,0.3,0.4])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=3, loc2=1, fc="black", alpha=0.5, ec='black')

step_4   = step_4[step_4_mask]
energy_4 = energy_4[step_4_mask]
err_4    = err_4[step_4_mask]

step_6   = step_6[step_6_mask]
energy_6 = energy_6[step_6_mask]
err_6    = err_6[step_6_mask]


plt.plot(step_4, energy_4, label=r"$\Lambda = 4~$fm${}^{-1}$")
plt.fill_between(step_4, energy_4-err_4, energy_4+err_4, alpha=0.5)

plt.plot(step_6, energy_6, label=r"$\Lambda = 6~$fm${}^{-1}$")
plt.fill_between(step_6, energy_6-err_6, energy_6+err_6, alpha=0.5)


plt.plot(step_4, [-2.224,]*len(step_4), color="black", ls="--", label="GFMC")

# plt.plot(step_6, energy_6, label=r"$\Lambda = 6~$fm${}^{-1}$")
# plt.fill_between(step_6, energy_6-err_6, energy_6+err_6, alpha=0.5)
plt.grid(True)
# plt.xlabel(r"SR Iteration", fontsize=35)
# plt.ylabel(r"Energy [MeV]", fontsize=35)
# plt.legend(fontsize=35)

# ax2.fill_between(r_ann[ann_mask], 
#     rho_ann[ann_mask] - drho_ann[ann_mask], 
#     rho_ann[ann_mask] + drho_ann[ann_mask], 
#     label="ANN",
#     # ls="none",
#     # marker='s',
#     color='orange',
#     alpha=0.5)
plt.locator_params(axis='y', nbins=5)
plt.locator_params(axis='x', nbins=6)

for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
for tick in plt.gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
# plt.yscale('log')
plt.xlim([x_min, x_max])
plt.ylim([-2.25,-2.05])
# plt.legend(fontsize=25)
# ax2.grid(True)


plt.savefig("energy_convergence.pdf")
plt.show()
