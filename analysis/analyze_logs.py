import sys, os
from matplotlib import pyplot as plt
import numpy
import pathlib

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:blue", "tab:orange"]) 



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

    return step, energy, err


step_6, energy_6, err_6 =  parse_log_file(log_6_file)
step_4, energy_4, err_4 =  parse_log_file(log_4_file)

# n=2
# print(len(energy))
# moving_energy = moving_average(energy,2*n + 1)
# print(len(moving_energy))

# print(energy_4)
print(energy_6)

plt.figure(figsize=(16,9))
# plt.plot(step[n:-n], moving_energy, label="Pionless 4He")
plt.plot(step_4, energy_4, label=r"$\Lambda = 4~$fm${}^{-1}$")
plt.fill_between(step_4, energy_4-err_4, energy_4+err_4, alpha=0.5)

plt.plot(step_6, energy_6, label=r"$\Lambda = 6~$fm${}^{-1}$")
plt.fill_between(step_6, energy_6-err_6, energy_6+err_6, alpha=0.5)
plt.grid(True)
plt.xlabel(r"SR Iteration", fontsize=35)
plt.ylabel(r"Energy [MeV]", fontsize=35)
plt.legend(fontsize=35)

for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(30) 
for tick in plt.gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(30) 
    # specify integer or one of preset strings, e.g.
    #tick.label.set_fontsize('x-small') 
    # tick.label.set_rotation('vertical')

plt.tight_layout()
plt.savefig("energy_convergence.pdf")
plt.show()
