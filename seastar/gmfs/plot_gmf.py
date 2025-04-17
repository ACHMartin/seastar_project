#!/usr/bin/env python3

# Python scrip to plot a scatterometer Geophysical Model Function (GMF).
# It will create two plots.
# One plot shows the GMF value as a funcion of azimuth angle
# for different wind speeds and fixed incidence angle.
# The other plot shows the GMF value as a funcion of wind speed
# for different azimuth angles and fixed incidence angle.
# The incidence angle can be chosen below by changing iinc.
#
# (c) 2017 Anton Verhoef, KNMI Royal Netherlands Meteorological Institute
#
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
  sys.exit('Usage: '+os.path.basename(sys.argv[0])+' <binary GMF file>')

fname = sys.argv[1]
try:
  f = open(fname, "rt", encoding='utf-8')
except Exception:
  sys.exit('File not found')

f.close()

# Dimensions of GMF table
m = 250  # wind speed min/max = 0.2-50 (step 0.2) [m/s] --> 250 pts
n = 73   # dir min/max = 0-180 (step 2.5) [deg]   -->  73 pts
p = 51   # inc min/max = 16-66 (step 1) [deg]     -->  51 pts
gmf_table = np.fromfile(fname, dtype=np.float32)

# Remove head and tail
gmf_table = gmf_table[1:-1]

# To access the table as a three-dimensional Fortran-ordered m x n x p matrix,
# reshape it
gmf_table = gmf_table.reshape((m, n, p), order="F")

# Choose incidence angle to plot
iinc = 34 # incidence angle index
incidence = float(iinc) + 16.0

# First plot: sigma0 vs. direction
fig = plt.figure(figsize=(11.69,8.27))
ax1 = fig.add_axes([0.07, 0.1, 0.48, 0.8])
ax1.set_title(fname + ', incidence angle ' + str(incidence))
ax1.set_xlabel('Wind direction (\u00b0)')
ax1.set_ylabel('sigma0 (dB)')
ax1.set_xlim([0,225])
ax1.set_xticks(range(0,225,45))
ax1.set_ylim([-40,-5])
ax1.grid()
pltarrx = [idir*2.5 for idir in range(n)]
for ispd in range(9,121,10):
  pltarry = [np.log10(gmf_table[ispd][idir][iinc])*10.0 for idir in range(n)]
  ax1.plot(pltarrx, pltarry, label=str((ispd+1)*0.2)+" m/s")
leg = ax1.legend(loc=1,prop={'size':9})

# Second plot: sigma0 vs. speed
ax2 = fig.add_axes([0.6, 0.1, 0.37, 0.8])
ax2.set_xlabel('Wind speed (m/s)')
#ax2.set_xscale('log')
ax2.set_xlim([0,25])
ax2.set_ylim([-40,-5])
ax2.grid()
pltarrx = [(ispd+1)*0.2 for ispd in range(m)]
for idir in range(0,73,9):
  pltarry = [np.log10(gmf_table[ispd][idir][iinc])*10.0 for ispd in range(m)]
  ax2.plot(pltarrx, pltarry, label=str(idir*2.5)+"\u00b0")
leg = ax2.legend(loc=4,prop={'size':9})

plt.savefig(fname + '.png')
print('Image saved in ' + fname + '.png')
plt.show()
