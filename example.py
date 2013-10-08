#!/usr/bin/env python

import numpy, cube, fnv
bohr = 1.889725989

# i/o
fdef = 'def.cube'  # charged defect supercell
fref = 'ref.cube'  # reference (neutral or bulk) supercell 
wf   = 'dwf.cube'  # defect wavefunction
is_charge = False  # is the input a charge density file?
out = 'pc.dat'     # output

# parameters
pdef = [0.0, 0.0, 0.0]      # cart coordinates of the defect
ecut = 20                   # in Ry   
q =  1
x = 0.        # weight of exp. tail
gamma = 0.6   # exp. tail
beta = 0.5    # gaussian width
eps =  1.96   # dielectric constant
axis =  2     # 0 -> x, 1 -> y, 2 -> z
plt = 1       # 0: n/a, 1: V, 2: rho

# fit the defect charge density
#fit = fnv.qfit(wf, pdef, axis)
#fit.fitting(q,x,gamma,beta)

# correction to the total energy
corr = fnv.correction(fdef, fref, pdef, axis, q, x, gamma, beta, ecut, eps, plt, is_charge, out)
corr.qmodel()
corr.energy()
#corr.convolve(0.2, is_charge)
