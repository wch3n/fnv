#!/usr/bin/env python
'''planar-average of a cube file
wei.chen@epfl.ch'''

import numpy as np
import sys

class Readcube:
    def __init__(self, filename, zdef=0):
        try:
            f = open(filename)
        except IOError:
            sys.exit('File not found.')

        [f.readline() for i in range(2)]
        # na: number of atoms
        na = int(f.readline().split()[0]) 
        # ng: 3D grid points, ns: spacing vector
        ng = np.array([[0,0,0]])
        ns = np.zeros((3,3))
        for i in range(3):
            s = f.readline().split()
            ng[:,i] = int(s[0])
            ns[i] = float(s[1]), float(s[2]), float(s[3])

        # read the positions
        pos = np.zeros((na,3))
        for i in range(na):
            s = f.readline().split()
            pos[i,:] = s[2:] 
        
        # real space lattice vector
        rvec = ns*ng.T

        dat = str2array(f.readlines())
        f.close()
 
        # comply to the cube format
        dat = dat.reshape(ng[0,:])
 
        self.na = na
        self.ng = ng
        self.ns = ns
        self.dat = dat
        self.rvec = rvec
        self.pos = pos
        self.zdef = zdef

def str2array(str):
    return np.fromstring(''.join(str), sep=' ')

class Readpot:
    def __init__(self, filename):
        try:
            f = open(filename)
        except IOError:
            sys.exit('File not found.')
        f.readline()
        head = f.readline().split()
        ng = [0,0,0]
        ng[0], ng[1], ng[2] = int(head[0]), int(head[1]), int(head[2])
        na = int(head[6])
        ntype = int(head[7]) # number of atom types
        head = f.readline().split()
        scale = float(head[1])
        # rvec: real-space lattice vector
        rvec = np.zeros([3,3])        
        for i in range(3):
            s = f.readline().split()
            rvec[i,:] = float(s[0]), float(s[1]), float(s[2])
        rvec *= scale
        [f.readline() for i in range(ntype+1)]
        # direct coordinates
        pos = np.zeros((na,3))
        for i in range(na):
            s = f.readline().split()
            pos[i,:] = s[1:4]
        dat = f.readlines()
        
        f.close()

        self.dat = dat
        self.na = na
        self.ng = ng
        self.rvec = rvec
        self.pos = pos

class Shift1d:
    def __init__(self, z, y, zdef):
        N = z.size
        Z = z.max()-z.min()
        # fractional zdef
        fzdef = zdef/Z

        # fftshift 
        # g: reciprocal vector
        g = np.fft.fftfreq(N)*N
        yg = np.fft.fft(y)
        ygs = yg*np.exp(-2j*np.pi*g*(0.5-fzdef))        
        ys = np.fft.ifft(ygs)

        # centering the defect
        self.zs = np.mgrid[-0.5:0.5:1./N]*Z
        self.fzdef = fzdef
        self.g = g
        self.ys = ys

if __name__ == "__main__":
    cube = Readcube(sys.argv[1], float(sys.argv[2]))
    ngrid = cube.ng[0]
    # print number of atoms, and fft grid
    print((cube.na, ngrid)) 

    dir = 2  # 0->x, 1->y, 2->z
    
    avg_1d = np.zeros(ngrid[dir])
    for i in range(ngrid[dir]):
        avg_1d[i] = np.average(cube.dat[:,:,i])
    
    zlen = np.linalg.norm(cube.rvec[dir,:])
    z = np.linspace(0, zlen, ngrid[dir],endpoint=False)

    if float(sys.argv[2]) == 0:
        dump = list(zip(z, avg_1d))
    else:
        shift = Shift1d(z, avg_1d, cube.zdef)
        dump = list(zip(z, shift.ys.real))

    np.savetxt(sys.argv[1].rsplit(".")[0]+"_1d.dat",dump)

    avg = np.average(avg_1d)
    print(avg)
