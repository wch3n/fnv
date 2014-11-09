# Wei Chen (wei.chen@epfl.ch) July 2011 GPL

"""Finite size correction for charged point defects according to 
Freysoldt et. al. (PRL 102, 016402, 2009)
"""

import numpy as np
import cube, time
import matplotlib.pylab as plt
from matplotlib import rc
from scipy.fftpack import fft, ifft, fftn, ifftn, fftshift, ifftshift
from scipy.special import erf, erfc
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.constants import physical_constants
from numpy import pi, ceil, exp 

class Correction(object):
    def __init__(self, filedef, fileref, pdef, axis=0 , q=1, \
                 x=0.54, gamma=2.0, beta=1.0, ecut=20.0, eps=1.0, plt=0, is_charge=False, out='V.dat'): 
        # read the two cube files (with electrostatic potentials) for the charged defect 
        # and the reference (neutral) calculations
        fdef = cube.readcube(filedef)
        if fileref == 'None':
            self.dat = fdef.dat
        else:
            fref = cube.readcube(fileref)
            # delta_data: V(def) - V(ref)
            # Ry --> Hartree if necessary (by a factor of 0.5)
            if is_charge == False:
                # precautious about the sign
                # eV -> V
                self.delta_data = -0.5*(fdef.dat - fref.dat) 
            else:
                self.delta_data = fdef.dat - fref.dat

        # rvec: real-space lattice vector
        # ngrid: the grid tuple
        rvec = fdef.rvec
        ngrid = fdef.ng[0]
        
        # -------------- init the g vectors from rvec
        # ngrid: [nx, ny, nz]
        # V: unit cell volume
        # dV: volume per FFT cube grid
        # b1,b2,b3: reciprocal lattice vector
        # pdef: a tuple of the defect xyz cartesian position (in bohr)
        V = np.linalg.det(rvec)
        dV = V/np.prod(ngrid)
        b1 = 2.*pi*np.cross(rvec[1],rvec[2])/V
        b2 = 2.*pi*np.cross(rvec[2],rvec[0])/V
        b3 = 2.*pi*np.cross(rvec[0],rvec[1])/V  
        gvec = np.array([b1,b2,b3]) 

        # setup |G|**2
        # gb: list of gvec boundaries 
        gb = np.zeros([3,2])
        for i in [0,1,2]:
            gb[i,:] = ceil(-ngrid[i]/2.), ceil(ngrid[i]/2.)
        # indices of the 3d mesh
        g1, g2, g3 = np.mgrid[gb[0][0]:gb[0][1]:1, \
                     gb[1][0]:gb[1][1]:1, gb[2][0]:gb[2][1]:1]
        g = [g1, g2, g3]
        # GG: |G|**2
        # GG/2 in Hartree
        GG = (g1*b1[0]+g2*b2[0]+g3*b3[0])**2 + \
             (g1*b1[1]+g2*b2[1]+g3*b3[1])**2 + \
             (g1*b1[2]+g2*b2[2]+g3*b3[2])**2 

        self.g = g 
        self.GG = GG
        self.dV = dV
        self.V = V        
        self.is_charge = is_charge
 
        # ----------------- init the position mesh grids  
        rmesh = np.mgrid[0:1:1./ngrid[0], 0:1:1./ngrid[1], 0:1:1./ngrid[2]] 
        
        # define the defect position in the direct lattice vector basis
        pdefd = np.dot(pdef, np.linalg.inv(rvec))
        
        # shift the mesh w.r.t the defect
        xx, yy, zz = rmesh[0]-pdefd[0], rmesh[1]-pdefd[1], rmesh[2]-pdefd[2]
        # center the defect 
        """xx[np.nonzero(xx >= 0.5)] -= 1.0
        yy[np.nonzero(yy >= 0.5)] -= 1.0
        zz[np.nonzero(zz >= 0.5)] -= 1.0"""
        # this turns out to be faster
        xx = np.mod(xx+0.5, 1.0) - 0.5
        yy = np.mod(yy+0.5, 1.0) - 0.5
        zz = np.mod(zz+0.5, 1.0) - 0.5
        
        # r: distance array (to the defect)
        X0 = xx*rvec[0,0] + yy*rvec[0,1] + zz*rvec[0,2]
        Y0 = yy*rvec[1,0] + yy*rvec[1,1] + zz*rvec[1,2]
        Z0 = xx*rvec[2,0] + yy*rvec[2,1] + zz*rvec[2,2]
        r = np.sqrt(X0**2 + Y0**2 + Z0**2)
        
        self.r = r
        self.rvec = rvec 
        self.X0 = X0
        self.Y0 = Y0
        self.x0 = X0[:,0,0]
        self.y0 = Y0[0,:,0]
        self.z0 = Z0[0,0,:]
        self.pdefd = pdefd
        self.q = q
        self.x = x
        self.gamma = gamma
        self.beta = beta
        self.ecut = ecut
        self.eps = eps        
        self.axis = axis
        self.plt = plt # level of plotting (0: no, 1: potential, 2: charge, 3: all) 
        self.out = out

    def qmodel(self, q=None, x=None, gamma=None, beta=None):
        # defect charge model (cf. PRL 102, 016402)
        # q: charge; x: normalization; gamma, beta: coefficient
        GG = self.GG
        g = self.g
        pdefd = self.pdefd
        dV = self.dV
        V = self.V
        r = self.r
        
        # are we supplying the model parameters other than the initial values?
        if None in (q, x, gamma, beta):
            q = self.q
            x = self.x
            gamma = self.gamma
            beta = self.beta 
        else:   
            self.q = q
            self.x = x
            self.gamma = gamma
            self.beta = beta 

        # normalizations
        self.N_gamma = 8*pi*(gamma**3)
        self.N_beta = pi**1.5*(beta**3) 
        
        # rho0r: r-space rho (aperiodic)
        self.rho0r = q*x*exp(-r/gamma)*(1/self.N_gamma) + \
                     q*(1-x)*exp(-r**2/beta**2)*(1/self.N_beta)

        # rhor: (periodic) charge density in real space from ifft(rhog)
        # rhog: charge density in k-space
        rhog = q*x/(1+gamma**2*GG)**2 + q*(1-x)*exp(-0.25*beta**2*GG)

        # FT shift w.r.t the defect position
        self.rhogs = rhog*exp(-2j*pi*(g[0]*pdefd[0] + g[1]*pdefd[1] + g[2]*pdefd[2])) 
        rhor = ifftn(ifftshift(self.rhogs))/dV

        # check the charge density localization
        print "Charge model: %0.2f exponential (gamma=%0.2f) +% 0.2f gaussian (beta=%0.2f)" \
               % (self.x, self.gamma, 1-self.x, self.beta)
        print "Localization of the model charge (0-1): %5.2f" \
               % (np.sum(self.rho0r) / np.sum(rhor.real))

        # plane-averaged charge density
        self.rhor_avg = self.planeave(rhor, self.axis)
        self.rho0r_avg = self.planeave(self.rho0r, self.axis)
        self.rhor = rhor
        self.rhog = rhog
        
        # we add the homogeneous background to rhor
        # n0: uniform background charge density
        """self.n0 = -q/V
        # rho1: summed charge density
        rho1r = self.rho0r + self.n0
        rho1g = fftn(rho1r)

        self.rho1r_avg = self.planeave(rho1r, self.axis)
        self.rho1r = rho1r
        self.rho1g = rho1g"""

    def planeave(self, array, dir):
        # perform a plane-average operation of 'array' along the 'dir' axis
        # dir: 0 -> x, 1 -> y, 2 -> z
        assert dir in [0,1,2], \
          "Invalid direction"
        dim = array.shape[dir]
        avg = np.empty(dim)
        if dir == 0:
            for i in xrange(dim):
                avg[i] = np.mean(array[i,:,:]).real
        elif dir == 1:
            for i in xrange(dim):
                avg[i] = np.mean(array[:,i,:]).real
        else:
            for i in xrange(dim):
                avg[i] = np.mean(array[:,:,i]).real
        return avg

    def energy(self):
        # calculate the correction energy
        r = self.r
        GG = self.GG
        dV = self.dV
        rhog = self.rhog
        ecut = self.ecut
        q = self.q
        x = self.x
        gamma = self.gamma
        beta = self.beta
        pdefd = self.pdefd
        g = self.g
        
        # indices for G vectors within the cutoff sphere 
        # note that ecut is in Ry
        self.gcut = np.nonzero(self.GG < self.ecut)
        print "G-vectors: %i" % self.gcut[0].size        

        # Vlr: long-range potential for the periodic array
        # Vlr(G!=0) = 4*pi*q(G)/(eps*GG)  
        # Vlr(G=0) := 2*pi*d2q/dG2/eps
        # Vlrr: in real space
        # Vlrg: in k-space
        np.seterr(all='ignore')
        self.Vlrg = self.poisson(self.rhogs)
        self.ig0 = np.nonzero(GG==0)
        self.VG0 = self.d2qdg2()
        self.Vlrg[self.ig0] = 2*pi*self.VG0/self.eps
        self.VG0 = self.Vlrg[self.ig0].real/np.sqrt(self.V)

        Ha = physical_constants['Hartree energy in eV'][0]
        self.Ha = Ha
        print "unit cell volume: %8.5f bohr^3" % (self.V) 
        print "(unscreened) V(G=0) := %8.5f" % (self.VG0*self.eps)
        print "(unscreened) q*V(G=0)/V = %8.5f" % (self.VG0*self.q/np.sqrt(self.V)*self.eps)
        print "(screened) q*V(G=0)/V = %8.5f eV" % (self.VG0*self.q/np.sqrt(self.V)*Ha)

        Vlrr = ifftn(ifftshift(self.Vlrg))/dV
        self.Vlrr = Vlrr.real        
 
        # plane-averaged component
        # Vlrr:
        Vlrr_avg = self.planeave(self.Vlrr, self.axis)
        # delta_data:
        delta_data_avg = self.planeave(self.delta_data, self.axis)
        # Vsrr: delta_data_avg - Vlrr_avg
        Vsrr_avg = delta_data_avg - Vlrr_avg

        # ---------------------------
        """Vlr1g = self.poisson(rho1g)
        Vlr1g[self.ig0] = 0
        Vlr1r = ifftn(ifftshift(Vlr1g))/dV
        self.Vlr1r = Vlr1r.real
        self.Vlr1r_avg = self.planeave(self.Vlr1r, self.axis)"""
        #----------------------------
        
        # Vlr0: aperiodic Vlr from real-space q model
        """self.Vlr0r = q*(x*(1/r - exp(-r/gamma)*(1/r+0.5/gamma)) \
                + (1-x)/r*erf(r/beta))
        ir0 = np.nonzero(r < 0.0001)
        self.Vlr0r[ir0] = q*(x*(0.5*gamma-r[ir0]**2/(12*gamma**3)) + \
                          (1-x)*(2/np.sqrt(pi)/beta - 2/3/np.sqrt(pi)*r[ir0]**2/beta**3))
        self.Vlr0r /= self.eps
        self.Vlr0r_avg = self.planeave(self.Vlr0r, self.axis)

        # periodicity induced Vlr
        self.dVlrr_avg = self.Vlrr_avg - self.Vlr0r_avg"""

        self.Eisol = self.isolenergy()
        self.Elatt = self.lattenergy()
        self.Ecorr0 = self.Elatt - self.Eisol
        
        # determine the Vsr
        a = (self.x0, self.y0, self.z0)[self.axis]
        # shift the axis so that the defect is now at the left end
        j = np.nonzero(a < a[0])
        a = self.reorder(a, j)

        delta_data_avg = self.reorder(delta_data_avg, j)
        Vlrr_avg = self.reorder(Vlrr_avg, j)
        Vsrr_avg = self.reorder(Vsrr_avg, j)

        self.Esrc  = Vsrr_avg[0]
        self.Ecorr = self.Ecorr0 + self.q*self.Esrc

        self.a = a
        self.delta_data_avg = delta_data_avg
        self.Vlrr_avg = Vlrr_avg
        self.Vsrr_avg = Vsrr_avg

        # output
        align_der = (x*8*gamma**2 + (1-x)*beta**2)*q*pi/(self.V*self.eps)
        print "(unscreened) Eisol: %9.5f Ha; Elatt: %9.5f Ha" \
                % (self.Eisol*self.eps, self.Elatt*self.eps)
        print "(unscreened) Ecorr: %9.5f Ha" % (self.Ecorr0*self.eps)        
        print "(screened) Ecorr: %9.5f eV" % (self.Ecorr0*Ha)
        print "SR alignment corr: %9.5f eV" % (self.Esrc*Ha)
        print "! (screened and aligned) Ecorr: %9.5f eV" %(self.Ecorr*Ha)
        print "! (screened) alignment derivative: %9.5f eV" % (align_der*Ha)

        # plotting
        if self.plt > 0 and self.is_charge == False:
            if self.plt == 1:
                font = {'fontname':'sans-serif'}
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(a, delta_data_avg*Ha, 'ko-', \
                        a, Vlrr_avg*Ha, 'b-', \
                        a, Vsrr_avg*Ha, 'r-')
                ax.set_ylabel('Electrostatic potential (eV)', **font)
                leg = ax.legend(('$\Delta V_{el}$', '$V_{lr}$', '$V_{sr}$'), 'upper right')
                ax.set_xlabel (r'$z$ (bohr)', **font)
                ax.autoscale(axis='x', tight=True)
                np.savetxt(self.out, zip(a, delta_data_avg*Ha, Vlrr_avg*Ha), fmt='%10.4f %15.6f %15.6f')
                plt.show()

            if self.plt == 9:
                rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
                fig_width = 2
                fig_height = fig_width
                fig_size = [fig_width, fig_height]
                params = {'backend': 'ps',
                          'axes.labelsize': 8,
                          'text.fontsize': 8,
                          'legend.fontsize': 8,
                          'xtick.labelsize': 6,
                          'ytick.labelsize': 6,
                          'text.usetex': True,
                          'figure.figsize': fig_size,
                          'lines.markersize': 2}

                plt.rcParams.update(params)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(a, delta_data_avg*Ha, 'ko-',  
                        a, Vlrr_avg*Ha, 'b-',  
                        a, Vsrr_avg*Ha, 'r-')
                ax.set_ylabel('Electrostatic potential (eV)')
                ax.set_xlabel (r'$z$ (bohr)')
                ax.autoscale(axis='x', tight=True)
                plt.savefig('V.eps')    

            elif self.plt == 2:
                font = {'fontname':'sans-serif'}
                fig = plt.figure()
                ax = fig.add_subplot(111)
                rhor_avg = self.reorder(self.rhor_avg, j)
                rho0r_avg = self.reorder(self.rho0r_avg, j)
                ax.plot(a, rhor_avg, 'r-', a, rho0r_avg, 'bo')
                ax.set_ylabel('Model charge density', **font)
                leg = ax.legend(('periodic', 'aperiodic'), 'upper right')
                ax.set_xlabel (r'$z$ (bohr)', **font)
                ax.autoscale(axis='x', tight=True)
                plt.show()

    def reorder(self, array, j):
        # make the elements correspond to the axis in an increasing order
        return np.concatenate((array[j], np.delete(array,j)))

    def d2qdg2(self):
        # d2rho/dg2 for g=0
        return -(4*self.x*self.gamma**2+0.5*(1-self.x)*self.beta**2)*self.q

    def poisson(self, rhog):
        # Poisson solver in the k-space
        return 4.0*pi*np.divide(rhog, self.GG)/self.eps

    def isolenergy(self):
        # Calculate the isolated self-energy due to the model charge
        # 1. k-space solution
        # ecut: cut-off energy in a.u. (Ry)
        # include the g-vectors within the ecut
        x = self.x
        gamma = self.gamma
        beta = self.beta
        q = self.q
        qg = lambda g: (q*x/(1+gamma**2*g**2)**2 + q*(1-x)*exp(-0.25*beta**2*g**2))**2
        Eisol, err = quad(qg, 0., np.sqrt(self.ecut))
        Eisol /= pi*self.eps
        return Eisol

    def lattenergy(self):
        # screened lattice energy (Madelung energy) of rhog
        # remove the G=0 component from gcut
        # there must be a better solution other than this
        gcut = [0,0,0]
        for i in xrange(3):
            gcut[i] = np.delete(self.gcut[i], self.gcut[i].size/2)     
        gcut = tuple(gcut)  
        Elatt = 2*pi*sum(self.rhog[gcut]**2 / self.GG[gcut]) / (self.V*self.eps)
        # add qV(G=0) for consistent alignment
        Elatt = Elatt.real + self.q*self.VG0/np.sqrt(self.V)
        return Elatt.real

    def convolve(self, sigma, is_charge=False):
        """convolve the 0/bulk potential difference using a Gaussian kernel"""
        f = interp1d(self.a, self.delta_data_avg, "cubic")
        n = self.a.size
        x = np.linspace(self.a.min(), self.a.max(), n*2)
        y = f(x)

        g = lambda p, s: 1.0/np.sqrt(2*np.pi)/s*np.exp(-0.5*p**2/s**2)
        p = np.linspace(-1,1,n*2)
        gk = fft(fftshift(g(p, sigma)))
        yk = fft(y)
        conv = ifft(yk*gk)/n


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel (r'$z$ (bohr)')

        if is_charge==True:
            # the input cube files are charge densities
            y *= self.V
            conv *= self.V
        else:   
            # the inputs are potentials
            y *= self.Ha
            conv *= self.Ha
            ax.set_ylabel('Electrostatic potential (eV)')
            
        ax.plot(x, y, 'o-', x, conv, '-', x, self.q*np.ones(n*2)*(1-1./self.eps), '--')
        ax.autoscale(axis='x', tight=True)
        plt.show()

class Qfit(correction):
    """fit the model charge to the defect wavefunction"""
    def __init__(self, wfcube, pdef, axis):
        self.axis = axis
        super(qfit, self).__init__(wfcube, 'None', pdef, axis)
        self.psi_avg = self.planeave(self.dat, self.axis)        
        # the averaging direction
        a = (self.x0, self.y0, self.z0)[self.axis]
        # shift the axis so that the defect is now at the left end
        self.j = np.nonzero(a < a[0])
        self.a = self.reorder(a, self.j)
        self.psi_avg = self.reorder(self.psi_avg, self.j)

    def fitting(self, q, x, gamma, beta):
        self.q = q
        super(qfit, self).qmodel(q,x,gamma,beta)        
        self.rhor_avg = self.planeave(self.rhor, self.axis)        
        self.rhor_avg = self.reorder(self.rhor_avg, self.j) / self.q

        self.plotwf()

    def plotwf(self):
        font = {'fontname':'sans-serif'}
        fig = plt.figure()
        ax = fig.add_subplot(111)
            
        ax.plot(self.a, self.psi_avg, 'r.-', self.a, self.rhor_avg, 'g--') 
        ax.set_yscale('log')
        ax.set_xlabel(r'$z$ (bohr)', **font)
        ax.set_ylabel(r'$|\psi|^2$', **font)
        ax.autoscale(axis='x', tight=True)
        plt.show()
