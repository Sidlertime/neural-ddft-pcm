from torch import nn
import torch
from .dft1d import dft1d
import numpy as np
import os
import sys
# Resolve reference dataset relative to this file for robust execution
REFERENCE_DATASET = os.path.join(os.path.dirname(__file__), "lj-bulk")

def get_bulk_densities():

    file_nrs = np.array([int(f[4:-4]) for f in os.listdir(REFERENCE_DATASET) if f.startswith("rho_")])
    rho_bulk_dict = {}
    
    for file_nr in file_nrs:
        rho_b = np.mean(np.load(f"{REFERENCE_DATASET}/rho_{file_nr}.npy"))
        mu = float(np.load(f"{REFERENCE_DATASET}/mu_{file_nr}.npy"))
        rho_bulk_dict[mu] = rho_b

    return rho_bulk_dict

def get_closest_key(dictionary, mu):
    return min(dictionary.keys(), key=lambda k: abs(k - mu))

class FMT_Functional(nn.Module):

    def __init__(self,
                 sigma = 1,
                 kT = 2, # CHANGED
                 epsilon = 1, # CHANGED
                 fmtmethod = 'WBII',
                 ljmethod = 'WDA',
                 dz = float,
                 Lz = int):
        """
        Inputs:
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.sigma = sigma
        self.kT = kT
        self.epsilon = epsilon
        self.Lz = Lz
        self.A = Lz**2
        self.dz = dz
        self.FMT1D = dft1d(fmtmethod=fmtmethod,ljmethod=ljmethod)
        self.FMT1D.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
        self.FMT1D.Set_Geometry(Lz)  
        self.FMT1D.Set_Temperature(self.kT, dz)
        self.bulk_density_dict = get_bulk_densities()

    def cal_dF_drho_FMT(self, rho, mu):

        closest_mu = get_closest_key(self.bulk_density_dict, float(mu))
        rho_b = self.bulk_density_dict[closest_mu]
        self.FMT1D.rho = rho[0,0,:]
        self.FMT1D.Set_BulkDensity(rho_b)
        self.FMT1D.Calculate_weighted_densities()
        self.FMT1D.Calculate_c1()
        dF_drho_FMT = -self.FMT1D.c1

        return dF_drho_FMT[None,None,:] # !!! I think that the inverse of c1 is calculated in this function
    
    def cal_packing_frac(self, rhob):

        eta = np.pi * rhob * self.sigma**3 / 6

        return eta
    
    def c2_hs(self, r, eta):

        c2_hs = (- (1 + eta*(4 + eta*(3-2*eta)))/(1-eta)**4)*np.ones_like(r)
        c2_hs += (((2-eta + 14*eta**2 - 6*eta**3)/(1-eta)**4) + (2*np.log(1-eta)/eta))*(r/self.sigma)
        c2_hs -= ((3 + 5*eta*(eta - 2)*(1-eta))/(1-eta)**4 + (3*np.log(1-eta)/eta))*(r/self.sigma)**3

        return c2_hs
    
    def radial2planar(self, c_r, r):

        c_x = np.zeros_like(c_r)
        for i, x in enumerate(r):
            integrand = r[i:] * c_r[i:] 
            c_x[i] = 2 * np.pi * np.trapz(integrand, r[i:]) 

        return c_x, r
    
    def cal_c2(self, rho, dz=1/32):

        """

            This c2 calculation now includes the radial to planar transformation.

        """

        R = 1
        rho_b = np.mean(rho)
        z = np.arange(0, R, dz)
        z_axis = np.arange(0, int(self.Lz/2), self.dz)
        eta = self.cal_packing_frac(rho_b)
        self.FMT1D.Set_BulkDensity(rho_b)
        c2 = np.zeros_like(z_axis)
        c2_hs = self.c2_hs(z, eta)
        hs_axis = np.arange(0, 1, dz)
        c2_hs_planar = self.radial2planar(c2_hs, hs_axis)[0]
        c2_mf = self.FMT1D.ulj[len(self.FMT1D.ulj)//2:]
        c2[:len(c2_mf)] = - c2_mf 
        c2[:len(c2_hs)] += c2_hs_planar

        return c2
    
    def cal_F(self, rho):

        self.FMT1D.rho = rho
        self.FMT1D.Calculate_weighted_densities()
        self.FMT1D.Calculate_Free_energy()

        F_exc = self.A * self.FMT1D.Fexc

        return F_exc
    
    def forward(self, rho, mu):

        dF_drho_FMT = self.cal_dF_drho_FMT(rho, mu)

        return dF_drho_FMT 
