# --- Import --- #
import numpy as np
import pandas as pd
import xarray as xr
from datetime import *
import glob
import logging
import vvmtools as vvmtools_aaron  # vvmtools V1

# --- Build Class --- #
class VVMtools(vvmtools_aaron.VVMTools):
    def __init__(self, casepath):
        super().__init__(casepath)
        
        self.TIMESTEPS = len(glob.glob(f"{casepath}/archive/*Dynamic*.nc"))

    
    def convert_to_agrid(self, var, time):
        """
        Default: Interpolate the entire domain on the designated time step.
        """
        # Wind field (u, v, w)
        if var == 'u':
            u_org   = self.get_var('u', time, numpy=True)
            u_agrid = (u_org[..., 1:]+u_org[..., :-1])/2   # itp_dim: (z, y, x) -> (x)
            return u_agrid[1:, :-1, :]                     # discard the last y and first z
        elif var == 'v':
            v_org   = self.get_var('v', time, numpy=True)
            v_agrid = (v_org[:, 1:, :]+v_org[:, :-1, :])/2 # itp_dim: (z, y, x) -> (y)
            return v_agrid[1:, :, :-1]                     # discard the last x and first z
        elif var == 'w':
            w_org   = self.get_var('w', time, numpy=True)
            w_agrid = (w_org[1:, ...]+w_org[:-1, ...])/2   # itp_dim: (z, y, x) -> (x)
            return w_agrid[:, :-1, :-1]                    # discard the last x and y
        
        # Vorticity field(eta, xi, zeta)
        elif var == 'eta':
            # Check dimension for the dynamic eta
            eta_temp = self.get_var('eta', 0, (1, 1, 1, 1, 1, 1), numpy=True)
            if len(eta_temp.shape) < 3:
                eta_org = self.get_var('eta_2', time, numpy=True)
            else:
                eta_org = self.get_var('eta', time, numpy=True)
            eta_agrid   = (eta_org[1:, :, 1:] +eta_org[1:, :, :-1]+      # itp_dim: (z, y, x)
                           eta_org[:-1, :, 1:]+eta_org[:-1, :, :-1])/4.  # -> (z, x)
            return eta_agrid[:, :-1, :]                                  # discard the last y
        elif var == 'xi':
            xi_org      = self.get_var('xi', time, numpy=True)  
            xi_agrid    = (xi_org[1:, 1:, :] +xi_org[1:, :-1, :]+        # itp_dim: (z, y, x)
                           xi_org[:-1, 1:, :]+xi_org[:-1, :-1, :])/4.    # -> (z, y)
            return xi_agrid[:, :, :-1]                                   # discard the last x
        elif var == 'zeta':
            zeta_org    = self.get_var('zeta', time, numpy=True)
            zeta_agrid  = (zeta_org[:, 1:, 1:] +zeta_org[:, 1:, :-1]+    # itp_dim: (z, y, x)
                           zeta_org[:, :-1, 1:]+zeta_org[:, :-1, :-1])/4.# -> (y, x)
            return zeta_agrid[1:, :, :]                                  # discard the first z
        
        # Theta (standard)
        elif var == 'th':
            th_org      = self.get_var('th', time, numpy=True)
            return th_org[1:, :-1, :-1]       # discard the first z, last y, last x
        
# --- Test --- #
if __name__ == "__main__":
    test_case     = '/data/mlcloud/ch995334/VVM/DATA/pbl_mod_wfire_coastal_s1/'
    test_instance = VVMtools(test_case)
    # Annoucing function testing
    print("Function testing: convert_to_agrid")
    # Necessary variables and get result
    test_var    = 'eta'
    time_step   = 100
    test_result = test_instance.convert_to_agrid(var=test_var, time=time_step)
    # Testing result
    print("Var:", test_var, "time_step:", time_step)
    print(test_result)
    print(test_result.shape)