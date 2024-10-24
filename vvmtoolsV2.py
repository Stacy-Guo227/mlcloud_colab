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
        
    def cal_TKE(self, time, domain_range, 
                conv_agrid:bool=True, 
                print_shape:bool=False):
        """
        Calculate TKE over the designated domain range and on the specified time step.
        Default:  u, v, w will be converted to a-grid before calculation.
        Default:  Calculate domain-average. TKE is only representative of turbulent motions when considering an area.
        Optional: Set print_shape=True to get the shape info of the retrieved u, v, w.
        """
        k1, k2, j1, j2, i1, i2 = domain_range
        if conv_agrid:
            # Check domain_range on the x-direction
            if i2 is not None and (i2 > 63):
                logging.warning(f"Grid {i2} on the x-axis is on/over the borderline "+ 
                                f"between two landtypes after interpolation (a-grid), "+ 
                                f"automatically set to 63.")
                i2           = 63
                domain_range = (k1, k2, j1, j2, i1, i2)
            if i1 is not None and (i1 < 64):
                logging.warning(f"Grid {i1} on the x-axis is on/over the borderline "+ 
                                f"between two landtypes after interpolation (a-grid), "+ 
                                f"automatically set to 64.")
                i1           = 64
                domain_range = (k1, k2, j1, j2, i1, i2)
            # Get converted u, v, w
            u = self.convert_to_agrid('u', time)[k1:k2, j1:j2, i1:i2].copy()
            v = self.convert_to_agrid('v', time)[k1:k2, j1:j2, i1:i2].copy()
            w = self.convert_to_agrid('w', time)[k1:k2, j1:j2, i1:i2].copy()
        else:      
            u = np.squeeze(self.get_var("u", time, domain_range, numpy=True))
            v = np.squeeze(self.get_var("v", time, domain_range, numpy=True))
            w = np.squeeze(self.get_var("w", time, domain_range, numpy=True))
        # Provide shape info
        if print_shape: print("Shape of u, v, w:", u.shape, v.shape, w.shape)

        # POSSIBLE TODO: argument xarray:bool 
        # -> might be more convenient to entail variable info
        
        return np.nanmean((u**2+v**2+w**2)/2, axis=(1, 2))
        
# --- Test --- #
if __name__ == "__main__":
    test_case     = '/data/mlcloud/ch995334/VVM/DATA/pbl_mod_wfire_coastal_s1/'
    test_instance = VVMtools(test_case)
    # Annoucing function testing
    print("Function testing: cal_TKE")
    # Necessary variables and get result
    # test_var    = 'eta'
    time_step   = 100
    test_range  = (None, None, None, None, 64, None)
    test_result = test_instance.cal_TKE(time=time_step, domain_range=test_range, print_shape=True, conv_agrid=False)
    # Testing result
    print("time_step:", time_step, "domain_range:", test_range)
    print(test_result)
    print(test_result.shape)