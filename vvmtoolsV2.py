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
        self.MEANAXIS  = {'x':-1, 'y':-2, 'xy':(-1, -2)}    # compute_mean_axis options

    def _Range_check_agrid(self, domain_range, conv_agrid:bool=False):
        """
        Default: conv_agrid set to FALSE because this function should be activated only when the conversion to a-grid is performed.
        """
        k1, k2, j1, j2, i1, i2 = domain_range
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
        return domain_range
    
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
        if conv_agrid:
            # Check domain_range on the x-direction
            domain_range           = self._Range_check_agrid(domain_range, conv_agrid)
            k1, k2, j1, j2, i1, i2 = domain_range
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
    
    def cal_enstrophy(self, time, domain_range, 
                      conv_agrid:bool=True, 
                      print_shape:bool=False):
        """
        Calculate enstrophy over the designated domain range and on the specified time step.
        Default:  eta, xi, zeta will be converted to a-grid before calculation.
        Default:  Calculate domain-average. Enstrophy is only representative of turbulent motions when considering an area.
        Optional: Set print_shape=True to get the shape info of the retrieved eta, xi, zeta.
        """
        if conv_agrid:
            # Check domain_range on the x-direction
            domain_range           = self._Range_check_agrid(domain_range, conv_agrid)
            k1, k2, j1, j2, i1, i2 = domain_range
            # Get converted eta, xi, zeta
            eta  = self.convert_to_agrid('eta', time)[k1:k2, j1:j2, i1:i2].copy()
            xi   = self.convert_to_agrid('xi', time)[k1:k2, j1:j2, i1:i2].copy()
            zeta = self.convert_to_agrid('zeta', time)[k1:k2, j1:j2, i1:i2].copy()
        else: 
            ## Check dimension for the dynamic eta
            eta_temp = self.get_var('eta', 0, (1, 1, 1, 1, 1, 1), numpy=True)
            if len(eta_temp.shape) < 3:
                eta = np.squeeze(self.get_var("eta_2", time, domain_range, numpy=True))
            else:
                eta = np.squeeze(self.get_var("eta", time, domain_range, numpy=True))
            ## Other components of vorticity
            xi   = np.squeeze(self.get_var("xi", time, domain_range, numpy=True))
            zeta = np.squeeze(self.get_var("zeta", time, domain_range, numpy=True))
        # Provide shape info
        if print_shape: print("Shape of eta, xi, zeta:", eta.shape, xi.shape, zeta.shape)

        # POSSIBLE TODO: argument xarray:bool 
        # -> might be more convenient to entail variable info
        
        return np.nanmean((eta**2+xi**2+zeta**2), axis=(1, 2))
    
    def cal_turb_flux(self, time, domain_range, 
                      wind_var, prop_var,
                      conv_agrid:bool=True):
        """
        Params:
        wind_var is the medium [u/v/w].
        prop_var is the property transported by the medium.
        """
        if conv_agrid:
            # Check domain_range on the x-direction
            domain_range           = self._Range_check_agrid(domain_range, conv_agrid)
            windvar  = self.convert_to_agrid(wind_var, time)
            propvar  = self.convert_to_agrid(prop_var, time)
        else:
            windvar  = np.squeeze(self.get_var(wind_var, time, numpy=True))
            propvar  = np.squeeze(self.get_var(prop_var, time, numpy=True))
        # Calculate entire-domain mean
        wind_bar = np.nanmean(windvar, axis=(-2, -1))
        prop_bar = np.nanmean(propvar, axis=(-2, -1))
        # Calculate flux
        k1, k2, j1, j2, i1, i2 = domain_range
        wind_reg = windvar[k1:k2, j1:j2, i1:i2].copy()
        wind_bar = np.repeat(wind_bar[..., np.newaxis], axis=1, repeats=wind_reg.shape[-2])
        wind_bar = np.repeat(wind_bar[..., np.newaxis], axis=2, repeats=wind_reg.shape[-1])
        prop_reg = propvar[k1:k2, j1:j2, i1:i2].copy()
        prop_bar = np.repeat(prop_bar[..., np.newaxis], axis=1, repeats=prop_reg.shape[-2])
        prop_bar = np.repeat(prop_bar[..., np.newaxis], axis=2, repeats=prop_reg.shape[-1])
        return (wind_reg-wind_bar)*(prop_reg-prop_bar)
    
    def get_pbl_height(self, time, domain_range, 
                       method:str, threshold:float=0., compute_mean_axis=None, 
                       conv_agrid:bool=True):
        """
        Available method now includes: 'wth', 'th05k', 'dthdz'
        """
        if method == 'wth':
            heights = self._pbl_height_wth(time, domain_range, conv_agrid)
        elif method == 'th05k':
            heights = self._pbl_height_th05k(time, domain_range, compute_mean_axis, conv_agrid)
        elif method == 'dthdz':
            heights = self._pbl_height_dthtz(time, domain_range, compute_mean_axis, conv_agrid)
        elif method in ('tke', 'TKE'):
            tke     = self.cal_TKE(time, domain_range, conv_agrid)
            hcidx   = np.nanargmin(abs(tke-threshold))
            heights = self.DIM['zc'][1:][hcidx] if conv_agrid is True else self.DIM['zc'][hcidx]
        elif method == 'enstrophy':
            enstrophy= self.cal_enstrophy(time, domain_range, conv_agrid)
            hcidx   = np.nanargmin(abs(enstrophy-threshold))
            heights = self.DIM['zc'][1:][hcidx] if conv_agrid is True else self.DIM['zc'][hcidx]
        else:
            raise ValueError("Unrecognized method for defining PBL height. Please choose from ['wth', 'th05k', 'dthdz', 'tke', 'enstrophy'].")
        return heights
        
    def _pbl_height_wth(self, time, domain_range, 
                        threshold:float, 
                        conv_agrid:bool=True):
        """
        Default: domain-average (from the calculation of flux)
        """
        # Check domain_range on the x-direction
        if conv_agrid:
            domain_range = self._Range_check_agrid(domain_range, conv_agrid)
            z            = self.DIM['zc'][1:]
        else:
            z            = self.DIM['zc']
        # Method: wth
        pass
    
    def _pbl_height_th05k(self, time, domain_range,
                          compute_mean_axis=None,
                          conv_agrid:bool=True):
        """
        Default: No average. If compute_mean_axis is activated, theta mean is calculated before rendering PBL height.
        params:
        compute_mean_axis: can be None (default), 'x', 'y', 'xy'
        """
        # Check domain_range on the x-direction
        if conv_agrid:
            domain_range = self._Range_check_agrid(domain_range, conv_agrid)
            k1, k2, j1, j2, i1, i2 = domain_range
            z            = self.DIM['zc'][1:]
            th           = self.convert_to_agrid('th', time)[k1:k2, j1:j2, i1:i2].copy()
        else:
            z            = self.DIM['zc']
            th           = self.get_var('th', time, domain_range, numpy=True)
        # Method: th05k
        if compute_mean_axis is not None:
            axis     = self.MEANAXIS[compute_mean_axis]
            th_mean  = np.nanmean(th, axis=axis).copy()
            if isinstance(axis, tuple):                           # compute_mean_axis=(1, 2)
                result = self._find_levels_1d(th_mean, conv_agrid)
            else:                                                 # compute_mean_axis=1 or 2
                result = np.zeros(th_mean.shape[-1])
                for ii in range(th_mean.shape[-1]):
                    result[ii] = self._find_levels_1d(th_mean[:, ii], conv_agrid)
        else:
            result = np.zeros(th[0, ...].shape)
            for j in range(th.shape[-2]):
                for i in range(th.shape[-1]):
                    result[j, i] = self._find_levels_1d(th[:, j, i], conv_agrid)
        return result
        
    def _find_levels_1d(self, var, conv_agrid:bool):
        target_var = var[0] + 0.5
        zidx       = np.argmax(var >= target_var)
        zidx       = np.where(np.any(var >= target_var), zidx, 0)   # set to 0 if no level satifies the condition
        if conv_agrid:
            return self.DIM['zc'][1:][zidx]
        else:
            return self.DIM['zc'][zidx]
        
    def _pbl_height_dthtz(self, time, domain_range,
                          compute_mean_axis=None,
                          conv_agrid:bool=True):
        # Check domain_range on the x-direction
        if conv_agrid:
            domain_range = self._Range_check_agrid(domain_range, conv_agrid)
            k1, k2, j1, j2, i1, i2 = domain_range
            z            = self.DIM['zc'][1:]
            th           = self.convert_to_agrid('th', time)[k1:k2, j1:j2, i1:i2].copy()
        else:
            z            = self.DIM['zc']
            th           = self.get_var('th', time, domain_range, numpy=True)
        # Method: dthdz
        if compute_mean_axis is not None:
            axis     = self.MEANAXIS[compute_mean_axis]
            th_mean  = np.nanmean(th, axis=axis).copy()
            slope    = (th_mean[1:, ...]-th_mean[:-1, ...])/(z[2]-z[1])
            slope_max_idx = np.argmax(slope, axis=0)
            if isinstance(axis, tuple):                           # compute_mean_axis=(1, 2)
                result = z[slope_max_idx]
            else:                                                 # compute_mean_axis=1 or 2
                result = np.zeros(th_mean.shape[-1])
                for ii in range(th_mean.shape[-1]):
                    result[ii] = z[slope_max_idx[ii]]
        else:
            slope         = (th[1:, ...]-th[:-1, ...])/(z[2]-z[1])
            slope_max_idx = np.argmax(slope, axis=0)
            result = np.zeros(th[0, ...].shape)
            for j in range(th.shape[-2]):
                for i in range(th.shape[-1]):
                    result[j, i] = z[slope_max_idx[j, i]]
        return result
        
# --- Test --- #
if __name__ == "__main__":
    test_case     = '/data/mlcloud/ch995334/VVM/DATA/pbl_mod_wfire_coastal_s1/'
    test_instance = VVMtools(test_case)
    # Annoucing function testing
    print("Function testing: get_pbl_height")
    # Necessary variables and get result
    test_var1   = 'w'
    test_var2   = 'th'
    time_step   = 180
    test_range  = (None, None, None, None, 64, None)
    test_result = test_instance.get_pbl_height(time=time_step, domain_range=test_range, method='tke', threshold=1e-1)
    # Testing result
    print("time_step:", time_step, "domain_range:", test_range)
    print(test_result)
    print(test_result.shape)