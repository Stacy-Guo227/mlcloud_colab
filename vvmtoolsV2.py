# --- Import --- #
import numpy as np
import xarray as xr
from datetime import *
import glob
import logging
from functools import partial
import multiprocessing
import vvmtools as vvmtools_aaron  # vvmtools V1

# --- Build Class --- #
class VVMtools(vvmtools_aaron.VVMTools):
    def __init__(self, casepath):
        super().__init__(casepath)
        
        self.TIMESTEPS = len(glob.glob(f"{casepath}/archive/*Dynamic*.nc"))
        self.MEANAXIS  = {'x':-1, 'y':-2, 'xy':(-1, -2)}    # compute_mean_axis options
    
    def convert_to_agrid(self, var, time, domain_range=(None, None, None, None, None, None)):
        """
        Convert the six dynamic variables (u, v, w, eta, xi, zeta) to a-grid. 
        After interpolation, the corresponding dimension would reduce by 1 grid at a time, the first grid on the uninterpolated dimensions would be omitted to acquire equal dimension outputs.  
        Theta is an acceptable input for dimension adjustment.
        
        :param var: Variable name
        :type  var: str
        :param time: Designated time step
        :type  time: int
        :param domain_range: Designated spatial range, dimension follows (z, y, x)
        :type  domain_range: tuple, optional, default:(None, None, None, None, None, None)
        
        :return: The interpolated data.
        :rtype : np.ndarray
        """
        # Wind field (u, v, w)
        if var == 'u':
            u_org   = self.get_var('u', time, domain_range, numpy=True)
            u_agrid = (u_org[..., 1:]+u_org[..., :-1])/2   # itp_dim: (z, y, x) -> (x)
            return u_agrid[1:, 1:, :]                      # discard the first y and z
        elif var == 'v':
            v_org   = self.get_var('v', time, domain_range, numpy=True)
            v_agrid = (v_org[:, 1:, :]+v_org[:, :-1, :])/2 # itp_dim: (z, y, x) -> (y)
            return v_agrid[1:, :, 1:]                      # discard the first x and z
        elif var == 'w':
            w_org   = self.get_var('w', time, domain_range, numpy=True)
            w_agrid = (w_org[1:, ...]+w_org[:-1, ...])/2   # itp_dim: (z, y, x) -> (x)
            return w_agrid[:, 1:, 1:]                      # discard the first x and y
        
        # Vorticity field(eta, xi, zeta)
        elif var == 'eta':
            # Check dimension for the dynamic eta
            eta_temp = self.get_var('eta', 0, (1, 1, 1, 1, 1, 1), numpy=True)
            if len(eta_temp.shape) < 3:
                eta_org = self.get_var('eta_2', time, domain_range, numpy=True)
            else:
                eta_org = self.get_var('eta', time, domain_range, numpy=True)
            eta_agrid   = (eta_org[1:, :, 1:] +eta_org[1:, :, :-1]+      # itp_dim: (z, y, x)
                           eta_org[:-1, :, 1:]+eta_org[:-1, :, :-1])/4.  # -> (z, x)
            return eta_agrid[:, 1:, :]                                   # discard the first y
        elif var == 'xi':
            xi_org      = self.get_var('xi', time, domain_range, numpy=True)  
            xi_agrid    = (xi_org[1:, 1:, :] +xi_org[1:, :-1, :]+        # itp_dim: (z, y, x)
                           xi_org[:-1, 1:, :]+xi_org[:-1, :-1, :])/4.    # -> (z, y)
            return xi_agrid[:, :, 1:]                                    # discard the first x
        elif var == 'zeta':
            zeta_org    = self.get_var('zeta', time, domain_range, numpy=True)
            zeta_agrid  = (zeta_org[:, 1:, 1:] +zeta_org[:, 1:, :-1]+    # itp_dim: (z, y, x)
                           zeta_org[:, :-1, 1:]+zeta_org[:, :-1, :-1])/4.# -> (y, x)
            return zeta_agrid[1:, :, :]                                  # discard the first z
        
        # Theta (standard)
        elif var == 'th':
            th_org      = self.get_var('th', time, domain_range, numpy=True)
            return th_org[1:, 1:, 1:]         # discard the first z, y, x
        
    def cal_TKE(self, time, domain_range, 
                conv_agrid:bool=True):
        """
        Calculate area-averaged Turbulent Kinetic Energy for designated domain range and time step.
        (TKE is only representative of turbulent motions when considering an area.)
        
        :param time: Designated time step
        :type  time: int
        :param domain_range: Designated spatial range, dimension follows (z, y, x)
        :type  domain_range: tuple, optional, default:(None, None, None, None, None, None)
        :param conv_agrid: Perform this calculation on a-grid.
        :type  conv_agrid: bool, optional, default=True
        
        :return: TKE
        :rtype : np.ndarray
        """
        if conv_agrid:
            u = self.convert_to_agrid('u', time, domain_range)
            v = self.convert_to_agrid('v', time, domain_range)
            w = self.convert_to_agrid('w', time, domain_range)
        else:      
            u = np.squeeze(self.get_var("u", time, domain_range, numpy=True))
            v = np.squeeze(self.get_var("v", time, domain_range, numpy=True))
            w = np.squeeze(self.get_var("w", time, domain_range, numpy=True))

        # POSSIBLE TODO: argument xarray:bool 
        # -> might be more convenient to entail variable info
        
        return np.nanmean((u**2+v**2+w**2)/2, axis=(1, 2))
    
    def cal_enstrophy(self, time, domain_range, 
                      conv_agrid:bool=True):
        """
        Calculate enstrophy for designated domain range and specified time step.
        
        :param time: Designated time step
        :type  time: int
        :param domain_range: Designated spatial range, dimension follows (z, y, x)
        :type  domain_range: tuple, optional, default:(None, None, None, None, None, None)
        :param conv_agrid: Perform this calculation on a-grid.
        :type  conv_agrid: bool, optional, default=True
        
        :return: enstrophy
        :rtype : np.ndarray
        """
        if conv_agrid:
            eta  = self.convert_to_agrid('eta', time, domain_range)
            xi   = self.convert_to_agrid('xi', time, domain_range)
            zeta = self.convert_to_agrid('zeta', time, domain_range)
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

        # POSSIBLE TODO: argument xarray:bool 
        # -> might be more convenient to entail variable info
        
        return np.nanmean((eta**2+xi**2+zeta**2), axis=(1, 2))
    
    def cal_turb_flux(self, time, domain_range, 
                      wind_var, prop_var,
                      conv_agrid:bool=True):
        """
        Calculate turbulent flux for designated domain_range and time step based on (a-a_bar)*(b-b_bar).
        
        :param time: Designated time step
        :type  time: int
        :param domain_range: Designated spatial range, dimension follows (z, y, x)
        :type  domain_range: tuple, optional, default:(None, None, None, None, None, None)
        :param wind_var: Medium variable (mostly u/v/w)
        :type  wind_var: str
        :param prop_var: Property variable, being transported by the medium
        :type  prop_var: str
        :param conv_agrid: Perform this calculation on a-grid.
        :type  conv_agrid: bool, optional, default=True
        
        :return: turbulent flux
        :rtype : np.ndarray
        """
        if conv_agrid:
            windvar  = self.convert_to_agrid(wind_var, time)
            propvar  = self.convert_to_agrid(prop_var, time)
            windreg  = self.convert_to_agrid(wind_var, time, domain_range)
            propreg  = self.convert_to_agrid(prop_var, time, domain_range)
        else:
            windvar  = np.squeeze(self.get_var(wind_var, time, numpy=True))
            propvar  = np.squeeze(self.get_var(prop_var, time, numpy=True))
            windreg  = windvar[k1:k2, j1:j2, i1:i2].copy()
            propreg  = propvar[k1:k2, j1:j2, i1:i2].copy()
        # Calculate entire-domain mean and broadcast
        wind_bar = np.nanmean(windvar, axis=(-2, -1))
        prop_bar = np.nanmean(propvar, axis=(-2, -1))
        wind_bar = np.repeat(wind_bar[..., np.newaxis], axis=1, repeats=windreg.shape[-2])
        wind_bar = np.repeat(wind_bar[..., np.newaxis], axis=2, repeats=windreg.shape[-1])
        prop_bar = np.repeat(prop_bar[..., np.newaxis], axis=1, repeats=propreg.shape[-2])
        prop_bar = np.repeat(prop_bar[..., np.newaxis], axis=2, repeats=propreg.shape[-1])
        # Return flux
        return (windreg-wind_bar)*(propreg-prop_bar)
        
    def _find_levels_1d(self, var, conv_agrid:bool):
        """
        Find the level where var=var[z=0]+0.5. Used by finding PBL height with theta+0.5k.
        
        :param var: Variable name (normally theta, 'th')
        :type  var: str
        :param conv_agrid: Perform this calculation on a-grid, the rendered height would be based on zc with first layer omitted. 
        :type  conv_agrid: bool
        
        :return: height
        :rtype : float
        """
        target_var = var[0] + 0.5
        zidx       = np.argmax(var >= target_var)
        zidx       = np.where(np.any(var >= target_var), zidx, 0)   # set to 0 if no level satifies the condition
        if conv_agrid:
            return self.DIM['zc'][1:][zidx]
        else:
            return self.DIM['zc'][zidx]
    
    def _pbl_height_th05k(self, time, domain_range,
                          compute_mean_axis=None,
                          conv_agrid:bool=True):
        """
        Find the PBL height based on levels meeting first level theta+0.5K.
        
        :param time: Designated time step
        :type  time: int
        :param domain_range: Designated spatial range, dimension follows (z, y, x)
        :type  domain_range: tuple, optional, default:(None, None, None, None, None, None)
        :param compute_mean_axis: If assigned, mean-theta would be calculated before rendering PBL height. Options include ['x', 'y', 'xy']
        :type  compute_mean_axis: str, optional, default=None
        :param conv_agrid: Perform this calculation on a-grid.
        :type  conv_agrid: bool, optional, default=True
        
        :return: PBL height(s)
        :rtype : float or np.ndarray
        """
        if conv_agrid:
            z            = self.DIM['zc'][1:]
            th           = self.convert_to_agrid('th', time, domain_range)
        else:
            z            = self.DIM['zc']
            th           = np.squeeze(self.get_var('th', time, domain_range, numpy=True))
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
        
    def _pbl_height_dthtz(self, time, domain_range,
                          compute_mean_axis=None,
                          conv_agrid:bool=True):
        """
        Find the PBL height based on levels with maximum theta gradient.
        
        :param time: Designated time step
        :type  time: int
        :param domain_range: Designated spatial range, dimension follows (z, y, x)
        :type  domain_range: tuple, optional, default:(None, None, None, None, None, None)
        :param compute_mean_axis: If assigned, mean-theta would be calculated before rendering PBL height. Options include ['x', 'y', 'xy']
        :type  compute_mean_axis: str, optional, default=None
        :param conv_agrid: Perform this calculation on a-grid.
        :type  conv_agrid: bool, optional, default=True
        
        :return: PBL height(s)
        :rtype : float or np.ndarray
        """
        if conv_agrid:
            z            = self.DIM['zc'][1:]
            th           = self.convert_to_agrid('th', time, domain_range)
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
        
    def _pbl_height_wth(self, time, domain_range, 
                        threshold:float, 
                        conv_agrid:bool=True):
        """
        Find the PBL height based on levels with vertical theta flux changeing signs and its minimum.
        Only area-averaged result is available, by definition.
        
        :param time: Designated time step
        :type  time: int
        :param domain_range: Designated spatial range, dimension follows (z, y, x)
        :type  domain_range: tuple, optional, default:(None, None, None, None, None, None)
        :param threshold: Magnitude to be considered valid
        :type  threshold: float
        :param conv_agrid: Perform this calculation on a-grid.
        :type  conv_agrid: bool, optional, default=True
        
        :return: PBL height
        :rtype : float
        """
        if conv_agrid:
            z            = self.DIM['zc'][1:]
        else:
            z            = self.DIM['zc']
        # Method: wth
        pass
    
    def get_pbl_height(self, time, domain_range, 
                       method:str, threshold:float=0., compute_mean_axis=None, 
                       conv_agrid:bool=True):
        """
        Find the PBL height based on the designated definition.
        Currently available: th05k, max. theta gradient, vertical theta flux profile, certain value of TKE, and certain value of enstrophy.
        
        :param time: Designated time step
        :type  time: int
        :param domain_range: Designated spatial range, dimension follows (z, y, x)
        :type  domain_range: tuple, optional, default:(None, None, None, None, None, None)
        :param method: PBL height definition options
        :type  method: str
        :param threshold: Magnitude to be considered valid (meaningless for th05k and dthdz)
        :type  threshold: float, optional, default=0.
        :param compute_mean_axis: If assigned, mean-theta would be calculated before rendering PBL height. Options include ['x', 'y', 'xy'] (meaningless for wth, tke, enstrophy)
        :type  compute_mean_axis: str, optional, default=None
        :param conv_agrid: Perform this calculation on a-grid.
        :type  conv_agrid: bool, optional, default=True
        
        :return: PBL height(s)
        :rtype : float or np.ndarray
        """
        if method == 'wth':
            heights = self._pbl_height_wth(time, domain_range, conv_agrid)
        elif method == 'th05k':
            heights = self._pbl_height_th05k(time, domain_range, compute_mean_axis, conv_agrid)
        elif method == 'dthdz':
            heights = self._pbl_height_dthtz(time, domain_range, compute_mean_axis, conv_agrid)
        elif method in ('tke', 'TKE'):
            tke     = self.cal_TKE(time, domain_range, conv_agrid)
            tke     = np.where((tke-threshold)<0., 0, tke)
            hcidx   = np.nanargmin(abs(tke-threshold))
            heights = self.DIM['zc'][1:][hcidx] if conv_agrid is True else self.DIM['zc'][hcidx]
        elif method == 'enstrophy':
            enstrophy= self.cal_enstrophy(time, domain_range, conv_agrid)
            enstrophy= np.where((enstrophy-threshold)<0., 0, enstrophy)
            hcidx   = np.nanargmin(abs(enstrophy-threshold))
            heights = self.DIM['zc'][1:][hcidx] if conv_agrid is True else self.DIM['zc'][hcidx]
        else:
            raise ValueError("Unrecognized method for defining PBL height. Please choose from ['wth', 'th05k', 'dthdz', 'tke', 'enstrophy'].")
        return heights
    
    def func_time_parallel(self, 
                           func, 
                           time_steps=None, # Signature shows np.arange(0, 720, 1)
                           func_config=None,
                           cores=5):
        """
        Modify the original func_time_parallel about the partial function pre-binding, so that this method could apply to functions with parameter names other than `func_config`.

        :param func: The time-dependent function to be parallelized. It should accept two arguments: 
                     the time step `t` and a config object (containing any additional parameters).
        :type func: callable
        :param time_steps: List or array of time steps over which to apply the function. Defaults to `np.arange(0, 720, 1)`.
        :type time_steps: list or array-like, optional
        :param func_config: A dictionary or object containing additional parameters for the function.
        :type func_config: dict or object, optional
        :param cores: The number of CPU cores to use for parallel processing, defaults to 20.
        :type cores: int, optional
        :return: The combined result of applying the function to all time steps.
        :rtype: numpy.ndarray
        :raises TypeError: If `time_steps` is not a list or array-like of integers.
        
        Example:
        >>> func_config = {'domain_range':(None, None, None, None, 64, None), 
                           'method':'th05k', 
                           'compute_mean_axis':'xy'}
        >>> pbl_height  = vvmtool.func_time_parallel(func=vvmtool.get_pbl_height, time_steps=np.arange(180, 350), func_config=func_config)
        """
        # If time_steps is None, use np.arange(0, 720, 1)
        if time_steps is None:
            time_steps = np.arange(0, 720, 1)
        
        if type(time_steps) == np.ndarray:
            time_steps = time_steps.tolist()
            
        if not isinstance(time_steps, (list, tuple)):
            raise TypeError("time_steps must be a list or tuple of integers.")

        # Create a partial function that pre-binds the config to the func
        func_with_config = partial(func, **func_config)  # !! Modify this part !!

        # Use multiprocessing to fetch variable data in parallel
        with multiprocessing.Pool(processes=cores) as pool:
            results = pool.starmap(func_with_config, [(time, ) for time in time_steps])
        
        # Combine and return the results
        return np.squeeze(np.array(results))
        
# --- Test --- #
if __name__ == "__main__":
    test_case     = '/data/mlcloud/ch995334/VVM/DATA/pbl_mod_wfire_coastal_s1/'
    test_instance = VVMtools(test_case)
    # Annoucing function testing
    print("Function testing: func_time_parallel")
    # Necessary variables and get result
    test_var1   = 'w'
    test_var2   = 'th'
    time_step   = 180
    test_range  = (None, None, None, None, 64, None)
    test_config = {'domain_range':test_range, 'method':'enstrophy', 'threshold':1e-5}
    test_result = test_instance.func_time_parallel(func=test_instance.get_pbl_height, 
                                                   time_steps=np.arange(350, 400),
                                                   func_config=test_config)
    # Testing result
    print("time_step:", time_step, "domain_range:", test_range)
    print(test_result)
    print(test_result.shape)