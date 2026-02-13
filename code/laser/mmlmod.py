import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.offsetbox as offsetbox

from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.special import factorial

from laser.utils import findkink, unwrapcd
import time
import datetime


def y_min(y: npt.NDArray[np.float64]):
    '''
    Return the "nice" value of the minimum of an array y for matplotlib
    graphics.

    Parameters
    ----------
    y : numpy.float64
        Vertical graph values. Should be an array.

    Returns
    ----------
    retval : numpy.float64
        The minimum of y, rounded to a value that's convenient for a
        matplotlib graph.
    '''
    miny = np.min(y)
    shift = -int(np.log10(np.abs(miny))) + 1
    if miny >= 0.0:
        retval = np.floor(miny*10**(shift))*10**(-shift)
    else:
        retval = np.ceil(miny*10**(shift)-1)*10**(-shift)

    return retval


def y_max(y: npt.NDArray[np.float64]):
    '''
    Return the "nice" value of the maximum of an array y for matplotlib
    graphics.

    Parameters
    ----------
    y : numpy.float64
        Vertical graph values. Should be an array.

    Returns
    ----------
    retval : numpy.float64
        The maximum of y, rounded to a value that's convenient for a
        matplotlib graph.
    '''
    maxy = np.max(y)
    shift = -int(np.log10(np.abs(maxy))) + 2
    if maxy >= 0.0:
        retval = np.ceil(maxy*10**(shift))*10**(-shift)
    else:
        retval = np.floor(maxy*10**(shift)+1)*10**(-shift)

    return retval


class LaserResonatorParameters(object):
    
    '''Collect parameters describing a passive laser resonator    
       Class used to store a set of parameters required to represent a passive laser cavity.
    
    Public Attributes
    -----------------
    r_1 : numpy.float64
        Intensity reflectance of the output coupling mirror (0 < r_1 < 1)
    r_2 : numpy.float64
        Intensity reflectance of the back mirror (0 < r_2 <= 1)
    tau_pho : numpy.float64
        The group round-trip propgation time of the laser cavity
        
    Public Methods
    --------------
    get_keywords : string
        String containing the attributes of a
        LaserResonatorParameters object
    '''
    
    def __init__(self, r_1, r_2, loss_db=0.0):

        '''Initialize a LaserResonatorParameters object.
        
        Parameters
        ----------
        r_1 : numpy.float64
            Intensity reflectance of the output coupling mirror (0 < r_1 < 1)
        r_2 : numpy.float64
            Intensity reflectance of the back mirror (0 < r_2 <= 1)
        loss_db : numpy.float64
            The total round-trip propagation loss in dB (exclusive of the mirrors); default = 0.0
        '''
        
        assert r_1 > 0.0, "r_1 must be > 0."
        assert r_1 < 1.0, "r_1 must be < 1."
        assert r_2 > 0.0, "r_2 must be > 0."
        assert r_2 <= 1.0, "r_2 must be <= 1."
        assert loss_db >=0.0, "loss_db must be >= 0."
 
        self.r_1 = r_1
        self.r_2 = r_2
        self._loss_db = loss_db
        
        gamma = np.sqrt(self.r_1 * self.r_2 * 10.0**(-self._loss_db/10.0))
        self.tau_pho = -0.5/np.log(np.abs(gamma))

    def __str__(self):
        ''' Return a string containing the attributes of a LaserResonatorParameters object.
            Example:
                params = LaserResonatorParameters(r_1, r_2, loss_db)
                print(params)
        '''
        template = ( "{}"
                     "\n"
                     "R_1 = {:.{prec}}; R_2 = {:.{prec}}; loss_db = {:.{prec}} dB; tau_pho/tau_grp = {:.{prec}}"
                     "\n" )
        param_str = template.format(self.__class__.__name__, self.r_1, self.r_2, self._loss_db, self.tau_pho, prec = 3)
        
        return param_str
    
    def get_keywords(self):
        ''' Return a string containing the attributes of a LaserResonatorParameters object in a simple form
            that can be included in the Keywords field of a PDF document's Properties.
        '''
        template = ( "{}: R_1 = {:.{prec}}; R_2 = {:.{prec}}; loss_db = {:.{prec}} dB; tau_pho/tau_grp = {:.{prec}}" )
        keyword_str = template.format(self.__class__.__name__, self.r_1, self.r_2, self._loss_db, self.tau_pho, prec = 3)

        return keyword_str


class LaserMaterialParameters(object):
    
    '''Base class for collection of parameters representing a laser material
       This class establishes the interface requirements for use with
       ActiveLaserMedia.

    Public Methods
    --------------
    get_keywords : string
        String containing the attributes of a LaserMaterialParametersFLL object;
        not implemented in this base class
    '''
    
    def __init__(self):
        '''Initialize a LaserMaterialParameters object;
           not implemented in this base class.
        '''
        raise NotImplementedError
        
    def __str__(self):
        ''' Return a string displaying the attributes of a LaserMaterialParameters object.
        
        Example
        -------
        params_mat = LaserMaterialParameters(...)
        
        print(params)
        '''
        raise NotImplementedError

    def get_keywords(self):
        ''' Return a string containing the attributes of a LaserMaterialParametersFLL object in a simple form
            that can be included in the Keywords field of a PDF document's Properties.
        '''
        raise NotImplementedError

class LaserMaterialParametersFLL(LaserMaterialParameters):
    
    '''Collect parameters representing a four-level laser material
       Class used to store a set of parameters used to create a laser medium
       (amplifier or absorber) containing an ideal four-level laser material.

    Public Attributes
    -----------------
    tau_par : numpy.float64
        Longitudinal decay time ("gain recovery time") in units of
        the group round-trip time
    gbar_0 : numpy.float64
        Dimensionless small-signal (unsaturated) round-trip intensity gain,
        multiplied by 1/g_th = tau_pho
    alpha : numpy.ndarray.float64
        Linewidth enhancement factor; default = 0.0
    i_sat : numpy.ndarray.float64
        Saturation intensity adjustment; default = 1.0

    Public Methods
    --------------
    get_keywords : string
        String containing the attributes of a LaserMaterialParametersFLL object
    '''
    
    def __init__(self, tau_par, gbar_0, alpha = 0.0, i_sat = 1.0):
        '''Initialize a LaserMaterialParametersFLL object.

        Parameters
        ----------
        tau_par : numpy.float64
            Longitudinal decay time ("gain recovery time") in units of
            the group round-trip time
        gbar_0 : numpy.float64
            Dimensionless small-signal (unsaturated) round-trip intensity gain,
            multiplied by 1/g_th = tau_pho
        alpha : numpy.ndarray.float64
            Linewidth enhancement factor; default = 0.0
        i_sat : numpy.ndarray.float64
            Saturation intensity adjustment; default = 1.0
        '''
        self.tau_par = tau_par
        self.gbar_0 = gbar_0
        self.alpha = alpha
        self.i_sat = i_sat
        
    def __str__(self):
        ''' Return a string displaying the attributes of a LaserMaterialParametersFLL object.
        
        Example
        -------
        params_fll = LaserMaterialParametersFLL(tau_par, gbar_0, alpha, i_sat)
        
        print(params_fll)
        '''
        template = ( "{}"
                     "\n"
                     "tau_par/tau_grp = {:.{prec}}; Gbar_0 = {:.{prec}}; alpha = {:.{prec}}; I_sat = {:.{prec}}"
                     "\n" )
        param_str = template.format(self.__class__.__name__, self.tau_par, self.gbar_0, self.alpha, self.i_sat, prec = 3)
 
        return param_str

    def get_keywords(self):
        ''' Return a string containing the attributes of a LaserMaterialParametersFLL object in a simple form
            that can be included in the Keywords field of a PDF document's Properties.
        '''
        template = ( "\n"
                     "{}: tau_par/tau_grp = {:.{prec}}; G_0 = {:.{prec}}; LEF = {:.{prec}}; I_sat = {:.{prec}}" )
        keyword_str = template.format(self.__class__.__name__, self.tau_par, self.gbar_0, self.alpha, self.i_sat, prec = 3)

        return keyword_str

class LaserMaterialParametersQDL(LaserMaterialParameters):
   
    '''Collect parameters representing a two-level quantum dot laser material
       Class used to store a set of parameters used to create a laser medium
       (amplifier or absorber) containing a two-level (wetting layer or
       "reservoir" and ground state) quantum dot laser material.
    
    Public Attributes
    -----------------
    tau_par : numpy.float64
        Longitudinal decay time ("gain recovery time") in units of
        the group round-trip time
    gbar_0 : numpy.float64
        Dimensionless small-signal (unsaturated) round-trip intensity gain
    alpha : numpy.ndarray.float64
        Linewidth enhancement factor; default = 0.0
    i_sat : numpy.ndarray.float64
        Saturation intensity adjustment; default = 1.0

    Public Methods
    --------------
    get_keywords : string
        String containing the attributes of a LaserMaterialParametersQDL object
    '''
    
    def __init__(self, tau_gr, tau_rg, tau_par_r, tau_par_g, tau_sp, 
                 mu_r, mu_g, g_0, gbar_0, alpha = 0.0, i_sat = 1.0):
        '''Initialize a LaserMaterialParametersQDL object.

        Parameters
        ----------
        tau_gr : numpy.float64
            Electron capture time in units of the group round-trip time
        tau_rg : numpy.float64
            Electron escape time in units of the group round-trip time
        tau_par_r : numpy.float64
            Wetting layer longitudinal decay time in units of the group
            round-trip time
        tau_par_g : numpy.float64
            Ground state longitudinal decay time in units of the group
            round-trip time
        tau_sp : numpy.float64
            Electron spontaneous emission time in units of the group
            round-trip time
        mu_r : numpy.float64
            Effective wetting layer degeneracy
        mu_g : numpy.float64
            Ground state degeneracy
        g_0 : numpy.float64
            Dimensionless differential gain constant, multiplied by
            1/g_th = tau_pho
        gbar_0 : numpy.float64
            Dimensionless small-signal (unsaturated) round-trip intensity gain
        alpha : numpy.ndarray.float64
            Linewidth enhancement factor; default = 0.0
        i_sat : numpy.ndarray.float64
            Saturation intensity adjustment; default = 1.0
        '''
        self._tau_gr = tau_gr
        self._tau_rg = tau_rg
        self._tau_par_r = tau_par_r
        self._tau_par_g = tau_par_g
        self._tau_sp = tau_sp
        self._mu_r = mu_r
        self._mu_g = mu_g
        self._g_0 = g_0
        self.gbar_0 = gbar_0
        self.alpha = alpha
        self.i_sat = i_sat
        self._rate_constants()
        
    def __str__(self):
        ''' Return a string displaying the attributes of a LaserMaterialParametersFLL object.

        Example
        -------
        params_qdl = LaserMaterialParametersQDL(tau_gr, tau_rg, tau_par_r, tau_par_g, tau_sp, mu_r, mu_g, gbar_0, g_0, alpha, i_sat)
        
        print(params_qdl)
        '''
        template = ( "{}"
                     "\n"
                     "tau_gr/tau_grp = {:.{prec}}; tau_rg/tau_grp = {:.{prec}}"
                     "\n"
                     "tau_par_r/tau_grp = {:.{prec}}; tau_par_g/tau_grp = {:.{prec}}; tau_sp/tau_grp = {:.{prec}}"
                     "\n"
                     "mu_r = {:.{prec}}; mu_g = {:.{prec}}; g_0 = {:.{prec}}; Gbar_0 = {:.{prec}}; alpha = {:.{prec}}; I_sat = {:.{prec}}"
                     "\n"
                     "rho_0_r = {:.{prec}}; rho_0_g = {:.{prec}}"
                     "\n"
                     "gamma_rr = {:.{prec}}; gamma_gr = {:.{prec}}; gamma_rg = {:.{prec}}; gamma_gg = {:.{prec}}"
                     "\n"
                     "lambda_p = {:.{prec}}; lambda_m = {:.{prec}}; tau_par_eff/tau_grp = {:.{prec}}"
                     "\n" )
        param_str = template.format(self.__class__.__name__, self._tau_gr, self._tau_rg, self._tau_par_r, self._tau_par_g, self._tau_sp,
                                    self._mu_r, self._mu_g, self._g_0, self.gbar_0, self.alpha, self.i_sat,
                                    self._rho_0_r, self._rho_0_g,
                                    self._gamma_rr, self._gamma_gr, self._gamma_rg, self._gamma_gg, 
                                    self._lambda_p, self._lambda_m, self._tau_par_eff, prec = 3)
 
        return param_str

    def _rate_constants(self):
        ''' Compute zero-order populations and rate constants (private).
            Compute the zero-order populations and rate constants based on the
            unsaturated gain gbar_0. Set the value of the gain recovery time tau_par.
        '''        
        self._rho_0_g = 0.5 * ( 1.0 + self.gbar_0 / self._g_0 )
        self._gamma_rr = 1.0/self._tau_par_r + (1.0 - self._rho_0_g)/self._tau_gr + (self._mu_g/self._mu_r)*self._rho_0_g/self._tau_rg
        self._gamma_gr = self._rho_0_g/self._tau_rg + (self._mu_r/self._mu_g)*(1.0 - self._rho_0_g)/self._tau_gr
        
        self._rho_0_r = ( self._rho_0_g*(1.0/self._tau_par_g + 1.0/self._tau_rg) + self._rho_0_g**2/self._tau_sp ) / self._gamma_gr
        self._gamma_rg = self._rho_0_r/self._tau_gr + (self._mu_g/self._mu_r)*(1.0 - self._rho_0_r)/self._tau_rg
        self._gamma_gg = 1.0/self._tau_par_g + 2.0*self._rho_0_g/self._tau_sp + (1.0 - self._rho_0_r)/self._tau_rg + (self._mu_r/self._mu_g)*self._rho_0_r/self._tau_gr
        
        trace = self._gamma_rr + self._gamma_gg
        root = np.sqrt((self._gamma_rr - self._gamma_gg)**2 + 4*self._gamma_rg*self._gamma_gr)
        self._lambda_p = 0.5*(trace + root)
        self._lambda_m = 0.5*(trace - root)

        self._tau_par_eff = (self._lambda_p - self._gamma_rr) / ( self._lambda_p * (self._lambda_p - self._lambda_m) )
        self.tau_par = 1.0 / self._lambda_p

    def get_keywords(self):
        ''' Return a string containing the attributes of a LaserMaterialParametersQDL object in a simple form
            that can be included in the Keywords field of a PDF document's Properties.
        '''
        template = ( "\n"
                     "{}: tau_gr/tau_grp = {:.{prec}}; tau_rg/tau_grp = {:.{prec}};\n"
                     "tau_par_r/tau_grp = {:.{prec}}; tau_par_g/tau_grp = {:.{prec}}; tau_sp/tau_grp = {:.{prec}}\n"
                     "mu_r = {:.{prec}}; mu_g = {:.{prec}}; g_0 = {:.{prec}}; Gbar_0 = {:.{prec}}; alpha = {:.{prec}}; I_sat = {:.{prec}}" )
        keyword_str = template.format(self.__class__.__name__, self._tau_gr, self._tau_rg, self._tau_par_r, self._tau_par_g, self._tau_sp,
                                      self._mu_r, self._mu_g, self._g_0, self.gbar_0, self.alpha, self.i_sat, prec = 3)

        return keyword_str


class LaserConfiguration(object):
    '''Base class for configuration of spatial modes
       Provide support for computation of the spatial mode coupling
       coefficient kappa_{q m n} for a region of a laser cavity.
    
    Public Methods
    --------------
    kappa_qmn : numpy.ndarray.complex128
        Compute array of spatial coupling coefficients;
         not implemented in this base class
    '''

    def __init__(self, params:LaserResonatorParameters):
        '''Initialize a LaserConfiguration object.

        Parameters
        ----------
        params : LaserResonatorParameters
            Object containing the intensity reflectances of the
            cavity mirrors
        '''
        self._r_1 = params.r_1
        self._r_2 = params.r_2

    def _delta(self, q):
        '''Return coupling coefficient contribution for the case where
           the region fills the resonator.
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.complex128
            Simple spatial coupling of each mode in q
        '''
        r = self._r_1 * self._r_2
        return 1.0/( 1.0 - 2j * q * np.pi / np.log(r) )                

    def _deltap(self, q, z):
        '''Return coupling coefficient contribution for a bounded region (private).
        
        Parameters
        ----------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
        z : numpy.ndarray.float64
            A two-element array containing the z values of the
            beginning and end of the region
    
        Returns
        --------
        numpy.ndarray.complex128
            Spatial coupling of a bounded region for each mode in q
        '''
        r_1 = self._r_1
        r_2 = self._r_2
        r = r_1 * r_2
        c2 = ( r_1 * np.sqrt(r_2) * np.log(1.0/r) ) / ( (np.sqrt(r_1) + np.sqrt(r_2)) * (1 - np.sqrt(r)) )
        norm = ( c2 / (2 * (z[1] - z[0]) * np.log(1.0/r)) ) * self._delta(q)

        arg =  2j * q * np.pi - np.log(r)

        return norm * ( (np.exp(arg * z[1]) - np.exp(arg * z[0])) - (np.exp(-arg * z[1]) - np.exp(-arg * z[0])) / r_1 )
    
    def _check_z(self, z, z_lims):
        '''Check that elements of z are properly bounded (private).
        
        Parameters
        ----------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
        z : numpy.ndarray.float64
            A two-element array containing the z values of the
            beginning and end of the region
        z_lims : numpy.ndarray.float64
            A two-element array containing the z coordinates
            of mirror 1 and mirror 2, respectively.
    
        Returns
        --------
        Check that z_lims[0] <= z[0] < z[1] <= z_lims[1]; if not, then raise
        an AssertionError
        '''
        assert z[0] >= z_lims[0], "z[0] out of bounds; must be >= {}.".format(z_lims[0])
        assert z[1] > z_lims[0], "z[1] out of bounds; must be > {}.".format(z_lims[0])
        assert z[0] < z_lims[1], "z[0] out of bounds; must be < {}.".format(z_lims[1])
        assert z[1] <= z_lims[1], "z[1] out of bounds; must be <= {}.".format(z_lims[1])
        assert z[1] > z[0], "z[0] {} must be < z[1] {}".format(z[0], z[1])
    
    def kappa_qmn(self, z, q, m, n):
        '''Compute the spatial coupling coefficient kappa_{q m n}.
        
        Parameters
        ----------
        z : numpy.ndarray.float64
            A two-element array containing the z values of the
            beginning and end of the region
        q, m, n : numpy.ndarray.int32
            Arrays containing longitudinal mode indices; should have
            identical shapes
    
        Returns
        --------
        kappa_qmn : numpy.ndarray.complex128
            Array of spatial coupling coefficients;  not implemented in
            this base class
        '''
        raise NotImplementedError

    def get_delta(self, q, z):
        '''Compute the spatial coupling coefficient Delta_{2 q}(R, z).
        
        Parameters
        ----------
        q : numpy.ndarray.int32
            Array containing longitudinal mode indices
        z : numpy.ndarray.float64
            A two-element array containing the z values of the
            beginning and end of a bounded region
    
        Returns
        --------
        delta : numpy.ndarray.complex128
            Array of spatial coupling coefficients
        '''
        if z.size == 0:
            delta = self._delta(2*q)
        else:
            delta = self._deltap(2*q, z)

        return delta


class LaserConfigurationURL(LaserConfiguration):

    '''Derived class for configuration of spatial modes
       Provide support for computation of the spatial mode
       coupling coefficient kappa_{q m n} for a region of a
       unidirectional ring laser cavity.
    
    Public Methods
    --------------
    kappa_qmn : numpy.ndarray.complex128
        Compute array of spatial coupling coefficients
    '''

    def __init__(self, params:LaserResonatorParameters):
        '''Initialize a LaserConfigurationURL object.

        Parameters
        ----------
        params : LaserResonatorParameters
            Object containing the intensity reflectances of the
            cavity mirrors and the photon lifetime
        '''
        LaserConfiguration.__init__(self, params)
        
    def kappa_qmn(self, z, q, m, n):
        '''Compute the spatial coupling coefficient kappa_{q m n}.
        
        Parameters
        ----------
        z : numpy.ndarray.float64
            A two-element array containing the z values of the
            beginning and end of the region; if z is an empty array,
            then assume that the region fills the resonator
        q, m, n : numpy.ndarray.int32
            Arrays containing longitudinal mode indices; should have
            identical shapes
    
        Returns
        --------
        kappa_qmn : numpy.ndarray.complex128
            Array of spatial coupling coefficients with the same
            shape as q, m, and n
        '''
        if z.size == 0:
            kappa = np.ones(q.shape, dtype=np.dtype('complex128'))
        else:
            self._check_z(z, [0.0, 1.0])
            kappa = np.ones(q.shape, dtype=np.dtype('complex128')) * self._deltap(0, z)
            
        return kappa

class LaserConfigurationSWL(LaserConfiguration):

    '''Derived class for configuration of spatial modes
       Provide support for computation of the spatial mode
       coupling coefficient kappa_{q m n} for a region of a
       standing-wave laser cavity (ignoring spatial hole
       burning).
    
    Public Methods
    --------------
    kappa_qmn : numpy.ndarray.complex128
        Compute array of spatial coupling coefficients
    '''

    def __init__(self, params:LaserResonatorParameters):
        '''Initialize a LaserConfigurationSWL object.

        Parameters
        ----------
        params : LaserResonatorParameters
            Object containing the intensity reflectances of the
            cavity mirrors and the photon lifetime
        '''
        LaserConfiguration.__init__(self, params)
        
    def kappa_qmn(self, z, q, m, n):
        '''Compute the spatial coupling coefficient kappa_{q m n}.
        
        Parameters
        ----------
        z : numpy.ndarray.float64
            A two-element array containing the z values of the
            beginning and end of the region; if z is an empty array,
            then assume that the region fills the resonator
        q, m, n : numpy.ndarray.int32
            Arrays containing longitudinal mode indices; should have
            identical shapes
    
        Returns
        --------
        kappa_qmn : numpy.ndarray.complex128
            Array of spatial coupling coefficients with the same
            shape as q, m, and n
        '''
        if z.size == 0:
            kappa = np.ones(q.shape, dtype=np.dtype('complex128')) + self._delta(2*(m - n))
        else:
            self._check_z(z, [0.0, 0.5])
            kappa = np.ones(q.shape, dtype=np.dtype('complex128')) * self._deltap(0, z) + self._deltap(2*(m - n), z)
            
        return kappa

class LaserConfigurationSHB(LaserConfiguration):
    '''Derived class for configuration of spatial modes
       Provide support for computation of the spatial mode
       coupling coefficient kappa_{q m n} for a region of a
       standing-wave laser cavity (including spatial hole
       burning).
    
    Public Methods
    --------------
    kappa_qmn : numpy.ndarray.complex128
        Compute array of spatial coupling coefficients
    '''

    def __init__(self, params:LaserResonatorParameters):
        '''Initialize a LaserConfigurationSHB object.

        Parameters
        ----------
        params : LaserResonatorParameters
            Object containing the intensity reflectances of the
            cavity mirrors and the photon lifetime
        '''
        LaserConfiguration.__init__(self, params)
        
    def kappa_qmn(self, z, q, m, n):
        '''Compute the spatial coupling coefficient kappa_{q m n}.
        
        Parameters
        ----------
        z : numpy.ndarray.float64
            A two-element array containing the z values of the
            beginning and end of the region; if z is an empty array,
            then assume that the region fills the resonator
        q, m, n : numpy.ndarray.int32
            Arrays containing longitudinal mode indices; should have
            identical shapes
    
        Returns
        --------
        kappa_qmn : numpy.ndarray.complex128
            Array of spatial coupling coefficients with the same
            shape as q, m, and n
        '''
        if z.size == 0:
            kappa = np.ones(q.shape, dtype=np.dtype('complex128')) + self._delta(2*(m - n)) + self._delta(2*(q - m))
        else:
            self._check_z(z, [0.0, 0.5])
            kappa = ( np.ones(q.shape, dtype=np.dtype('complex128')) * self._deltap(0, z) + self._deltap(2*(m - n), z)
                    + self._deltap(2*(q - m), z) )
            
        return kappa


class FrequencyShifts(object):

    '''Estimate frequency shifts and time delays    
       Estimate the frequency shift (due to frequency-pulling and dispersion)
       and the round-trip time delay (due to dispersion) of each mode q.
    
    Public Methods
    --------------
    get_q : numpy.ndarray.int32
        Array of integers [q_min, ..., q_max]
    get_qp : numpy.ndarray.int32
        Array of integers [q_min - q_mean, ..., 0, ..., q_max - q_mean]
    get_d_omega : numpy.ndarray.float64
        Frequency shift of each mode q relative to 2 * q * pi
    get_delta_omega : numpy.ndarray.float64
        Frequency of each mode q (relative to q = 0)
    get_omega : numpy.ndarray.float64
        Normalized frequency detuning for each mode q
    get_lef : numpy.float64
        Gain-weighted linewidth enhancement factor
    get_lef_disp : list
        Effective dispersion of linewidth enhancement factor
    get_gamma : numpy.ndarray.complex128
        Complex ODE decay constant of each mode q
    get_delay : numpy.ndarray.float64
        Time delay prefactor for each mode q
    get_keywords : string
        String containing the attributes of a
        FrequencyShifts object
    '''

    #def __init__(self, q_min, q_max, tau_prp, tau_pho, disp, *materials):
    def __init__(self, q_min, q_max, tau_prp, tau_pho, disp, epsilon, *materials):
        '''Initialize a FrequencyShifts object.

        Parameters
        ----------
        q_min : int
            The minimum mode number to be used in the simulation.
        q_max : int
            The maximum mode number to be used in the simulation.
            (Mode numbers range from q_min to q_max, representing
            q_max - q_min + 1 modes.)
        tau_prp : numpy.float64
            Transverse decoherence time in units of the group round-trip time
        tau_pho : numpy.float64
            Photon lifetime in units of the group round-trip time
        disp : numpy.ndarray.float64
            List of normalized dispersion coefficients; can be empty
        epsilon : numpy.ndarray.float64
            List of polynomial coefficients for the frequency shift
        materials
            A variable-length number of arguments that represent
            materials within the laser resonator
        '''
        assert q_min < q_max, "q_min ({}) must be smaller than q_max ({}).".format(q_min, q_max)
        self._q_min = q_min
        self._q_max = q_max
        self._q_0 = (q_max + q_min) // 2
        self._tau_prp = tau_prp
        self._tau_pho = tau_pho
        self._disp = np.array(disp)
        self._epsilon = epsilon
        self._lef(materials)
    
    def __str__(self):
        ''' Return a string displaying the attributes of a FrequencyShifts object.
            Example:
                shifts = FrequencyShifts(q_max, tau_prp, tau_pho, disp)
                print(shifts)
        '''
        template = ( "\n"
                     "{}"
                     "\n"
                     "q = [{}, {}]; tau_prp/tau_grp = {:.{prec}}; tau_prp/tau_pho = {:.{prec}}; alpha = {:.{prec}}"
                     "\n")
        param_str = template.format(self.__class__.__name__, self._q_min, self._q_max, self._tau_prp, self._tau_prp/self._tau_pho, self._alpha, prec = 3)
        
        if self._disp.size != 0:
            disp_str = ''
            template = 'D_{} = {:.{prec}}; '
            for m in np.arange(self._disp.size):
                disp_str += template.format(m + 2, self._disp[m], prec = 3)
            param_str += disp_str[0:-2] + '\n'

        return param_str

    def _lef(self, materials):
        '''Compute the gain-weighted linewidth enhancement factor (private)
        
        Parameter
        ---------
        materials
            A variable-length number of arguments that represent
            materials within the laser resonator
        '''
        g = 0.0
        ga = 0.0
        for material in materials:
            g += material.gbar_0
            ga += material.gbar_0 * material.alpha
        
        self._alpha = ga / g
        self._gbar_0 = g

    def _freq_pull(self, q):
        '''Return frequency shift due to frequency pulling (private).
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        -------
        numpy.ndarray.float64
            Frequency pulling of each mode in q
        '''
        freq_pull = (self._alpha - 2 * q * np.pi * self._tau_prp) / (2 * self._tau_pho + self._tau_prp)
        
        return freq_pull

    def _freq_disp(self, q):
        '''Return frequency shift due to dispersion (private).
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.float64
            Frequency shift of each mode in q
        '''
        disp_sum = np.zeros(q.shape)
        if self._disp.size != 0:
            for m in range(self._disp.size):
                disp_sum += self._disp[m] * (2 * (q - self._q_0) * np.pi) ** (m + 2) / factorial(m + 2)

        return disp_sum

    def _time_disp(self, q):
        '''Return time delay due to dispersion (private).
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.float64
            Time delay of each mode in q
        '''
        disp_sum = np.zeros(q.shape)
        if self._disp.size != 0:
            for m in range(self._disp.size):
                disp_sum += self._disp[m] * (2 * (q - self._q_0) * np.pi) ** (m + 1) / factorial(m + 1)

        return disp_sum

    def _d_omega(self, q):
        '''Return total shift in mode frequency (private).
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.float64
            Total frequency shift of each mode in q
        '''
        #return -( self._epsilon +  self._tau_prp / (2 * self._tau_pho + self._tau_prp) ) * 2 * q * np.pi / ( 1 + self._epsilon )
        #return self._epsilon[1] * 2 * q * np.pi + self._epsilon[2] * (2 * q * np.pi)**2 - ( self._tau_prp / (2 * self._tau_pho + self._tau_prp) ) * 2 * q * np.pi # / ( 1 + self._epsilon[1] )
        #return -( self._tau_prp / (2 * self._tau_pho + self._tau_prp) ) * (self._epsilon[0] + (1 + self._epsilon[1]) * 2 * q * np.pi )
        #return -( (2 * self._tau_pho * self._epsilon[1] + self._tau_prp) / (2 * self._tau_pho + self._tau_prp) ) * 2 * (q - self._epsilon[0]) * np.pi
        return (self._epsilon[0] + self._epsilon[1] * 2 * q * np.pi) - ( self._tau_prp / (2 * self._tau_pho + self._tau_prp) ) * 2 * q * np.pi
    

    def _delta_omega(self, q):
        '''Return mode frequency relative to q = 0 (private).
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.float64
            Relative frequency of each mode in q
        '''
        #return self._epsilon[0] + 2 * q * np.pi / ( (1 + 0.5 * self._tau_prp / self._tau_pho) )
        #return ( 2 * self._tau_pho / (2 * self._tau_pho + self._tau_prp) ) * 2 * (q - self._epsilon[0]) * np.pi
        return 2 * q * np.pi + self._d_omega(q)

    def get_keywords(self):
        ''' Return a string containing the attributes of a FrequencyShifts object in a simple form
            that can be included in the Keywords field of a PDF document's Properties.
        '''
        template = ( "\n{}: q = [{}, {}]; tau_prp/tau_grp = {:.{prec}}; alpha = {:.{prec}}" )
        keyword_str = template.format(self.__class__.__name__, self._q_min, self._q_max, self._tau_prp, self._alpha, prec = 3)
        
        if self._disp.size != 0:
            disp_str = '; '
            template = 'D_{} = {:.{prec}}; '
            for m in np.arange(self._disp.size):
                disp_str += template.format(m + 2, self._disp[m], prec = 3)
            keyword_str += disp_str[0:-2]

        return keyword_str

    def get_q(self):
        '''Return mode indices corresponding to the values of q_min and q_max.
    
        Returns
        --------
        numpy.ndarray.int32
            An array containing longitudinal mode indices
        '''
        return np.arange(self._q_min, self._q_max + 1)
    
    def get_qp(self):
        '''Return mode indices corresponding to the values of q_min and q_max,
           centered on q^prime = 0.
    
        Returns
        --------
        numpy.ndarray.int32
            An array containing longitudinal mode indices
        '''
        return np.arange(self._q_min - self._q_0, self._q_max - self._q_0 + 1)
    
    def get_d_omega(self, q):
        '''Return total shift in mode frequency.
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.float64
            Total frequency shift of each mode in q
        '''
        return self._d_omega(q)
    
    def get_delta_omega(self, q):
        '''Return mode frequency relative to q = 0.
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.float64
            Relative frequency of each mode in q
        '''
        return self._delta_omega(q)
    
    def get_omega(self, q):
        '''Return normalized detuning relative to q = 0.
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.float64
            delta_omega(q) * tau_prp
        '''
        return self._delta_omega(q) * self._tau_prp
    
    def get_freq_disp(self, q):
        '''Return frequency shift due to dispersion.
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.float64
            Frequency shift of each mode in q
        '''
        return self._freq_disp(q)

    def get_lef(self):
        '''Return the gain-weighted linewidth enhancement factor.
        
        Returns
        --------
        numpy.float64
            The LEF averaged (by gain/loss) over all media in
            the resonator
        '''
        return self._alpha

    def get_lef_disp(self, m_max):
        '''Return effective dispersion of linewidth enhancement factor.
        
        Parameter
        ---------
        m_max : int32
            Return list of dispersion coefficients for m = 2, ..., m_max
    
        Returns
        --------
        list
            Dispersion coefficients for m = 2, ..., m_max
        '''
        m_list = np.arange(2, m_max + 1)
        disp = []
        for m in m_list:
            a_m = (factorial(m)  / (2 * self._tau_pho)) * (2 * self._alpha / (1 + self._alpha**2))**(m - 1) * (-self._tau_prp)**m
            disp.append(a_m)
        
        return disp

    def get_gamma(self, q):
        '''Return complex decay / shift constant for each mode q for
           derivative computation, assuming that the time step
           is scaled by tau_pho.
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.complex128
            cavity decay rate + frequency pulling + dispersion
        '''
        return -0.5 + 1j * self._tau_pho * ( self._d_omega(q) + self._freq_disp(q) )

    def get_delay(self, q):
        '''Return group round-trip-time scaling factor for each mode q
           due to time-delay dispersion.
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.complex128
            Scaling factor for derivative computation.
        '''
        return 1.0 / ( 1.0 + self._time_disp(q) )
    
    def get_epsilon(self):
        '''Return polynomial coefficients for the frequency shift.
        
        Returns
        --------
        numpy.ndarray.float64
            Polynomial coefficients for the frequency shift
        '''
        return self._epsilon
    
    def set_epsilon(self, epsilon):
        '''Set polynomial coefficients for the frequency shift.
        
        Parameter
        ---------
        epsilon : numpy.ndarray.float64
            Polynomial coefficients for the frequency shift
        '''
        self._epsilon = epsilon
    

class ActiveLaserMedium(object):

    '''Base class for configuration of amplifiers and absorbers
       Provide support for computation of the macrosopic polarization
       and the Jacobian needed by ODE solvers.
    
    Public Methods
    --------------
    f : numpy.ndarray.complex128
        Compute vector of macroscopic polarizations;
         not implemented in this base class
    dfde_ri : numpy.ndarray.float64
        Compute matrix of the gradient of the components of
        polarization with respect to the real and imaginary parts
        of the components of the electric field (Optional)
    dfde_ap : numpy.ndarray.float64
        Compute matrix of the gradient of the components of
        polarization with respect to the amplitudes and phases
        of the components of the electric field (Optional)
    get_gain : numpy.ndarray.float64
        Compute the real parts of the components of the
        round-trip unsaturated intensity gain
    get_c : numpy.ndarray.complex128
        Return the FWM frequency coupling coefficients C_{m n}
    get_delta : numpy.ndarray.complex128
        Return the spatial coupling coefficient Delta_q(R, z)
    get_kc : numpy.ndarray.complex128
        Return a 2D array of the FWM mixing spatial and frequency
        coupling coefficients kappa_{q m n}(z) C_{m n}
    get_keywords : string
        String containing the attributes of an ActiveLaserMedium object
    '''

    def __init__(self, params_mat, config_res:LaserConfiguration, freq_shifts:FrequencyShifts, z=[]):
        '''Initialize an ActiveLaserMedium object.

        Parameters
        ----------
        params_mat :
            Object containing the material parameters of the medium
        config_res : LaserConfiguration
            Object derived from LaserConfiguration base class; provides
            support for spatial coupling coefficients
        freq_shifts : FrequencyShifts
            Object providing functions that provide frequency shifts
            and time delays
        z : two-element list of floats
            List with the coordinates of the beginning and end of the
            medium
        '''
        self._config_res = config_res
        self._params_mat = params_mat
        
        self._tau_par = self._params_mat.tau_par
        self._gbar_0 = self._params_mat.gbar_0
        self._alpha = self._params_mat.alpha
        self._i_sat = self._params_mat.i_sat
        
        self._z = np.array(z)

        self._shifts = freq_shifts

        self._indices()

        self._gain_q = 0.5 * self._gbar_0 * self._lmc_q(self._q)
        self._dfde_diag = np.diag(self._gain_q)
        
    def __str__(self):
        '''Return a string displaying the attributes of an ActiveLaserMedium object.

        Example
        -------
        medium = ActiveLaserMedium(params_mat, config_res, freq_shifts, z)
        
        print(medium)
        '''
        retstr = "\n" + self.__class__.__name__ + "\n"

        if self._z.size != 0:
            template = "z = [{:.{prec}}, {:.{prec}}]\n"
            retstr += template.format(self._z[0], self._z[1], prec = 3)
        
        retstr += self._config_res.__class__.__name__ + "\n"
        retstr += self._params_mat.__str__()

        return retstr

    def _indices(self):
        ''' Prepare index arrays for computation of the polarization (private).
        
            Precompute the 1D and 3D arrays needed to compute gain,
            frequency mode coupling, and spatial coupling arrays
        '''
        self._q = self._shifts.get_q()
        self._qp = self._shifts.get_qp()
        self._q_3d, self._m_3d, self._n_3d = np.meshgrid(self._q, self._q, self._q, sparse=False, indexing='ij')
        self._index_qmn = self._q_3d - self._m_3d + self._n_3d - np.min(self._q_3d - self._m_3d + self._n_3d)

    def _lmc_q(self, q):
        '''Return Lorentzian (with LEF) of each mode in q (private).
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices

        Returns
        --------
        numpy.ndarray.complex128
            Lorentzian array mathcal{L}_q using normalized detuning relative to q = 0,
            displaced by the linewidth enhancement factor
        '''
        omega = self._shifts.get_omega(q)
        return (1 - 1j * self._alpha)**2 / (1 - 1j * (omega + self._alpha))# + 1j * self._alpha

    def _b_mn(self, m, n):
        '''Return frequency coupling coefficient of each mode in q (private).
        
        Parameter
        ---------
        m, n : numpy.ndarray.int32
            Arrays containing longitudinal mode indices
        alpha : numpy.ndarray.float64
            Linewidth enhacement factor
    
        Returns
        --------
        numpy.ndarray.complex128
            Array containing B_{m n} using normalized detuning relative to q = 0
        '''
        return 0.5 * ( self._lmc_q(m) + np.conj(self._lmc_q(n)) )

    def _c_mn(self, m, n):
        '''Return the population pulsation coefficient of each mode in q (private).
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.complex128
            Array containing C_{m - n} using normalized detuning relative to m = n
        '''
        delta_omega = self._shifts.get_delta_omega(m) - self._shifts.get_delta_omega(n)

        return 1.0 / ( 1 - 1j * delta_omega * self._tau_par )

    def _e_qmn(self, e, index):
        '''Return the components of the electric field in a 3D array (private).
           Given the components of the electrical field e and an array
           of indices, distribute those components into an array with
           the same shape as those indices to prep for vectorized computation
           of the wave-mixing amplitudes
           
        Parameters
        ----------
        e : numpy.ndarray.complex128
            A 1D array containing the components of the electric field
        index : numpy.ndarray.int32
            An nD array containing mode indices
            
        Returns
        -------
        numpy.ndarray.complex128
           Array with the same shape as index to prep for vectorized
           computation of the wave-mixing amplitudes
        '''
        ez = np.zeros(3*e.size - 2, dtype=e.dtype)
        ez[e.size - 1 : 2*e.size - 1] = e

        return ez[index]

    def get_keywords(self):
        ''' Return a string containing the attributes of an ActiveLaserMedium object in a simple form
            that can be included in the Keywords field of a PDF document's Properties.
        '''
        keyword_str = self.__class__.__name__ + ": "
        if self._z.size != 0:
            template = 'z = [{:.{prec}}, {:.{prec}}]'
            keyword_str += template.format(self._z[0], self._z[1], prec = 3) + "; "
        keyword_str += self._config_res.__class__.__name__
        keyword_str += self._params_mat.get_keywords()
    
        return keyword_str

    def get_gain(self):
        '''Return the gain of each mode for diagnostic plots
        
        Returns
        -------
        numpy.ndarray.float64
           1D array of the real parts of the unsaturated round-trip
           intensity gain for each mode
        '''
        return 2.0 * self._gain_q

    def get_c(self):
        '''Return four-wave mixing frequency coupling coefficients
           C_{m n} as a function of m - n for diagnostic plots
        
        Returns
        -------
        numpy.ndarray.complex128
           1D array of four-wave mixing frequency coupling
           coefficients between modes
        '''
        qp = self._qp
    
        return self._c_mn(qp, 0)

    def get_delta(self):
        '''Return spatial coupling coefficient Delta_{2 q}(R, z)
           for diagnostic plots
        
        Returns
        -------
        numpy.ndarray.complex128
           1D array of spatial coupling coefficients
        '''
        return self._config_res.get_delta(self._qp, self._z)
    
    def get_kc(self):
        '''Return four-wave mixing spatial and frequency coupling
           coefficients kappa_{q m n}(z) C_{m n} as a function of
           m - n and q - m for 2D diagnostic plots
        
        Returns
        -------
        numpy.ndarray.complex128
           2D array of spatial and frequency coupling coefficients
           between modes
        '''
        qp = self._qp
        qm, mn = np.meshgrid(qp, qp, sparse=False, indexing='ij')
    
        c = self._c_mn(mn, 0)
        z = self._z
        kappa = self._config_res.kappa_qmn(z, qm, 0, -mn)

        return kappa * c

    def f(self, t, e):
        '''Compute vector of macroscopic polarizations;
           not implemented in this base class
        '''
        raise NotImplementedError

class ActiveLaserMediumFWM(ActiveLaserMedium):
    '''Derived class for configuration of FWM amplifiers and absorbers
       Provide support for computation of the macrosopic polarization
       and the Jacobian needed by ODE solvers in the case of four-wave
       mixing.
    
    Public Methods
    --------------
    f : numpy.ndarray.complex128
        Compute vector of macroscopic polarizations
    dfde_ri : numpy.ndarray.float64
        Compute matrix of the gradient of the components of
        polarization with respect to the real and imaginary parts
        of the components of the electric field
    dfde_ap : numpy.ndarray.float64
        Compute matrix of the gradient of the components of
        polarization with respect to the amplitudes and phases
        of the components of the electric field
    '''
    def __init__(self, params_mat, config_res:LaserConfiguration, freq_shifts:FrequencyShifts, z=[]):
        '''Initialize an ActiveLaserMediumFWM object.

        Parameters
        ----------
        params_mat :
            Object containing the material parameters of the medium
        config_res : LaserConfiguration
            Object derived from LaserConfiguration base class; provides
            support for spatial coupling coefficients
        freq_shifts : FrequencyShifts
            Object providing functions that provide frequency shifts
            and time delays
        z : two-element list of floats
            List with the coordinates of the beginning and end of the
            medium
        '''
        ActiveLaserMedium.__init__(self, params_mat, config_res, freq_shifts, z)

        self._indices()
        self._arrays()
        
    def _indices(self):
        ''' Prepare index arrays for computation of the polarization (private).
        
            Precompute the 1D and 3D arrays needed to compute gain,
            frequency mode coupling, and spatial coupling arrays
        '''
        ActiveLaserMedium._indices(self)
        self._index_qnm = self._q_3d - self._n_3d + self._m_3d - np.min(self._q_3d - self._n_3d + self._m_3d)

    def _arrays(self):
        ''' Precompute arrays for computation of the polarization and the
            Jacobian (private).
        '''
        # Array for FWM computation of F_q
        b = self._b_mn(self._m_3d, self._n_3d)
        c = self._c_mn(self._m_3d, self._n_3d)
        kappa = self._config_res.kappa_qmn(self._z, self._q_3d, self._m_3d, self._n_3d)
        self._kbc = self._i_sat * b * kappa * c

        # Arrays for FWM computation of the Jacobian of F_q
        b = self._b_mn(self._q_3d - self._m_3d + self._n_3d, self._n_3d)
        c = self._c_mn(self._q_3d, self._m_3d)
        kappa = self._config_res.kappa_qmn(self._z, self._q_3d, self._q_3d - self._m_3d + self._n_3d, self._n_3d)
        kbc_qqmnn = self._i_sat * b * kappa * c
        
        b = self._b_mn(self._n_3d, self._m_3d)
        c = self._c_mn(self._n_3d, self._m_3d)
        kappa = self._config_res.kappa_qmn(self._z, self._q_3d, self._n_3d, self._m_3d)
        kbc_qnm = self._i_sat * b * kappa * c

        self._kbc_qmn = (0.5 * self._gbar_0) * self._lmc_q(self._q_3d) * self._kbc
        self._kbc_kln = (0.5 * self._gbar_0) * self._lmc_q(self._q_3d) * (kbc_qqmnn + self._kbc)
        self._kbc_kml = (0.5 * self._gbar_0) * self._lmc_q(self._q_3d) * kbc_qnm

    def f(self, t, e):
        '''Compute macroscopic polarizations for ODE solvers.
           For each mode, compute macroscopic polarization 
           in the case of four-wave mixing.
        
        Parameters
        ----------
        t : float
            The current time passed by the derivative function
            used by the ODE solver
        e : numpy.ndarray.complex128
            A vector containing the current electric field amplitudes
            passed by the derivative function used by the ODE solver
        
        Returns
        -------
        numpy.ndarray.complex128
            A vector containing the updated macroscopic polarization
        '''
        kbce_qmn = self._kbc * self._e_qmn(e, self._index_qmn)

        f = self._gain_q * ( e - np.dot(np.dot(kbce_qmn, np.conj(e)), e) )

        return f

    def dfde_ri(self, t, e):
        '''Compute gradient of the polarizations for ODE solvers.
           Compute matrix of the gradient of the components of
           polarization with respect to the real and imaginary parts
           of the components of the electric field in the case of
           four-wave mixing.
        
        Parameters
        ----------
        t : float
            The current time passed by the Jacobian function used by
            the ODE solver
        e : numpy.ndarray.complex128
            A vector containing the current electric field complex
            amplitudes passed by the Jacobian function used by
            the ODE solver
        
        Returns
        -------
        numpy.ndarray.float
            A 2D matrix containing the gradient of the polarization
        '''
        kbce_kln = self._kbc_kln * self._e_qmn(e, self._index_qmn)
        kbce_kml = self._kbc_kml * self._e_qmn(e, self._index_qnm)

        dfde = self._dfde_diag - np.dot(kbce_kln, np.conj(e))
        dfdc = -np.dot(kbce_kml, e)

        dfdr = dfde + dfdc
        dfdi = 1j * ( dfde - dfdc )
        
        return np.block([[dfdr.real, dfdi.real], [dfdr.imag, dfdi.imag]])

    def dfde_ap(self, t, a, p):
        '''Compute gradient of the polarizations for ODE solvers.
           Compute matrix of the gradient of the components of
           polarization with respect to the amplitudes and phases
           of the components of the electric field in the case of
           four-wave mixing.
        
        Parameters
        ----------
        t : float
            The current time passed by the Jacobian function used by
            the ODE solver
        a : numpy.ndarray.float
            A vector containing the current electric field real
            amplitudes passed by the Jacobian function used by
            the ODE solver
        p : numpy.ndarray.float
            A vector containing the current electric field phases
            passed by the Jacobian function used by the ODE solver
        
        Returns
        -------
        numpy.ndarray.float
            A 2D matrix containing the the gradient of the polarization
        '''
        z = np.exp(1j * p)
        e = a * z

        zc = np.conj(z)
        ec = np.conj(e)
        
        a_inv = 1.0 / np.fmax(1.0e-6 * np.ones(a.shape), a)
        
        e_qmn = self._e_qmn(e, self._index_qmn)
        e_qnm = self._e_qmn(e, self._index_qnm)

        kbce_qmn = self._kbc * e_qmn
        kbce_kln = self._kbc_kln * e_qmn
        kbce_kml = self._kbc_kml * e_qnm

        s_k = zc * self._gain_q * np.dot(np.dot(kbce_qmn, ec), e)
        m_kl = (np.dot(kbce_kml, e).T * zc).T * zc
        n_kl = (np.dot(kbce_kln, ec).T * zc).T * z
        
        dfda = self._dfde_diag - (m_kl + n_kl)
        dfdp = np.diag(1j * s_k) + (m_kl - n_kl) * (1j * a)
        dfada = np.diag(s_k * a_inv**2) - ((m_kl + n_kl).T * a_inv).T
        dfadp = (dfdp.T * a_inv).T
        
        return np.block([[dfda.real, dfdp.real], [dfada.imag, dfadp.imag]])

class ActiveLaserMediumFWMNew(ActiveLaserMediumFWM):
    def _b_mn(self, m, n):
        '''Return frequency coupling coefficient of each mode in q (private).
        
        Parameter
        ---------
        m, n : numpy.ndarray.int32
            Arrays containing longitudinal mode indices
        alpha : numpy.ndarray.float64
            Linewidth enhacement factor
    
        Returns
        --------
        numpy.ndarray.complex128
            Array containing B_{m n} using normalized detuning relative to q = 0
        '''
        return -( self._shifts.get_gamma(m) + np.conj(self._shifts.get_gamma(n)) )


class ActiveLaserMediumAWM(ActiveLaserMedium):
    '''Derived class for configuration of AWM amplifiers and absorbers
       Provide support for computation of the macrosopic polarization
       and the Jacobian needed by ODE solvers in the nonperturbative
       case of "all-wave" mixing.
    
    Public Methods
    --------------
    f : numpy.ndarray.complex128
        Compute vector of macroscopic polarizations
    dfde_ri : numpy.ndarray.float64
        Not (yet) implemented in this derived class
    dfde_ap : numpy.ndarray.float64
        Not (yet) implemented in this derived class
    '''
    def __init__(self, params_mat, config_res:LaserConfiguration, freq_shifts:FrequencyShifts, z=[]):
        '''Initialize an ActiveLaserMediumAWM object.

        Parameters
        ----------
        params_mat :
            Object containing the material parameters of the medium
        config_res : LaserConfiguration
            Object derived from LaserConfiguration base class; provides
            support for spatial coupling coefficients
        freq_shifts : FrequencyShifts
            Object providing functions that provide frequency shifts
            and time delays
        z : two-element list of floats
            List with the coordinates of the beginning and end of the
            medium
        '''
        ActiveLaserMedium.__init__(self, params_mat, config_res, freq_shifts, z)

        ActiveLaserMedium._indices(self)
        self._arrays()

    def _arrays(self):
        ''' Precompute arrays for computation of the polarization and the
            Jacobian (private).
            '''
        b = 0.5 * self._lmc_q(self._q_3d)
        c = self._c_mn(self._m_3d, self._n_3d)
        kappa = self._config_res.kappa_qmn(self._z, self._q_3d, self._m_3d, self._n_3d)
        self._kmc = self._i_sat * b * kappa * c

    def f(self, t, e):
        '''Compute macroscopic polarizations for ODE solvers.
           For each mode, compute macroscopic polarization 
           in the case of "all-wave" mixing.
        
        Parameters
        ----------
        t : float
            The current time passed by the derivative function
            used by the ODE solver
        e : numpy.ndarray.complex128
            A vector containing the current electric field amplitudes
            passed by the derivative function used by the ODE solver

        Returns
        -------
        numpy.ndarray.complex128
            A vector containing the updated macroscopic polarization
        '''
        kmce_qmn = self._kmc * self._e_qmn(e, self._index_qmn)

        a = np.identity(e.size, dtype=e.dtype) + np.dot(kmce_qmn, np.conj(e))
        b = np.dot(np.transpose(kmce_qmn, (0, 2, 1)), e)

        ar = a.real
        ai = a.imag
        br = b.real
        bi = b.imag

        m = np.block([[ar + br, -ai + bi], [ai + bi, ar - br]])

        ge = self._gain_q * e
        h = np.hstack((ge.real, ge.imag))

        x = np.linalg.solve(m, h)
        v = np.hsplit(x, 2)

        f = v[0] + 1j*v[1]

        return f

class ActiveLaserMediumFWH(ActiveLaserMediumFWM):
    '''Derived class for configuration of hybrid FWM amplifiers and absorbers
       Provide support for computation of the macrosopic polarization
       and the Jacobian needed by ODE solvers in the case of four-wave
       mixing. This class separates saturation from other wave mixing,
       and HAS NOT BEEN TESTED (so USE WITH CAUTION).
    
    Public Methods
    --------------
    f : numpy.ndarray.complex128
        Compute vector of macroscopic polarizations
    dfde : numpy.ndarray.complex128
        Compute matrix of the gradient of the components of
        polarization with respect to the real and imaginary parts
        of the components of the electric field
    '''
    def __init__(self, params_mat, config_res:LaserConfiguration, freq_shifts:FrequencyShifts, z=[]):
        '''Initialize an ActiveLaserMediumFWH object.

        Parameters
        ----------
        params_mat :
            Object containing the material parameters of the medium
        config_res : LaserConfiguration
            Object derived from LaserConfiguration base class; provides
            support for spatial coupling coefficients
        freq_shifts : FrequencyShifts
            Object providing functions that provide frequency shifts
            and time delays
        z : two-element list of floats
            List with the coordinates of the beginning and end of the
            medium
        '''
        ActiveLaserMediumFWM.__init__(self, params_mat, config_res, freq_shifts, z)

        self._indices()
        self._arrays()

    def _indices(self):
        ''' Prepare index arrays for computation of the polarization (private).
        
            Precompute the 1D and 3D arrays needed to compute gain,
            frequency mode coupling, and spatial coupling arrays
        '''
        ActiveLaserMediumFWM._indices(self)
        self._q_2d, self._n_2d = np.meshgrid(self._q, self._q, sparse=False, indexing='ij')
        self._index_kk = self._q_2d - np.min(self._q_2d)
        self._index_ll = self._n_2d - np.min(self._n_2d)

    def _c_q(self, q):
        '''Return the population pulsation coefficient of each mode in q (private).
           The coefficient responsible for saturation (q = 0) is set to 0.
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.complex128
            Array containing C_{m - n} using normalized detuning relative to q = 0,
            with the q = 0 elements set to 0
        '''
        return 1.0 / ( 1  - 1j * self._shifts.get_delta_omega(q) * self._tau_par ) - (q==0)

    def _arrays(self):
        ''' Precompute arrays for computation of the polarization and the
            Jacobian (private).
        '''
        ActiveLaserMediumFWM._arrays(self)
        self._kb_qnn = self._i_sat * self._config_res.kappa_qmn(self._z, self._q_2d, self._n_2d, self._n_2d) / ( 1 + self._shifts.get_omega(self._n_2d)**2 )

    def _sat(self, e):
        ''' Compute the saturation factor for a particular complex electric field.

        Parameters
        ----------
        e : numpy.ndarray.complex128
            A vector containing the current electric field amplitudes
            passed by the derivative function used by the ODE solver
        
        Returns
        -------
        numpy.ndarray.complex128
            A vector containing the saturation factor
        '''
        return 1.0 / ( np.ones(self._q.shape, dtype=np.dtype('complex128')) + np.dot(self._kb_qnn, np.abs(e)**2) )

    def f(self, t, e):
        '''Compute macroscopic polarizations for ODE solvers.
           For each mode, compute macroscopic polarization 
           in the case of four-wave mixing using the hybrid
           algorithm.
        
        Parameters
        ----------
        t : float
            The current time passed by the derivative function
            used by the ODE solver
        e : numpy.ndarray.complex128
            A vector containing the current electric field amplitudes
            passed by the derivative function used by the ODE solver
        
        Returns
        -------
        numpy.ndarray.complex128
            A vector containing the updated macroscopic polarization
        '''
        kbce_qmn = self._kbc * self._e_qmn(e, self._index_qmn)
        
        f = self._gain_q * ( self._sat(e) * e - np.dot(np.dot(kbce_qmn, np.conj(e)), e) )

        return f

    def dfde(self, t, e):
        '''Compute gradient of the polarizations for ODE solvers.
           Compute matrix of the gradient of the components of
           polarization with respect to the real and imaginary parts
           of the components of the electric field in the case of
           hybrid four-wave mixing.
        
        Parameters
        ----------
        t : float
            The current time passed by the derivative function
            used by the ODE solver
        e : numpy.ndarray.complex128
            A vector containing the current electric field amplitudes
            passed by the derivative function used by the ODE solver
        
        Returns
        -------
        numpy.ndarray.float64
            A 2D matrix containing the real and imaginary parts of the
            gradient of the polarization
        '''
        sat_k = self._sat(e)
        gse_k = self._gain_q * sat_k**2 * e
        gsekb_kl = gse_k[self._index_kk] * self._kb_qnn
        dfde_diag = np.diag( self._gain_q * sat_k )
        
        kbce_kln = self._kbc_kln * self._e_qmn(e, self._index_qmn)
        kbce_kml = self._kbc_kml * self._e_qmn(e, self._index_qnm)
        
        dfde = dfde_diag - gsekb_kl * np.conj(e[self._index_ll]) - np.dot(kbce_kln, np.conj(e))
        dfdc = -gsekb_kl * e[self._index_ll] - np.dot(kbce_kml, e)

        dfdr = dfde + dfdc
        dfdi = 1j * ( dfde - dfdc )
        
        return np.block([[dfdr.real, dfdi.real], [dfdr.imag, dfdi.imag]])

class ActiveLaserMediumAWH(ActiveLaserMediumAWM):
    '''Derived class for configuration of AWM amplifiers and absorbers
       Provide support for computation of the macrosopic polarization
       and the Jacobian needed by ODE solvers in the nonperturbative
       case of "all-wave" mixing. This class separates saturation from
       other wave mixing, and HAS NOT BEEN TESTED (so USE WITH CAUTION).
    
    Public Methods
    --------------
    f : numpy.ndarray.complex128
        Compute vector of macroscopic polarizations
    dfde : numpy.ndarray.complex128
        Not (yet) implemented in this derived class
    '''
    def __init__(self, params_mat, config_res:LaserConfiguration, freq_shifts:FrequencyShifts, z=[]):
        '''Initialize an ActiveLaserMediumAWH object.

        Parameters
        ----------
        params_mat :
            Object containing the material parameters of the medium
        config_res : LaserConfiguration
            Object derived from LaserConfiguration base class; provides
            support for spatial coupling coefficients
        freq_shifts : FrequencyShifts
            Object providing functions that provide frequency shifts
            and time delays
        z : two-element list of floats
            List with the coordinates of the beginning and end of the
            medium
        '''
        ActiveLaserMediumAWM.__init__(self, params_mat, config_res, freq_shifts, z)

        self._indices()
        self._arrays(self)

    def _indices(self):
        ''' Prepare index arrays for computation of the polarization (private).
        
            Precompute the 1D and 3D arrays needed to compute gain,
            frequency mode coupling, and spatial coupling arrays
        '''
        ActiveLaserMediumAWM._indices(self)
        self._q_2d, self._n_2d = np.meshgrid(self._q, self._q, sparse=False, indexing='ij')
        self._index_kk = self._q_2d - np.min(self._q_2d)
        self._index_ll = self._n_2d - np.min(self._n_2d)

    def _c_q(self, q):
        '''Return the population pulsation coefficient of each mode in q (private).
           The coefficient responsible for saturation (q = 0) is set to 0.
        
        Parameter
        ---------
        q : numpy.ndarray.int32
            An array containing longitudinal mode indices
    
        Returns
        --------
        numpy.ndarray.complex128
            Array containing C_{m - n} using normalized detuning relative to q = 0,
            with the q = 0 elements set to 0
        '''
        return 1.0 / ( 1  - 1j * self._shifts.get_delta_omega(q) * self._tau_par ) - (q==0)

    def _arrays(self):
        ''' Precompute arrays for computation of the polarization and the
            Jacobian (private).
        '''
        ActiveLaserMediumAWM._arrays(self)
        self._kb_qnn = ( self._i_sat * self._config_res.kappa_qmn(self._z, self._q_2d, self._n_2d, self._n_2d)
                         / ( 1 + self._shifts.get_omega(self._n_2d)**2 ) )

    def _sat(self, e):
        ''' Compute the saturation factor for a particular complex electric field.

        Parameters
        ----------
        e : numpy.ndarray.complex128
            A vector containing the current electric field amplitudes
            passed by the derivative function used by the ODE solver
        
        Returns
        -------
        numpy.ndarray.complex128
            A vector containing the saturation factor
        '''
        return 1.0 / ( np.ones(self._q.shape, dtype=np.dtype('complex128')) + np.dot(self._kb_qnn, np.abs(e)**2) )

    def f(self, t, e):
        '''Compute macroscopic polarizations for ODE solvers.
           For each mode, compute macroscopic polarization 
           in the case of "all-wave" mixing using the hybrid
           algorithm
        
        Parameters
        ----------
        t : float
            The current time passed by the derivative function
            used by the ODE solver
        e : numpy.ndarray.complex128
            A vector containing the current electric field amplitudes
            passed by the derivative function used by the ODE solver
        
        Returns
        -------
        numpy.ndarray.complex128
            A vector containing the updated macroscopic polarization
        '''
        kmce_qmn = self._kmc * self._e_qmn(e, self._index_qmn)

        a = np.identity(e.size, dtype=e.dtype) + np.dot(kmce_qmn, np.conj(e))
        b = np.dot(np.transpose(kmce_qmn, (0, 2, 1)), e)

        ar = a.real
        ai = a.imag
        br = b.real
        bi = b.imag

        m = np.block([[ar + br, -ai + bi], [ai + bi, ar - br]])

        ge = self._gain_q * self._sat(e) * e
        h = np.hstack((ge.real, ge.imag))

        x = np.linalg.solve(m, h)
        v = np.hsplit(x, 2)

        f = v[0] + 1j*v[1]

        return f


class MultiModeLaserModel(object):
    '''Solve for the stable state of a multimode laser.
       Gather all input parameters, pre-compute required arrays,
       numerically solve the evolution equation for each mode
       in the simulation, and then plot final results.
    
    Public Methods
    --------------
    modplot :
       Plot gain and frequency coupling coefficients as a pre-run
       diagnostic
    integrate :
       Using scipy.integrate.solve_ivp, numerically integrate the nonlinear,
       coupled ODEs for each longitudinal mode in the simulation
    '''
    def __init__(self, params:LaserResonatorParameters, freq_shifts:FrequencyShifts, *media:ActiveLaserMedium):
        '''Initialize a ModeLockedLaserModel object.

        Parameters
        ----------
        params : LaserResonatorParameters
            Object containing the intensity reflectances of the
            cavity mirrors (needed to coompute output fields)
        freq_shifts : FrequencyShifts
            Object providing functions that provide frequency shifts
            and time delays
        *media : ActiveLaserMedium
            A variable-length number of arguments that represent
            media within the laser resonator
        '''
        self._params = params
        self._shifts = freq_shifts

        self._media = []
        for medium in media:
            self._media.append(medium)

        self._indices()
        self._arrays()

    def __str__(self):
        '''Return a string displaying the attributes of a ModeLockedLaserModel object.

        Example
        -------
        model = ModeLockedLaserModel(params, freq_shifts, amplifier, absorber)
        
        print(model)
        '''
        retstr = self.__class__.__name__ + "\n"
        retstr += "\n" + self._params.__str__() + self._shifts.__str__()
        for medium in self._media:
            retstr += medium.__str__()
        return retstr

    def _indices(self):
        ''' Prepare index arrays (private).
        
            Precompute the 1D index arrays needed to compute the derivative
            and plot results
        '''
        self._q = self._shifts.get_q()
        self._qp = self._shifts.get_qp()

        if self._q[-1] - self._q[0] < 40:
            self._q_ticks = self._q[0::2]
            self._qp_ticks = self._qp[0::2]
        elif self._q[-1] - self._q[0] < 80:
            self._q_ticks = self._q[0::5]
            self._qp_ticks = self._qp[0::5]
        else:
            self._q_ticks = self._q[0::10]
            self._qp_ticks = self._qp[0::10]

    def _arrays(self):
        ''' Precompute arrays for computation of the derivative and the
            Jacobian (private).
        '''
        self._delay = self._shifts.get_delay(self._q)
        self._gamma = self._shifts.get_gamma(self._q)
        
        dgdr = np.diag(self._gamma)
        dgdi = 1j * np.diag(self._gamma)
        self._dgde_ri = np.block([[dgdr.real, dgdi.real], [dgdr.imag, dgdi.imag]])
        
        dgda = np.diag(self._gamma.real)
        dgdp = np.zeros_like(dgda)
        self._dgde_ap = np.block([[dgda, dgdp], [dgdp, dgdp]])
        self._a_floor = 1.0e-6 * np.ones(self._q.shape)
        self._delay_mat = np.tile(np.transpose(np.tile(self._delay, (self._q.size, 1))), (2,2))
        
    def _f(self, t, e):
        '''Compute macroscopic polarizations for _deriv(t, e) (private).
           For each mode, compute macroscopic polarization 
           by accumulating the polarization from each medium

        Parameters
        ----------
        t : float
            The current time passed by self._deriv(t, e)
        e : numpy.ndarray.complex128
            A vector containing the current electric field amplitudes
            passed by self._deriv(t, e)
        
        Returns
        -------
        numpy.ndarray.complex128
            A vector containing the total updated macroscopic polarization
        '''
        f = np.zeros_like(e)
        for medium in self._media:
            f += medium.f(t, e)
        return f

    def _dfde_ri(self, t, e):
        '''Compute gradient of the polarizations for _jac_ri(t, e) (private).
           Compute matrix of the gradient of the components of
           polarization with respect to the real and imaginary parts
           of the components of the electric field by accumulating
           the gradient from each medium
        
        Parameters
        ----------
        t : float
            The current time passed by self._jac_ri(t, e)
        e : numpy.ndarray.complex128
            A vector containing the current electric field complex
            amplitudes passed by self._jac_ri(t, e)
        
        Returns
        -------
        numpy.ndarray.float64
            A 2D matrix containing the gradient of the polarization
        '''
        dfde = np.zeros((2*e.size, 2*e.size))
        for medium in self._media:
            dfde += medium.dfde_ri(t, e)
        return dfde

    def _dfde_ap(self, t, a, p):
        '''Compute gradient of the polarizations for _jac_ap(t, a, p) (private).
           Compute matrix of the gradient of the components of
           polarization with respect to the amplitudes and phases
           of the components of the electric field by accumulating
           the gradient from each medium
        
        Parameters
        ----------
        t : float
            The current time passed by self._jac_ap(t, a, p)
        a : numpy.ndarray.float
            A vector containing the current electric field real
            amplitudes passed by self._jac_ap(t, a, p)
        p : numpy.ndarray.float
            A vector containing the current electric field phases
            passed by self._jac_ap(t, a, p)
        
        Returns
        -------
        numpy.ndarray.float64
            A 2D matrix containing the gradient of the polarization
        '''
        dfde = np.zeros((a.size + p.size, a.size + p.size))
        for medium in self._media:
            dfde += medium.dfde_ap(t, a, p)
        return dfde

    def _deriv_ri(self, t, y):
        '''Compute the derivatives of the electric field amplitudes (private).
           For each mode, compute the time derivative of the normalized
           electric field amplitude

        Parameters
        ----------
        t : float
            The current time passed by the ODE solver
        y : numpy.ndarray.float64
            A vector containing the stacked real and imaginary parts of
            the current electric field amplitudes passed by the ODE solver
        
        Returns
        -------
        numpy.ndarray.float64
            A vector containing the stacked real and imaginary parts of
            the derivative of the electric field amplitudes
        '''
        v = np.hsplit(y, 2)
        e = v[0] + 1j * v[1]

        dedt = self._delay * ( self._gamma * e + self._f(t, e) )

        return np.hstack((dedt.real, dedt.imag))

    def _deriv_ap(self, t, y):
        '''Compute the polar derivatives of the electric field amplitudes (private).
           For each mode, compute the time derivative of the normalized
           electric field amplitude

        Parameters
        ----------
        t : float
            The current time passed by the ODE solver
        y : numpy.ndarray.float64
            A vector containing the stacked amplitudes and phases of
            the current electric field amplitudes passed by the ODE solver
        
        Returns
        -------
        numpy.ndarray.float64
            A vector containing the stacked derivatives of the amplitudes
            and phases of the electric field amplitudes
        '''
        v = np.hsplit(y, 2)
        a = v[0]
        phi = v[1]

        phasor = np.exp(1j * phi)
        e = a * phasor
        f = np.conj(phasor) * self._f(t, e)
        
        dadt = self._delay * ( self._gamma.real * a + f.real )
        dpdt = self._delay * ( self._gamma.imag + f.imag / np.fmax(self._a_floor, a) )

        return np.hstack((dadt, dpdt))

    def _jac_ri(self, t, y):
        '''Compute the Jacobian for stiff solvers (private).
           Compute the Jacobian matrix of the real and imaginary parts
           of the drivative self._deriv_ri(t, y) with respect to the
           real and imaginary parts of the components of the electric field
        
        Parameters
        ----------
        t : float
            The current time passed by the ODE solver
        y : numpy.ndarray.float64
            A vector containing the stacked real and imaginary parts of
            the current electric field amplitudes passed by the ODE solver
        
        Returns
        -------
        numpy.ndarray.float64
            A 2D matrix containing the Jacobian arrayed to be
            consistent with self._deriv_ri(t, y)
        '''
        v = np.hsplit(y, 2)
        e = v[0] + 1j * v[1]

        return self._delay_mat * ( self._dgde_ri + self._dfde_ri(t, e) )

    def _jac_ap(self, t, y):
        '''Compute the Jacobian for stiff solvers (private).
           Compute the Jacobian matrix of the real and imaginary parts
           of the drivative self._deriv_ap(t, y) with respect to the
           amplitudes and phases of the components of the electric field
        
        Parameters
        ----------
        t : float
            The current time passed by the ODE solver
        y : numpy.ndarray.float64
            A vector containing the stacked ampliudes and phases of
            the current electric field amplitudes passed by the ODE solver
        
        Returns
        -------
        numpy.ndarray.float64
            A 2D matrix containing the Jacobian arrayed to be
            consistent with self._deriv_ap(t, y)
        '''
        v = np.hsplit(y, 2)
        a = v[0]
        p = v[1]

        return self._delay_mat * ( self._dgde_ap + self._dfde_ap(t, a, p) )

    def _savepath(self, dirpath, job_index, format_string):
        '''Build a string containing the path of an output file (private).
           
        Parameters
        ----------
        dirpath : string
            A path to a directory where the output file will be created
        job_index : int
            In the case of a batch of parallel jobs, include the job
            number in the file name
        format_string : string
            The filename extension indicating the format of the saved file.
        
        Returns
        -------
        string
            A string specifying the complete path (directory + file
            name) to a file containing the results of the simulation
        '''
        savepath = dirpath + self.__class__.__name__
        if job_index:
            savepath += '_{:04d}'.format(job_index)

        template = '_{}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'
        lt = time.localtime()
        savepath += template.format(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec)
        savepath += '.{}'.format(format_string)

        return savepath

    def _e_out(self, t_steps, t_max):
        '''Compute the output electric field (private).
           Compute the electric field (summed over all modes) transmitted
           through the output coupling mirror
           
        Parameters
        ----------
        usepolar : Boolean
           If True, then use polar instead of cartesian derivatives
        t_steps : int
            Number of time steps to be taken from the end of the simulation
            (redefined as t = 0)
        t_max : float
            Maximum time in units of the group round-trip propagation time
        
        Returns
        -------
        t : numpy.ndarray.float64
            Array of times when the field was evaluated
        eout : numpy.ndarray.complex128
            Array of the values of the total output field
            at times in t
        '''
        t = np.linspace(0, t_max, t_steps, endpoint=True)
    
        r_1 = self._params.r_1
        r_2 = self._params.r_2
        r = r_1 * r_2
        norm = np.sqrt( (1 - r_1) * np.sqrt(r_2) * np.log(1/r) 
                        / ( (np.sqrt(r_1) + np.sqrt(r_2)) * (1 - np.sqrt(r)) ) )

        t_mg, q_mg = np.meshgrid(t, self._q, sparse=False, indexing='ij')
        dwq = np.tile(2 * np.pi * self._q, {t.size, 1})
        phasor = np.exp(-1j * dwq * t_mg)
        eout = norm * np.sum(self._eq[-1] * phasor, axis=1)

        return t, eout

    def _psd(self, ef):
        '''Compute the power spectral density of an electric field (private).
           Given the frequency components of an electric field, compute the
           power spectral density without relying on DFTs
        
        Parameters
        ----------
        ef : numpy.ndarray.complex128
           Array containing the complex amplitudes of each field mode
        
        Returns
        -------
        freq : numpy.ndarray.float64
           Array of frequencies in units of the inverse of the group
           round-trip propagation time
        psd : numpy.ndarray.float64
           Array of spectral powers at each of the corresponding
           frequencies, normalized to freq = 0
        '''
        freq = self._q - np.min(self._q)
        if freq.size < 40:
            f_ticks = freq[0::2]
        elif freq.size < 80:
            f_ticks = freq[0::5]
        else:
            f_ticks = freq[0::10]

        ec = ef.conj()

        a = np.zeros_like(ef)
        a[0] = np.dot(ef, ec)
        for f in freq[1:]:
            a[f] = np.dot(ef[f:], ec[:(-f)])
            
        psd = 10*np.log10(np.fmax(np.abs(a[:]/a[0]), np.finfo(float).eps))
        
        return freq, f_ticks, psd

    def _unwrap(self, eq):
        '''Compute the unwrapped phase of an electric field,
           trying to remove kinks in the first derivative. If
           there are many kinks, then this algorithm may fail.
        
        Parameters
        ----------
        eq : numpy.ndarray.complex128
           Array containing the complex amplitudes of each field mode
        
        Returns
        -------
        phiu : numpy.ndarray.float64
           Unwrapped phase angle with fewer kinks in the first deriv
        '''
        q = self._q
        phiq = np.unwrap(np.angle(eq))
        
        kink_list_lr = findkink(q - q[0], phiq, np.array([-1, 0, 1]))
        kink_list_rl = findkink(q - q[0], np.flip(phiq), np.array([-1, 0, 1]))
        
        index_list = q - np.min(q)
        if kink_list_lr.size == 0 or kink_list_rl.size == 0:
            phiu = phiq
        elif kink_list_lr[0] > kink_list_rl[0]:
            phiu = unwrapcd(index_list, phiq, q)
        else:
            phiu = np.flip(unwrapcd(index_list, np.flip(phiq), q))
        
        return phiu

    def _simplot(self, usepolar, usetex, uselog, show, dirpath, job_index, format_string):
        '''Plot the results of a mode-locked laser simulation (private).
           Plots include modal amplitudes and phases as a function of
           simulation time; total output intensity and instantaneous
           frequency for several round-trip times after the simulation
           is complete; final intensity and phase of each mode; and the
           power spectral density of the total output field.
        
        Parameters
        ----------
        usepolar : Boolean
           If True, then use polar instead of cartesian derivatives
        usetex : Boolean
           If True, then use the local system's LaTeX engine to beautify graphs
        uselog : Boolean
           If True, then plot the PSD on a logarithmic (dB) scale
        show : Boolean
           If True, then show the plots at the conclusion of the simulation
           (e.g., when using a Jupyter notebook); if running Python from the
           command line or executing batch parallel jobs, set show = False
        dirpath : string
            If dirpath is not None, then it specifies a path to a directory
            where an output file will be created to store the plots
        job_index : int
            If not None, then (usually in the case of a batch of parallel jobs,
            include the job number in the file name)
        format_string : string
            The filename extension indicating the format of the saved file; if
            "pdf," then embed the paramters input to the entire simulation in
            the Keywords field of a PDF document's Properties
        '''
        def label_str(str, usetex):
            if usetex:
                return r'$' + str + '$'
            else:
                return str

        labelsize = 18
        fontsize = 24
        font = {'family' : 'serif',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : fontsize,
                }
        plt.rc('text', usetex=usetex)
        plt.rc('font', family='serif')

        fig, ax = plt.subplots(4, 2, figsize=(16, 24))
        fig.subplots_adjust(hspace=0.25, wspace = 0.25)
        for axis_row in ax:
            for axis in axis_row:
                axis.tick_params(axis='both', labelsize=labelsize)
                axis.grid(True)
                axis.xaxis.set_minor_locator(AutoMinorLocator())

        t = self._t
        eq = self._eq
        aq = self._aq
        phiq = self._phiq
        deqdt = np.gradient(eq, t, axis=0, edge_order=2)
        dphiqdt = np.imag(np.conj(eq) * deqdt / np.fmax(1.0e-09 * np.ones(aq.shape), aq**2))
        fom = deqdt / eq

        ax[0,0].set_xlabel(label_str('t', usetex), fontdict=font)
        ax[0,0].set_ylabel(label_str('\\operatorname{Re}[\\dot{E}_q(t) / E_q(t)]', usetex), fontdict=font)
        ax[0,0].set_xlim(t[0], t[-1])
        ax[0,0].plot(t, np.real(fom))
        ymax = 2 * np.max(np.abs(np.real(fom[-2])))
        ax[0,0].set_ylim(bottom=-ymax, top=ymax)
        #y_max = lambda y: np.max(np.abs(y))
        #ax[0,0].set_ylim(bottom=0, top=y_max(np.real(deqdt / np.fmax(1.0e-09 * np.ones(aq.shape), eq))))
        #ax[0,0].set_ylim(bottom=0)

        ax[0,1].set_xlabel(label_str('t', usetex), fontdict=font)
        ax[0,1].set_ylabel(label_str('\\operatorname{Im}[\\dot{E}_q(t) / E_q(t)]', usetex), fontdict=font)
        ax[0,1].set_xlim(t[0], t[-1])
        ax[0,1].plot(t, np.imag(fom))
        ymax = 2 * np.max(np.abs(np.imag(fom[-2])))
        ax[0,1].set_ylim(bottom=-ymax, top=ymax)

        # ax[0,0].set_xlabel(label_str('t', usetex), fontdict=font)
        # ax[0,0].set_ylabel(label_str('|E_q(t)|', usetex), fontdict=font)
        # ax[0,0].set_xlim(t[0], t[-1])
        # ax[0,0].plot(t, aq)
        # ax[0,0].set_ylim(bottom=0)

        # if usepolar:
        #     ax[0,1].set_xlabel(label_str('t', usetex), fontdict=font)
        #     ax[0,1].set_ylabel(label_str('\phi_q(t)', usetex), fontdict=font)
        #     ax[0,1].set_xlim(t[0], t[-1])
        #     ax[0,1].plot(t, phiq)
        # else:
        #     ax[0,1].set_xlabel(label_str('t', usetex), fontdict=font)
        #     ax[0,1].set_ylabel(label_str('\dot{\phi}_q(t)', usetex), fontdict=font)
        #     ax[0,1].set_xlim(t[0], t[-1])
        #     ax[0,1].plot(t, dphiqdt)

        t_steps = 1001
        t_max = 4.0
        if usepolar:
            y = np.hstack((self._aq[-1], self._phiq[-1]))
            dydt = self._deriv_ap(t[-1], y)
            dvdt = np.hsplit(dydt, 2)
            dpdt = dvdt[1] / self._params.tau_pho
        else:
            dpdt = dphiqdt[-1]
        
        t_rt, e_out = self._e_out(t_steps, t_max)

        ax[1,0].set_xlabel(label_str('t/\\tau_\\mathrm{ml}', usetex), fontdict=font)
        ax[1,0].set_ylabel(label_str('|E(t)|^2', usetex), fontdict=font)
        ax[1,0].plot(t_rt, np.abs(e_out)**2)
        ax[1,0].set_xlim(t_rt[0], t_rt[-1])
        ax[1,0].set_ylim(bottom=0, top=y_max(np.abs(e_out)**2))

        phi_out = np.unwrap(np.angle(e_out))
        dphidt = np.gradient(phi_out, t_rt, edge_order=2)
        
        ax[1,1].set_xlabel(label_str('t/\\tau_\\mathrm{ml}', usetex), fontdict=font)
        ax[1,1].set_ylabel(label_str('\\omega(t)', usetex), fontdict=font)
        ax[1,1].plot(t_rt, dphidt)
        ax[1,1].set_xlim(t_rt[0], t_rt[-1])

        q = self._q
        q_ticks = self._q_ticks
 
        ax[2,0].set_xlabel(label_str('q', usetex), fontdict=font)
        ax[2,0].set_xlim(q[0], q[-1])
        ax[2,0].set_xticks(q_ticks)
        if uselog:
            ax[2,0].set_ylabel(label_str('|E_q(t_f)|^2', usetex) + ' (dB)', fontdict=font)
            markerline, stemlines, baseline = ax[2,0].stem(q, 10 * (np.log10(aq[-1]**2) - np.min(np.log10(aq[-1]**2))), '-')#, use_line_collection=True)
        else:
            ax[2,0].set_ylabel(label_str('|E_q(t_f)|^2', usetex), fontdict=font)
            markerline, stemlines, baseline = ax[2,0].stem(q, aq[-1]**2, '-')#, use_line_collection=True)
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.setp(baseline, 'color','r', 'linewidth', 2)

        phi = self._unwrap(self._eq[-1])
        phi -= phi[len(phi) // 2]


        ax[2,1].set_xlabel(label_str('q', usetex), fontdict=font)
        ax[2,1].set_ylabel(label_str('\\phi_q(t_f)', usetex), fontdict=font)
        ax[2,1].set_xlim(q[0], q[-1])
        ax[2,1].set_xticks(q_ticks)
        markerline, stemlines, baseline = ax[2,1].stem(q, phi, '-')#, use_line_collection=True)
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.setp(baseline, 'color','r', 'linewidth', 2)

        freq, f_ticks, psd = self._psd(self._eq[-1])
        
        ax[3,0].set_xlabel(label_str('f', usetex), fontdict=font)
        ax[3,0].set_ylabel('power spectral density (dB)', fontdict=font)
        ax[3,0].set_xlim(freq[0], freq[-1])
        ax[3,0].set_xticks(f_ticks)
        ax[3,0].plot(freq, psd, 'o', fillstyle='none', markeredgecolor='blue', markeredgewidth=1.5)

        #d_omega = self._shifts.get_d_omega(self._q)
        if usepolar:
            y = np.hstack((self._aq[-1], self._phiq[-1]))
            dydt = self._deriv_ap(t[-1], y)
            dvdt = np.hsplit(dydt, 2)
            dpdt = dvdt[1] / self._params.tau_pho
        else:
            dpdt = dphiqdt[-1]
        #delta = np.poly1d(np.polyfit(q, d_omega - dpdt, 2), variable='q') / (2 * np.pi)
        def dnu_q(q, eq, tau_pho):
            epsilon = self._shifts.get_epsilon()
            self._shifts.set_epsilon(np.array([0, 0]))
            d_omega = self._shifts.get_d_omega(self._q) + self._shifts.get_freq_disp(q)
            fq = self._f(0, eq)
            self._shifts.set_epsilon(epsilon)
            return d_omega + np.imag(fq / eq) / tau_pho
        def func(q, epsilon_0, epsilon_1):
            return epsilon_0 + epsilon_1 * 2 * np.pi * q
        ydata = dnu_q(q, eq[-2], self._params.tau_pho)
        param, param_cov = curve_fit(func, q, ydata)
        fit = func(q, *param)

        annotation = r'$\epsilon$ = [{:.2e}, {:.2e}]'.format(-param[0], -param[1])
        ax[3,1].set_xlabel(label_str('q', usetex), fontdict=font)
        ax[3,1].set_ylabel(label_str('\\dot{\\phi}_q(t_f)', usetex), fontdict=font)
        ax[3,1].set_xlim(q[0], q[-1])
        ax[3,1].set_xticks(q_ticks)
        ax[3,1].plot(q, dpdt, label=label_str('\\dot{\\phi}_q(t_f)', usetex))
        ax[3,1].plot(q, dnu_q(q, eq[-2], self._params.tau_pho), '--', label=label_str('\\delta \\nu_q', usetex))
        ax[3,1].plot(q, fit, '-.', label='fit')
        ax[3,1].legend(fontsize=labelsize, loc='upper right')
        ob = offsetbox.AnchoredText(annotation, loc='lower left', pad=0, borderpad=0.65, prop=dict(size=fontsize))
        ob.patch.set(boxstyle='round', edgecolor='#D7D7D7', facecolor='white', alpha=0.75)
        ax[3,1].add_artist(ob)

        # ax[3,1].set_xlabel(label_str('q', usetex), fontdict=font)
        # ax[3,1].set_ylabel(label_str('\\dot{\\phi}_q(t_f)', usetex), fontdict=font)
        # ax[3,1].set_xlim(q[0], q[-1])
        # ax[3,1].set_xticks(q_ticks)
        # ax[3,1].plot(q, dpdt, label=label_str('\\dot{\\phi}_q(t_f)', usetex))
        # ax[3,1].plot(q, d_omega, label=label_str('\\delta \\omega_q', usetex))
        # ax[3,1].legend(fontsize=labelsize)
        # ax[3,1].text(q[0] + 1, np.min(np.concatenate((dpdt, d_omega))), label_str('\\tau_\\mathrm{{ml}}/\\tau_\\mathrm{{rt}} = {:.{prec}}'.format(1 - delta[1], prec=4), usetex), fontdict=font)
  
        if show:
            plt.show()
        if dirpath:
            savepath = self._savepath(dirpath, job_index, format_string)
            if format_string == 'pdf':
                with PdfPages(savepath) as pdf:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                    
                    keywords = self._params.get_keywords() + self._shifts.get_keywords()
                    for medium in self._media:
                        keywords += '\n' + medium.get_keywords()

                    d = pdf.infodict()
                    d['Title'] = self.__class__.__name__ + ' Simulation'
                    d['Author'] = 'R. G. Beausoleil'
                    d['Subject'] = 'Mode-Locked Laser Simulation Results'
                    d['Keywords'] = keywords
                    d['CreationDate'] = datetime.datetime.today()
                    d['ModDate'] = datetime.datetime.today()                    
            else:
                fig.savefig(savepath, bbox_inches='tight')
                plt.close()
            print("Saved {0}".format(savepath))
        else:
            plt.close()
            
        return -param

    def modplot(self, plot_glmc = True, plot_cd = True, plot_kc = True):
        '''Plot gain and coupling coefficients.
           Plot the gain lineshape, the four-wave mixing frequency-
           coupling coefficient, the spatial coupling coefficient,
           and the total FWM coupling coefficient as a function of
           mode number; provides a useful pre-run diagnostic

        Parameters
        ----------
        plot_glmc : Boolean
           If True, then for each active laser medium plot the real
           and imaginary parts of the lineshape function multiplied
           by the dimensionless unsaturated gain; default = True
        plot_bcd : Boolean
           If True, then for each active laser medium plot the real
           and imaginary parts of the FWM frequency-coupling coefficient
           C_{m n} as a function of m - n; default = True
        plot_kbc : Boolean
           If True, then for each active laser medium plot the real
           and imaginary parts of the FWM spatial and frequency-coupling
           coefficients kappa_{q m n}(z) C_{m n} as a 2D function of
           m - n and q - m; default = False
        '''
        labelsize = 18
        fontsize = 24
        font = {'family' : 'serif',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : fontsize,
                }
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    
        q = self._q
        qp = self._qp
        q_ticks = self._q_ticks
        qp_ticks = self._qp_ticks
        regions = np.arange(len(self._media)) + 1
        
        rows = 0
        if plot_glmc:
            rows += 1
        if plot_cd:
            rows += 1
        if plot_kc:
            rows += len(self._media)

        fig, ax = plt.subplots(rows, 2, figsize=(16, 6 * rows))
        if rows == 1:
            ax = np.array([ax])
        for axis_row in ax:
            for axis in axis_row:
                axis.tick_params(axis='both', labelsize=labelsize)
                axis.grid(True)
                axis.xaxis.set_minor_locator(AutoMinorLocator())

        row = 0
        if plot_glmc:
            ax[row,0].set_xlabel(r'$q$', fontdict=font)
            ax[row,0].set_ylabel(r'$\textrm{Re}[\widetilde{G}_q]$', fontdict=font)
            ax[row,0].set_xlim(q[0], q[-1])
            ax[row,0].set_xticks(q_ticks)
            for medium, region in zip(self._media, regions):
                gain_r = medium.get_gain().real
                ax[row,0].plot(q, gain_r, '-', label = 'Medium {} (real)'.format(region))
            ax[row,0].legend(fontsize=labelsize)

            ax[row,1].set_xlabel(r'$q$', fontdict=font)
            ax[row,1].set_ylabel(r'$\textrm{Im}[\widetilde{G}_q]$', fontdict=font)
            ax[row,1].set_xlim(q[0], q[-1])
            ax[row,1].set_xticks(q_ticks)
            for medium, region in zip(self._media, regions):
                gain_i = medium.get_gain().imag
                ax[row,1].plot(q, gain_i, '-', label = 'Medium {} (imag)'.format(region))
            ax[row,1].legend(fontsize=labelsize)

            row += 1

        if plot_cd:
            ax[row,0].set_xlabel(r'$m - n$', fontdict=font)
            ax[row,0].set_ylabel(r'$C_{m n}$', fontdict=font)
            ax[row,0].set_xlim(qp[0], qp[-1])
            ax[row,0].set_xticks(qp_ticks)
            for medium, region in zip(self._media, regions):
                c = medium.get_c()
                ax[row,0].plot(qp, c.real, '-', label = 'Medium {} (real)'.format(region))
                ax[row,0].plot(qp, c.imag, '-', label = 'Medium {} (imag)'.format(region))
            ax[row,0].legend(fontsize=labelsize)
            
            ax[row,1].set_xlabel(r'$q - m$', fontdict=font)
            ax[row,1].set_ylabel(r'$\Delta_{2(q - m)}$', fontdict=font)
            ax[row,1].set_xlim(qp[0], qp[-1])
            for medium, region in zip(self._media, regions):
                delta = medium.get_delta()
                ax[row,1].plot(qp, delta.real, '-', label = 'Medium {} (real)'.format(region))
                ax[row,1].plot(qp, delta.imag, '-', label = 'Medium {} (imag)'.format(region))
            ax[row,1].set_xlim(qp[0], qp[-1])
            ax[row,1].set_xticks(q_ticks - (q[-1] + q[0]) // 2)
            ax[row,1].legend(fontsize=labelsize)

            row += 1

        if plot_kc:
            for medium, region in zip(self._media, regions):
                kc = medium.get_kc()
                kc_r = kc.real
                kc_i = kc.imag

                ax[row,0].set_xlabel(r'$m - n$', fontdict=font)
                ax[row,0].set_ylabel(r'$q - m$', fontdict=font)
                ax[row,0].set_title(r'$\textrm{{Re}}[\kappa_{{q m n}}\, C_{{m n}}]$' + '  (Medium {})'.format(region), fontdict=font)
                ax[row,0].set_xticks(qp_ticks)
                ax[row,0].set_yticks(qp_ticks)
                cax = ax[row,0].imshow(kc_r, interpolation='bilinear', cmap=plt.get_cmap('Spectral'),
                            origin='lower', extent=[qp[0], qp[-1], qp[0], qp[-1]],
                            vmax=kc_r.max(), vmin=kc_r.min())
                cbar = fig.colorbar(cax, ax=ax[row,0])
                cbar.ax.tick_params(axis='y', labelsize=labelsize)
            
                ax[row,1].set_xlabel(r'$m - n$', fontdict=font)
                ax[row,1].set_ylabel(r'$q - m$', fontdict=font)
                ax[row,1].set_title(r'$\textrm{{Im}}[\kappa_{{q m n}}\, C_{{m n}}]$' + '  (Medium {})'.format(region), fontdict=font)
                ax[row,1].set_xticks(qp_ticks)
                ax[row,1].set_yticks(qp_ticks)
                cax = ax[row,1].imshow(kc_i, interpolation='bilinear', cmap=plt.get_cmap('Spectral'),
                            origin='lower', extent=[qp[0], qp[-1], qp[0], qp[-1]],
                            vmax=kc_i.max(), vmin=kc_i.min())
                cbar = fig.colorbar(cax, ax=ax[row,1])
                cbar.ax.tick_params(axis='y', labelsize=labelsize)

                row += 1

        plt.tight_layout(pad=2.0)
        plt.show()

    def integrate(self, t_max, eq0=None, norm=1.0e-06, method='RK23',
                  usepolar=False, usetex=True, uselog=False, show=True, dirpath=None, job_index=None, format_string='pdf'):
        '''Numerically integrate the laser mode equations of motion.
           Using scipy.integrate.solve_ivp, numerically integrate the nonlinear,
           coupled ODEs for each longitudinal mode in the simulation
        
        Parameters
        ----------
        t_max : float
           Maximum integration time, expressed in units of the group round-trip
           propagation time; the initial time is always 0 by default
        eq0 : numpy.ndarray.complex128
           Array of initial amplitudes for each mode; if None, then each mode is
           assigned an amplitude of 0.01 and a random phase angle; default = None
        method : string
           Integration method available through scipy.integrate.solve_ivp; currently
           supported methods are RK45, RK23, DOP853, Radau, BDF, and LSODA; Radau,
           BDF, and LSODA require a Jacobian function for efficient integration;
           default = 'DOP853'
        usepolar : Boolean
           If True, then use polar instead of cartesian derivatives; currently
           supported ONLY for the methods RK45, RK23, and DOP853 (i.e., methods
           that do not support a Jacobian); default = False
        usetex : Boolean
           If True, then use the local system's LaTeX engine to beautify graphs;
           default = True
        uselog : Boolean
           If True, then plot the PSD on a logarithmic (dB) scale; default = False
        show : Boolean
           If True, then show the plots at the conclusion of the simulation
           (e.g., when using a Jupyter notebook); if running Python from the
           command line or executing batch parallel jobs, set show = False;
           default = True
        dirpath : string
            If dirpath is not None, then it specifies a path to a directory
            where an output file will be created to store the plots;
            default = None
        job_index : int
            If not None, then (usually in the case of a batch of parallel jobs,
            include the job number in the file name); default = None
        format_string : string
            The filename extension indicating the format of the saved file; if
            "pdf", then embed the paramters input to the entire simulation in
            the Keywords field of a PDF document's Properties; default = "pdf"
        '''        
        if eq0 is None:
            #eq0 = 1.0e-06 * np.ones(self._q.size) * np.exp(2j * np.pi * np.random.random_sample(self._q.size))
            eq0 = norm * ( np.random.random_sample(self._q.size) + 1j * np.random.random_sample(self._q.size) )
        else:
            assert eq0.size == self._q.size, "Size of eq0 ({}) inconsistent with field mode count ({}).".format(eq0.size, self._q.size)
            assert eq0.dtype == np.dtype('complex128'), "eq0 should be a numpy array with dtype = numpy.complex128."

        tau_pho = self._params.tau_pho
        t_span = (0, t_max / tau_pho)    # Note that the derivative will step time in units of tau_pho!
        if usepolar:
            y0 = np.hstack((np.abs(eq0), np.angle(eq0)))
        else:
            y0 = np.hstack((eq0.real, eq0.imag))
            
        def hasjac(dfde):
            retval = True
            for medium in self._media:
                retval = retval and hasattr(medium, dfde)
            return retval
        
        if method == 'RK23' or method == 'RK45' or method == 'DOP853':
            jac = None
            if usepolar:
                start = time.perf_counter()
                sol = solve_ivp(self._deriv_ap, t_span, y0, method=method)
                finish = time.perf_counter()
            else:
                start = time.perf_counter()
                sol = solve_ivp(self._deriv_ri, t_span, y0, method=method)
                finish = time.perf_counter()
        elif method == 'Radau' or method == 'BDF' or method == 'LSODA':
            if usepolar:
                if hasjac('dfde_ap'):
                    jac = self._jac_ap
                else:
                    jac = None
                start = time.perf_counter()
                sol = solve_ivp(self._deriv_ap, t_span, y0, method=method, jac=jac)
                finish = time.perf_counter()
            else:
                if hasjac('dfde_ri'):
                    jac = self._jac_ri
                else:
                    jac = None
                start = time.perf_counter()
                sol = solve_ivp(self._deriv_ri, t_span, y0, method=method, jac=jac)
                finish = time.perf_counter()
        else:
            raise ValueError("Method '{}' not supported.".format(method))

        if finish - start > 1.0:
            elapsed = int(round(finish - start))
        else:
            elapsed = round(finish - start, 3)
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)

        if show:
            if hours:
                print("Elapsed time: {} hours, {} minutes, {} seconds ({})".format(hours, minutes, seconds, method))
            elif minutes and not hours:
                print("Elapsed time: {} minutes, {} seconds ({})".format(minutes, seconds, method))
            else:
                print("Elapsed time: {} seconds ({})".format(seconds, method))
            print("Derivative function calls: {} ({:.{prec}} calls/sec)".format(sol.nfev, float(sol.nfev)/elapsed, prec = 3))
            if method == 'Radau' or method == 'BDF' or method == 'LSODA':
                print("Jacobian function calls: {} ({:.{prec}} calls/sec)".format(sol.njev, float(sol.njev)/elapsed, prec = 3))

        self._t = sol.t * tau_pho
        v = np.hsplit(np.transpose(sol.y), 2)
        if usepolar:
            self._aq = v[0]
            self._phiq = v[1]
            self._eq = self._aq * np.exp(1j * self._phiq)
        else:
            self._eq = v[0] + 1j * v[1]
            self._aq = np.abs(self._eq)
            self._phiq = np.unwrap(np.angle(self._eq))
        
        sol.method = method
        sol.elapsed = elapsed           
        sol.epsilon = self._simplot(usepolar, usetex, uselog, show, dirpath, job_index, format_string)
        
        return sol
