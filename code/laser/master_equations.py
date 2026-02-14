"""
Master Equations Solver for Quantum Cascade Laser (QCL) Simulation

Simulates a QCL (or any other laser) using the master equations from Burghoff's
"Unraveling the origin of frequency modulated combs using active cavity mean-field theory"
(Optica 2020, preprint at ArXiv:2006.12397).

This reduces the Maxwell-Bloch equations of a laser down to two equations on a fine time grid.
A GUI allows certain parameters to be adjusted on the fly if desired.

It also produces a theoretical plot for the theoretical form of the extendon
(equation (7), currently only valid for a Fabry-Perot cavity with either R1=1 or R2=1).
To save simulation time it initializes the field with this as well.

Contributors: David Burghoff, Levi Humbard
Python translation: [Your Name]
"""

from typing import Dict, Tuple, Callable, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_bvp
from scipy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
import time


@dataclass
class SimulationParameters:
    """
    Data class for laser simulation parameters.
    
    Attributes:
        J: Current density (A/cm^2)
        kpp: Group velocity dispersion (ps^2/mm)
        gammaK: Kerr coefficient
        R1: Left mirror reflectivity
        R2: Right mirror reflectivity
        lm0: Center wavelength (m)
        Lc: Cavity length (m)
        df: Detuning frequency (Hz)
        mu: Dipole matrix element (C*m initially, converted in code)
        T1: Population lifetime (s)
        T2: Coherence time (s)
        n: Refractive index
        Lmod: Active region length (m)
        aw: Waveguide loss (cm^-1, converted to m^-1)
        Gamma: Overlap factor
        h: Save interval
        dtperTr: Time steps per round trip
        numTr: Number of round trips
        Crnt: Courant number
        gc: Gain curvature flag
        Ls: Gain lineshape (str or callable)
        useN2N3: Flag to use N2/N3 terms
        initphi: Initial phase configuration
        maxsave: Maximum number of save points
        plotprogress: Flag to plot progress
        plottheory: Flag to plot theory
        plotinterval: Plot update interval (s)
    """
    J: float
    kpp: float
    gammaK: float
    R1: float
    R2: float
    lm0: float
    Lc: float
    df: float
    mu: float
    T1: float
    T2: float
    n: float
    Lmod: float
    aw: float
    Gamma: float
    h: float
    dtperTr: int
    numTr: int
    Crnt: float
    gc: int
    Ls: Union[str, Callable]
    useN2N3: int
    initphi: str
    maxsave: int
    plotprogress: bool
    plottheory: bool
    plotinterval: float


@dataclass
class DerivedParameters:
    """
    Derived parameters calculated from input parameters.
    
    Attributes:
        Psat: Saturation power (W)
        g0: Small-signal gain (m^-1)
        dw: Angular detuning frequency (rad/s)
        am: Mirror loss (m^-1)
    """
    Psat: float
    g0: float
    dw: float
    am: float


@dataclass
class Solution:
    """
    Solution structure storing the field evolution.
    
    Attributes:
        z: Position array (doubled cavity) (m)
        P: Steady state power function
        t: Time array (s)
        E: Complex envelope fields
        phi: Instantaneous phase
        A: Field amplitudes
        f: Instantaneous frequency (THz)
        S: Spectrum (FFT of field)
        Sf: Spectrum frequencies (Hz)
        zn: Normal cavity position (m)
        toNormal: Function to convert to forward/backward waves
        params: Input parameters
        analytic: Analytical calculations
    """
    z: NDArray[np.float64]
    P: NDArray[np.float64]
    t: NDArray[np.float64]
    E: NDArray[np.complex128]
    phi: NDArray[np.float64]
    A: NDArray[np.float64]
    f: NDArray[np.float64]
    S: NDArray[np.complex128]
    Sf: NDArray[np.float64]
    zn: NDArray[np.float64]
    toNormal: Callable
    params: SimulationParameters
    analytic: Dict[str, Any]


# Physical constants
EC: float = 1.60217662e-19  # Electron charge (C)
C: float = 2.99792458e8  # Speed of light (m/s)
HBAR: float = 1.0545718e-34  # Reduced Planck constant (J*s)
EPS0: float = 8.85418782e-12  # Vacuum permittivity (F/m)


def fftfreqs(n: int, d: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate FFT frequency arrays.
    
    Args:
        n: Number of points
        d: Spacing between points (m)
        
    Returns:
        Tuple of (frequencies, shifted frequencies)
    """
    fs = np.fft.fftfreq(n, d)
    fss = fftshift(fs)
    return fs, fss


def master_equations(p: SimulationParameters) -> Solution:
    """
    Main simulation function for laser master equations.
    
    Simulates laser dynamics by solving coupled master equations for forward
    and backward propagating fields.
    
    Args:
        p: SimulationParameters object containing all simulation parameters
        
    Returns:
        Solution object containing field evolution and analysis
        
    Example:
        >>> params = SimulationParameters(...)  # with appropriate values
        >>> soln = master_equations(params)
    """
    
    # ========================================================================
    # Load parameters and set constants
    # ========================================================================
    J = p.J
    kpp = p.kpp
    gammaK = p.gammaK
    R1 = p.R1
    R2 = p.R2
    lm0 = p.lm0
    Lc = p.Lc
    df = p.df
    mu = p.mu
    T1 = p.T1
    T2 = p.T2
    n = p.n
    Lmod = p.Lmod
    aw = p.aw
    Gamma = p.Gamma
    h = p.h
    dtperTr = p.dtperTr
    numTr = p.numTr
    Crnt = p.Crnt
    gc = p.gc
    Ls = p.Ls
    useN2N3 = p.useN2N3
    initphi = p.initphi
    maxsave = p.maxsave
    plotprogress = p.plotprogress
    plottheory = p.plottheory
    plotinterval = p.plotinterval
    
    # ========================================================================
    # Unit conversions
    # ========================================================================
    mu = mu * EC  # Dipole matrix element to C*m
    J = J * 1e4 / EC  # Current is sheet density per time, not A/cm^2
    kpp = kpp * 1e-30 / 1e-3  # GVD to s^2/m
    aw = aw * 100  # Waveguide loss to m^-1
    
    # ========================================================================
    # Simulation parameters
    # ========================================================================
    Tr = 2 * Lc / (C / n)  # Round trip time (s)
    dt = Tr / dtperTr  # Time step size (s)
    dz = dt * C / n / Crnt  # Space step size (m)
    Ndz = round(Lc / dz)  # Number of spatial steps
    Nz = Ndz + 1  # Number of nodes
    Lc = dz * Ndz  # Cavity length (recalculated for exact fit)
    z = dz * np.arange(Ndz + 1)  # Array containing node positions (m)
    Nt = round(numTr * 2 * Lc / (C / n) / dt)  # Total number of time steps
    z2 = np.concatenate([z[:-1], Lc + z[:-1]])  # Doubled cavity for periodic BC
    
    # ========================================================================
    # Derived parameters
    # ========================================================================
    Psat = 2 * HBAR**2 / (mu**2 * T1 * T2)  # Saturation power (W)
    g0 = (mu**2 * Gamma * 2 * np.pi * C / lm0 * T1 * T2 * J / 
          (HBAR * EPS0 * C * n * Lmod))  # Small-signal gain (m^-1)
    dw = 2 * np.pi * df  # Angular detuning (rad/s)
    am = np.log(1 / R1 / R2) / (2 * Lc)  # Mirror loss (m^-1)
    
    derived = DerivedParameters(Psat=Psat, g0=g0, dw=dw, am=am)
    
    # ========================================================================
    # Gain lineshape function
    # ========================================================================
    if Ls == 'lorentzian':
        Ls_func = lambda w: 1.0 / (1 + 1j * w * T2)
    elif Ls == 'parabolic':
        Ls_func = lambda w: 1 - 1j * w * T2 + (1j * w * T2)**2
    else:
        Ls_func = Ls
    
    fs, fss = fftfreqs(len(z2), dz)
    kss = 2 * np.pi * fss
    maxLs = np.max(np.real(Ls_func(-C / n * kss)))
    
    # ========================================================================
    # Convolution function to emulate numerical diffusion
    # ========================================================================
    # Set up a convolution function to prevent numerical instability
    # (emulates numerical diffusion, behaves like gain curvature)
    zw = 1e-7  # Width parameter (m)
    z2s = fftshift(z2.copy())
    zi = np.where(z2s == 0)[0][0]
    z2s[:zi] = z2s[:zi] - 2 * Lc
    
    # Gaussian convolution kernel
    gfcn = ifftshift(1 / zw / np.sqrt(2 * np.pi) * np.exp(-0.5 * (z2s / zw)**2))
    Fgfcn = fft(gfcn)
    nrm = np.real(ifft(Fgfcn * fft(np.ones_like(z2))))
    Fgfcn = Fgfcn / nrm
    
    def cfcn(xi: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Convolution function for numerical stability."""
        return ifft(Fgfcn * fft(xi))
    
    # ========================================================================
    # Calculate steady state power function
    # ========================================================================
    X, P = steady_state_power(Ndz, dz, Lc, R1, R2, Psat, g0, aw, maxLs, C, n, dt)
    P0 = P[0]
    
    # ========================================================================
    # Gain and K functions
    # ========================================================================
    # Effective gain including waveguide loss and saturation
    geff = -aw + g0 * (1 - 1 / Psat * (P + 2 * np.flipud(P)))
    Ig = np.cumsum(geff) * dz  # Integrated gain
    Ig[Ndz:] = Ig[Ndz:] + np.log(R2)  # Add mirror gain (R2)
    K = np.exp(Ig)  # Gain kernel
    
    # Interpolated K function at half-steps
    Kh = np.zeros(4 * Ndz)
    Kh[::2] = K
    Kh[1::2] = np.interp(np.arange(1, 4 * Ndz, 2), 
                         np.arange(0, 4 * Ndz + 1, 2), 
                         np.concatenate([K, [K[0]]]))
    Kh = np.concatenate([Kh, Kh])  # Two periods of K(u/2)
    
    # Fourier representation of K for fast convolution
    FKh = fft(np.flipud(Kh[:4 * Ndz]))
    
    def Khat(f: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Fourier-based Khat function (5x faster than direct convolution).
        
        Args:
            f: Input field array
            
        Returns:
            Convolved result
        """
        ko = 1 / (4 * Lc) * np.roll(ifft(FKh * fft(np.concatenate([f, f]))), 1) * dz
        return ko[:2 * Ndz]
    
    Km = 1 / (2 * Lc) * np.sum(K) * dz  # Mean K value
    
    # ========================================================================
    # Analytical calculations and initialization
    # ========================================================================
    phia, fa, Pa, betaeff, fceo_XS, fceo_Kerr = analytic_calculations(
        kpp, Psat, T2, gc, gammaK, g0, C, n, T1, P, Lc, am, z2, P0, Km
    )
    
    # Initialize field based on initphi parameter
    if initphi == 'fundamental':
        phiI = phia
    elif initphi == 'rand':
        phiI = np.random.randn(len(phia))
    elif initphi.startswith('cosine'):
        Nharm = int(initphi[6:])
        phiI = 3 * np.cos(Nharm * 2 * np.pi / (2 * Lc) * z2)
    elif initphi.startswith('harmonic'):
        Nharm = int(initphi[8:])
        phias = np.roll(phia, -Ndz + round(2 * Ndz / Nharm / 2))
        phiI = phias[np.mod(np.arange(2 * Ndz), 2 * Ndz / Nharm).astype(int)]
    else:
        phiI = phia  # Default to fundamental
    
    # Initialize complex field envelope
    F = np.sqrt(Pa) * np.exp(1j * phiI)
    
    # Convert to normal (forward/backward) coordinates
    def toNormal(Ei: NDArray[np.complex128]) -> Tuple[NDArray[np.complex128], 
                                                        NDArray[np.complex128]]:
        """
        Convert doubled cavity field to forward and backward waves.
        
        Args:
            Ei: Field in doubled cavity representation
            
        Returns:
            Tuple of (forward wave, backward wave)
        """
        Epi = np.concatenate([Ei[:Ndz], [np.nan * Ei[Ndz]]])
        Emi = np.concatenate([[np.nan * Ei[0]], np.flipud(Ei[Ndz:])])
        return Epi, Emi
    
    def toFlipped(Epi: NDArray[np.complex128], 
                  Emi: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Convert forward and backward waves to doubled cavity representation.
        
        Args:
            Epi: Forward wave
            Emi: Backward wave
            
        Returns:
            Field in doubled cavity representation
        """
        return np.concatenate([Epi[:-1], np.flipud(Emi[1:])])
    
    # Initialize forward and backward waves
    Ep0, Em0 = toNormal(F * np.sqrt(K))
    Ep0[-1] = Em0[-1] / np.sqrt(R2)
    Em0[0] = Ep0[0] / np.sqrt(R1)
    
    # Amplitude and phase representation
    Ap0 = np.abs(Ep0)
    Am0 = np.abs(Em0)
    pp0 = np.unwrap(np.angle(Ep0))
    pm0 = np.unwrap(np.angle(Em0))
    
    # Current state variables
    Ap = Ap0.copy()
    Am = Am0.copy()
    pp = pp0.copy()
    pm = pm0.copy()
    Ep = Ep0.copy()
    Em = Em0.copy()
    
    # ========================================================================
    # Spatial derivative functions
    # ========================================================================
    def uddz(Epi: NDArray[np.complex128], 
             Emi: NDArray[np.complex128],
             R1v: float, 
             R2v: float) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """
        Upwind spatial derivative with boundary conditions.
        
        Args:
            Epi: Forward propagating field
            Emi: Backward propagating field
            R1v: Left mirror reflectivity
            R2v: Right mirror reflectivity
            
        Returns:
            Tuple of (forward derivative, backward derivative)
        """
        Epi_ext = np.concatenate([[Emi[1] * np.sqrt(R1v)], Epi])
        Emi_ext = np.concatenate([Emi, [Epi[-2] * np.sqrt(R2v)]])
        dEpo = (Epi_ext[1:] - Epi_ext[:-1]) / dz
        dEmo = (Emi_ext[1:] - Emi_ext[:-1]) / dz
        return dEpo, dEmo
    
    def ud2dz2(Epi: NDArray[np.complex128], 
               Emi: NDArray[np.complex128],
               R1v: float, 
               R2v: float) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """
        Second spatial derivative with boundary conditions.
        
        Args:
            Epi: Forward propagating field
            Emi: Backward propagating field
            R1v: Left mirror reflectivity
            R2v: Right mirror reflectivity
            
        Returns:
            Tuple of (forward 2nd derivative, backward 2nd derivative)
        """
        Epi_ext = np.concatenate([[Emi[2] * np.sqrt(R1v)], 
                                  [Emi[1] * np.sqrt(R1v)], Epi])
        Emi_ext = np.concatenate([Emi, [Epi[-2] * np.sqrt(R2v)], 
                                 [Epi[-3] * np.sqrt(R2v)]])
        dEpo = (Epi_ext[2:] - 2 * Epi_ext[1:-1] + Epi_ext[:-2]) / dz**2
        dEmo = (Emi_ext[2:] - 2 * Emi_ext[1:-1] + Emi_ext[:-2]) / dz**2
        return dEpo, dEmo
    
    # ========================================================================
    # Allocate storage for results
    # ========================================================================
    Nout = min(maxsave, Nt)
    iio = np.round(np.linspace(0, Nt - 1, Nout)).astype(int)
    
    aE = np.zeros((len(F), Nout), dtype=np.complex128)
    aA = np.zeros((len(F), Nout))
    aphi = np.zeros((len(F), Nout))
    af = np.zeros((len(F), Nout))
    aS = np.zeros((len(F), Nout), dtype=np.complex128)
    ts = np.zeros(Nout)
    
    Po = np.zeros(Nt)  # Output power
    po = np.zeros(Nt)  # Output phase
    ts_full = np.zeros(Nt)  # Full time array
    
    Po[0] = np.abs(Ap0[-1])**2
    po[0] = np.angle(pp0[-1])
    
    # ========================================================================
    # Main time-stepping loop
    # ========================================================================
    disptimer = time.time()
    
    for ii in range(1, Nt):
        # Current power
        Pp = np.abs(Ep)**2
        Pm = np.abs(Em)**2
        
        if hasattr(p, 'ampphase') and p.ampphase:
            # ================================================================
            # Amplitude-phase mode (less numerical diffusion, more prone to divergence)
            # ================================================================
            dApdz, dAmdz = uddz(Ap, Am, R1, R2)
            dAp2dz2, dAm2dz2 = ud2dz2(Ap, Am, R1, R2)
            dppdz, dpmdz = uddz(pp, pm, 1.0, 1.0)
            dpp2dz2, dpm2dz2 = ud2dz2(pp, pm, 1.0, 1.0)
            
            # Time derivatives at current time level (for higher-order terms)
            dApdt0 = -C / n * dApdz
            dAmdt0 = C / n * dAmdz
            dppdt0 = -C / n * dppdz
            dpmdt0 = C / n * dpmdz
            dAp2dt20 = (C / n)**2 * dAp2dz2
            dAm2dt20 = (C / n)**2 * dAm2dz2
            dpp2dt20 = (C / n)**2 * dpp2dz2
            dpm2dt20 = (C / n)**2 * dpm2dz2
            
            # Forward amplitude evolution
            dApdt = (C / n * (-dApdz - kpp / 2 * (2 * dApdt0 * dppdt0 + Ap * dpp2dt20)
                     - aw / 2 * Ap
                     + g0 / 2 * (Ap - 1 / Psat * (Pp + 2 * Pm) * Ap - T2 * dApdt0 
                                + gc * T2**2 * (dAp2dt20 - dppdt0**2 * Ap)
                                + 1 / Psat * ((3 * T1 + 11/2 * T2) * dAmdt0 * Am * Ap
                                + useN2N3 * ((T1 + 5/2 * T2) * Am * Am * dApdt0 
                                           + (2 * T1 + 4 * T2) * Ap * Ap * dApdt0)))))
            
            # Backward amplitude evolution
            dAmdt = (C / n * (dAmdz - kpp / 2 * (2 * dAmdt0 * dpmdt0 + Am * dpm2dt20)
                     - aw / 2 * Am
                     + g0 / 2 * (Am - 1 / Psat * (Pm + 2 * Pp) * Am - T2 * dAmdt0 
                                + gc * T2**2 * (dAm2dt20 - dpmdt0**2 * Am)
                                + 1 / Psat * ((3 * T1 + 11/2 * T2) * dApdt0 * Ap * Am
                                + useN2N3 * ((T1 + 5/2 * T2) * Ap * Ap * dAmdt0 
                                           + (2 * T1 + 4 * T2) * Am * Am * dAmdt0)))))
            
            # Forward phase evolution
            dppdt = (C / n * (-dppdz + kpp / 2 * (dAp2dt20 / Ap - dppdt0**2)
                     - 1j * gammaK * (Pp + 2 * Pm)
                     + g0 / 2 * (-T2 * dppdt0 + gc * T2**2 * (2 * dApdt0 * dppdt0 / Ap + dpp2dt20)
                                + 1 / Psat * (-(T1 + 1/2 * T2) * Am * Am * dpmdt0
                                + useN2N3 * ((T1 + 5/2 * T2) * Am * Am * dppdt0 
                                           + T2 * Ap * Ap * dppdt0)))))
            
            # Backward phase evolution
            dpmdt = (C / n * (dpmdz + kpp / 2 * (dAm2dt20 / Am - dpmdt0**2)
                     - 1j * gammaK * (Pm + 2 * Pp)
                     + g0 / 2 * (-T2 * dpmdt0 + gc * T2**2 * (2 * dAmdt0 * dpmdt0 / Am + dpm2dt20)
                                + 1 / Psat * (-(T1 + 1/2 * T2) * Ap * Ap * dppdt0
                                + useN2N3 * ((T1 + 5/2 * T2) * Ap * Ap * dpmdt0 
                                           + T2 * Am * Am * dpmdt0)))))
            
            # Time step update
            Ap = Ap + dApdt * dt
            Am = Am + dAmdt * dt
            pp = pp + dppdt * dt
            pm = pm + dpmdt * dt
            
            # Reconstruct complex fields
            Ep = Ap * np.exp(1j * pp)
            Em = Am * np.exp(1j * pm)
            
        else:
            # ================================================================
            # Linear mode (more numerical diffusion, more stable)
            # ================================================================
            dEpdz, dEmdz = uddz(Ep, Em, R1, R2)
            dEp2dz2, dEm2dz2 = ud2dz2(Ep, Em, R1, R2)
            
            # Time derivatives at current level
            dEpdt0 = -C / n * dEpdz
            dEmdt0 = C / n * dEmdz
            
            # Forward field evolution
            dEpdt = (C / n * (-dEpdz + 1j * kpp / 2 * (C / n)**2 * dEp2dz2 
                     - aw / 2 * Ep
                     - 1j * gammaK * (Pp + 2 * Pm) * Ep
                     + g0 / 2 * (Ep - 1 / Psat * (Pp + 2 * Pm) * Ep - T2 * dEpdt0 
                                + gc * T2**2 * (C / n)**2 * dEp2dz2
                                + 1 / Psat * ((2 * T1 + 3 * T2) * np.conj(dEmdt0) * Em * Ep
                                            + (T1 + 5/2 * T2) * np.conj(Em) * dEmdt0 * Ep
                                + useN2N3 * (T1 + 5/2 * T2) * np.conj(Em) * Em * dEpdt0
                                + useN2N3 * (T1 + 5/2 * T2) * np.conj(Ep) * dEpdt0 * Ep
                                + useN2N3 * (T1 + 3/2 * T2) * Ep * np.conj(dEpdt0) * Ep))))
            
            # Backward field evolution
            dEmdt = (C / n * (dEmdz + 1j * kpp / 2 * (C / n)**2 * dEm2dz2 
                     - aw / 2 * Em
                     - 1j * gammaK * (Pm + 2 * Pp) * Em
                     + g0 / 2 * (Em - 1 / Psat * (Pm + 2 * Pp) * Em - T2 * dEmdt0 
                                + gc * T2**2 * (C / n)**2 * dEm2dz2
                                + 1 / Psat * ((2 * T1 + 3 * T2) * np.conj(dEpdt0) * Ep * Em
                                            + (T1 + 5/2 * T2) * np.conj(Ep) * dEpdt0 * Em
                                + useN2N3 * (T1 + 5/2 * T2) * np.conj(Ep) * Ep * dEmdt0
                                + useN2N3 * (T1 + 5/2 * T2) * np.conj(Em) * dEmdt0 * Em
                                + useN2N3 * (T1 + 3/2 * T2) * Em * np.conj(dEmdt0) * Em))))
            
            # Time step update
            Ep = Ep + dEpdt * dt
            Em = Em + dEmdt * dt
        
        # Record output power and phase
        Po[ii] = np.abs(Ep[-1])**2
        po[ii] = np.unwrap([po[ii-1], np.angle(Ep[-1])])[1]
        ts_full[ii] = ts_full[ii-1] + dt
        
        # Save snapshots at specified intervals
        if ii in iio:
            myi = np.where(iio == ii)[0][0]
            phi = np.unwrap(np.angle(toFlipped(Ep, Em)))
            
            aE[:, myi] = toFlipped(Ep, Em)
            aphi[:, myi] = phi
            aA[:, myi] = np.abs(toFlipped(Ep, Em))
            # Instantaneous frequency (THz)
            af[:, myi] = np.diff(np.concatenate([[phi[-1]], phi])) / dz * (-C / n) / (2 * np.pi) / 1e12
            aS[:, myi] = fftshift(fft(toFlipped(Ep, Em)))
            ts[myi] = ts_full[ii]
        
        # Optional: plot progress (would need matplotlib integration)
        if plotprogress and (time.time() - disptimer) > plotinterval:
            # Plot progress here (simplified in this translation)
            disptimer = time.time()
    
    # ========================================================================
    # Construct solution structure
    # ========================================================================
    soln = Solution(
        z=z2,
        P=P,
        t=ts,
        E=aE,
        phi=aphi,
        A=aA,
        f=af,
        S=aS,
        Sf=-C / n * fss,
        zn=z,
        toNormal=toNormal,
        params=p,
        analytic={
            'phi': phia,
            'f': fa,
            'P': Pa,
            'betaeff': betaeff,
            'fceo_XS': fceo_XS,
            'fceo_Kerr': fceo_Kerr
        }
    )
    
    return soln


def steady_state_power(Ndz: int, dz: float, Lc: float, R1: float, R2: float,
                       Psat: float, g0: float, aw: float, maxLs: float,
                       C: float, n: float, dt: float) -> Tuple[NDArray[np.float64], 
                                                                 NDArray[np.float64]]:
    """
    Calculate steady-state power distribution using boundary value problem solver.
    
    Solves the coupled ODEs for forward and backward power in steady state.
    
    Args:
        Ndz: Number of spatial steps
        dz: Spatial step size (m)
        Lc: Cavity length (m)
        R1: Left mirror reflectivity
        R2: Right mirror reflectivity
        Psat: Saturation power (W)
        g0: Small-signal gain (m^-1)
        aw: Waveguide loss (m^-1)
        maxLs: Maximum of gain lineshape
        C: Speed of light (m/s)
        n: Refractive index
        dt: Time step (s)
        
    Returns:
        Tuple of (node positions, power distribution)
    """
    # Initial guess parameters
    xhigh = Lc
    xlow = 0.0
    fR_const = Psat
    
    def guess(x: float) -> NDArray[np.float64]:
        """Initial guess for power distribution."""
        omega = (1 / xhigh) * np.log(10)
        return np.array([fR_const * np.exp(omega * x),
                        0.2 * fR_const * np.exp(omega * x)])
    
    def bvp_ode(x: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        ODEs for steady-state power.
        
        y[0]: Forward power
        y[1]: Backward power
        """
        dydt = np.array([
            g0 * (1 - (1 / Psat) * (y[0] + 2 * y[1])) * y[0] - aw * y[0],
            -(g0 * (1 - (1 / Psat) * (y[1] + 2 * y[0])) * y[1] - aw * y[1])
        ])
        return dydt
    
    def bvp_bc(ya: NDArray[np.float64], yb: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Boundary conditions.
        
        Left: ya[0] = R1 * ya[1]  (forward = R1 * backward)
        Right: yb[1] = R2 * yb[0]  (backward = R2 * forward)
        """
        return np.array([ya[0] - R1 * ya[1], yb[1] - R2 * yb[0]])
    
    # Initial mesh
    x_init = np.linspace(xlow, xhigh, 100)
    y_init = np.array([guess(x) for x in x_init]).T
    
    # Solve BVP
    sol = solve_bvp(bvp_ode, bvp_bc, x_init, y_init)
    
    # Evaluate at grid points
    xint = np.linspace(xlow, xhigh, Ndz)
    Sxint = sol.sol(xint)
    
    # Reshape: extend to doubled cavity
    Node_xint = (Ndz / xhigh) * xint
    Node_Extend_xint = np.concatenate([Node_xint, 2 * Ndz + Node_xint])
    plt_Array = np.concatenate([Sxint[0, :], np.flipud(Sxint[1, :])])
    
    return Node_Extend_xint, plt_Array


def analytic_calculations(kpp: float, Psat: float, T2: float, gc: int,
                          gammaK: float, g0: float, C: float, n: float,
                          T1: float, P: NDArray[np.float64], Lc: float,
                          am: float, z2: NDArray[np.float64], P0: float,
                          Km: float) -> Tuple[NDArray[np.float64], NDArray[np.float64],
                                               NDArray[np.float64], float, float, float]:
    """
    Perform analytical calculations for comparison.
    
    Calculates theoretical predictions for phase, frequency, power, and
    carrier-envelope offset frequency contributions.
    
    Args:
        kpp: Group velocity dispersion (s^2/m)
        Psat: Saturation power (W)
        T2: Coherence time (s)
        gc: Gain curvature flag
        gammaK: Kerr coefficient
        g0: Small-signal gain (m^-1)
        C: Speed of light (m/s)
        n: Refractive index
        T1: Population lifetime (s)
        P: Power distribution
        Lc: Cavity length (m)
        am: Mirror loss (m^-1)
        z2: Doubled cavity position array (m)
        P0: Initial power (W)
        Km: Mean K value
        
    Returns:
        Tuple of (phase, frequency, power, effective beta, fceo from XS, fceo from Kerr)
    """
    # Effective dispersion including Kerr contribution
    kppeff = kpp - 2 * Psat * (T2**2)**gc * gammaK
    betaeff = kppeff * (C / n)**3
    
    # Cross-saturation parameter
    gm = (g0 / (2 * Psat) * (C / n)**2 * (T1 + 0.5 * T2) * 
          (P[-1] - P[0]) / (4 * Lc * P[0]))
    
    # Self-saturation parameter
    r = g0 / (2 * Psat) * C / n * 3 * np.mean(P) / P0
    
    # Amplitude with gain compression
    Aa = np.sqrt(P0) / np.sqrt(1 + gm / (2 * r))
    
    # Effective Kerr parameter
    gammaKp = gammaK * C / n * 3 * Km
    
    # Gain dispersion
    Dg = 2 * g0 * T2**2 * (C / n)**3 * gc
    
    # Analytical phase (parabolic)
    phia = 0.5 * gm / betaeff * Aa**2 * (z2 - Lc)**2
    
    # Instantaneous frequency (THz)
    fa = -C / (4 * np.pi * n) * gm / (betaeff / 2) * np.abs(Aa)**2 * (z2 - Lc)
    
    # CEO frequency contribution from cross-saturation
    fceo_XS = -1 / (24 * np.pi) * gm / (betaeff / 2) * Lc**2 * gm * np.abs(Aa)**4
    
    # CEO frequency contribution from Kerr effect
    fceo_Kerr = -1 / (2 * np.pi) * gammaK * C / n * 3 * Km * np.abs(Aa)**2
    
    # Add CEO frequency to instantaneous frequency
    fa = fa + fceo_XS + fceo_Kerr
    
    # Power including gain dispersion effects
    Pa = Aa**2 - Dg / (4 * r) * (Aa**2 * gm * (z2 - Lc) / betaeff)**2
    
    # Stability number (for diagnostics)
    stabnum = P0 * Dg / r * (gm * Lc / 2 / (betaeff / 2 - gammaKp * Dg / 4 / r))**2
    
    # Bandwidth estimate
    BW = 1 / (24 * np.pi) * 2 * Lc * am * (g0 - aw - am) / kppeff * (T1 + 0.5 * T2)
    
    print(f"Stability number: {stabnum}")
    print(f"Bandwidth (Hz): {BW}")
    
    return phia, fa, Pa, betaeff, fceo_XS, fceo_Kerr
