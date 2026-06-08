import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import odeint
from scipy.special import factorial
from scipy.signal import savgol_filter
from enum import Enum
from laser.utils import *
from laser.tools import InjectedFieldConst
import time


class Config(Enum):
    URL, SWL, SHB = range(3)


class LaserParameters(object):
    def __init__(self, q_max = 0, npts = 0, inj_amplitude = 0.0, inj_shift = 0.0, r_1 = 0.0, r_2 = 0.0, tau_prp = 0.0, tau_par = [0.0],
                 g_0 = [0.0], z_list = [], disp = []):
        self._q_max = q_max
        self._npts = npts
        self._inj_amplitude = inj_amplitude
        self._shift = inj_shift

        self._r_1 = r_1
        self._r_2 = r_2
        self._r  = self._r_1 * self._r_2
        self._c2 = (self._r_1 * np.sqrt(self._r_2) * np.log(1.0/(self._r))) / ( (np.sqrt(self._r_1) + np.sqrt(self._r_2)) * (1 - np.sqrt(self._r)) )
        
        self._tau_pho = -1.0/np.log(self._r)
        self._tau_prp = tau_prp

        assert len(tau_par) == len(g_0), "Inconsistent gain region parameter lists."
        self._tau_par = np.array(tau_par)
        self._g_0 = np.array(g_0)

        #(z_0, w_0) = leggauss(self._npts)
        #self._z_lg = np.zeros((self._g_0.size, self._npts))
        #self._w_lg = np.zeros((self._g_0.size, npts))
        #if not z_list:
            #self._z_list = np.array([[] for _ in range(len(g_0))])
            #a = 0.0
            #b = 0.5
            #for j in np.arange(self._z_list.size):
                #self._z_lg[j] = 0.5 * ( (b - a) * z_0 + (a + b) )
                #self._w_lg[j] = 0.5 * (b - a) * w_0
        #else:
            #assert len(g_0) == len(z_list), "Inconsistent gain region coordinate list."
            #self._z_list = np.array(z_list)
            #for j in np.arange(len(z_list)):
                #z = self._z_list[j]
                #a = z[0]
                #b = z[1]
                #self._z_lg[j] = 0.5 * ( (b - a) * z_0 + (a + b) )
                #self._w_lg[j] = 0.5 * (b - a) * w_0

        if not z_list:
            self._z_list = np.array([[] for _ in range(len(g_0))])
        else:
            assert len(g_0) == len(z_list), "Inconsistent gain region coordinate list."
            self._z_list = np.array(z_list)

        self._disp = np.array(disp)
        
    def __str__(self):
        template = 'q_max:\t{}\tnpts:\t{}\nR_1:\t{:.{prec}}\nR_2:\t{:.{prec}}\n\n'
        param_str = template.format(self._q_max, self._npts, self._r_1, self._r_2, prec = 3)
        template = 'tau_pho/tau_grp:\t{:.{prec}}\ntau_prp/tau_grp:\t{:.{prec}}\ntau_prp/tau_pho:\t{:.{prec}}\n\n'
        param_str += template.format(self._tau_pho, self._tau_prp, self._tau_prp/self._tau_pho, prec = 3)
 
        reg_str = '[tau_par/tau_grp, g_0, z_list] = '
        for region in np.arange(self._g_0.size):
            if self._z_list[region].size == 0:
                template = '[{:.{prec}}, {:.{prec}}, [ ]], '
                reg_str += template.format(self._tau_par[region], self._g_0[region], prec = 3)
            else:
                template = '[{:.{prec}}, {:.{prec}}, [{:.{prec}}, {:.{prec}}]], '
                reg_str += template.format(self._tau_par[region], self._g_0[region], self._z_list[region][0], self._z_list[region][1], prec = 3)
        param_str += reg_str[0:-2] + '\n\n'

        if self._disp.size != 0:
            disp_str = ''
            template = 'D_{} = {:.{prec}}; '
            for m in np.arange(self._disp.size):
                disp_str += template.format(m + 2, self._disp[m], prec = 3)
            param_str += disp_str[0:-2] + '\n\n'

        return param_str

    def _fig(self, ax, usetex, font):
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 5)

        if usetex:
            param_str_04 = r'$r_1 = {:.{prec}}, r_2 = {:.{prec}}$'.format(self._r_1, self._r_2, prec = 3)
            ax.text(0, 4, param_str_04, fontdict=font)

            param_str_03 = r'$\tau_p / \tau_g = {:.{prec}}$'.format(self._tau_pho, prec = 3)
            ax.text(0, 3, param_str_03, fontdict=font)

            param_str_02 = r'$\tau_\perp / \tau_g = {:.{prec}}$'.format(self._tau_prp, prec = 3)
            ax.text(0, 2, param_str_02, fontdict=font)

            param_str_01 = r'[$\tau_\parallel / \tau_g, g_0, z] =$ '
            for region in np.arange(self._g_0.size):
                if self._z_list[region].size == 0:
                    template = r'$[{:.{prec}}, {:.{prec}}, [~]]$, '
                    param_str_01 += template.format(self._tau_par[region], self._g_0[region], prec = 3)
                else:
                    template = r'$[{:.{prec}}, {:.{prec}}, [{:.{prec}}, {:.{prec}}]]$, '
                    param_str_01 += template.format(self._tau_par[region], self._g_0[region], self._z_list[region][0], self._z_list[region][1], prec = 3)
            ax.text(0, 1, param_str_01[0:-2], fontdict=font)

            if self._disp:
                param_str_00 = ''
                template = r'$D_{} = {:.{prec}} \times 10^{{{}}}$; '
                for m in np.arange(self._disp.size):
                    (a, x) = manexp10(self._disp[m])
                    param_str_00 += template.format(m + 2, a, x, prec = 3)
                ax.text(0, 0, param_str_00[0:-2], fontdict=font)
        else:
            param_str_04 = 'r_1 = {:.{prec}}, r_2 = {:.{prec}}'.format(self._r_1, self._r_2, prec = 3)
            ax.text(0, 4, param_str_04, fontdict=font)

            param_str_03 = 'tau_pho/tau_grp = {:.{prec}}'.format(self._tau_pho, prec = 3)
            ax.text(0, 3, param_str_03, fontdict=font)
            
            param_str_02 = 'tau_prp/tau_grp = {:.{prec}}'.format(self._tau_prp, prec = 3)
            ax.text(0, 2, param_str_02, fontdict=font)

            param_str_01 = '[tau_par/tau_grp, g_0, z_list] = '
            for region in np.arange(self._g_0.size):
                if self._z_list[region].size == 0:
                    template = '[{:.{prec}}, {:.{prec}}, [ ]], '
                    param_str_01 += template.format(self._tau_par[region], self._g_0[region], prec = 3)
                else:
                    template = '[{:.{prec}}, {:.{prec}}, [{:.{prec}}, {:.{prec}}]], '
                    param_str_01 += template.format(self._tau_par[region], self._g_0[region], self._z_list[region][0], self._z_list[region][1], prec = 3)
            ax.text(0, 1, param_str_01[0:-2], fontdict=font)

            if self._disp:
                param_str_00 = ''
                template = 'D_{} = {:.{prec}}; '
                for m in np.arange(self._disp.size):
                    param_str_00 += template.format(m + 2, self._disp[m], prec = 3)
                ax.text(0, 0, param_str_00[0:-2], fontdict=font)


class QSwitchedLaserModel(object):
    def __init__(self, config:Config, params:LaserParameters):
        self._config = config
        self._params = params

        self._regions = np.arange(self._params._g_0.size)

        self._q     = np.arange(-self._params._q_max, self._params._q_max + 1)
        self._q_minus_p = np.arange(-2 * self._params._q_max, 2 * self._params._q_max + 1)
        if self._params._q_max < 20:
            self._q_ticks = self._q[0::2]
        else:
            self._q_ticks = self._q[0::4]

        if self._params._z_list.size != 0:
            z = self._params._z_list[0]
            self._z_lg, self._w_lg = gauss_legendre(self._params._npts, a = z[0], b = z[1])
        else:
            self._z_lg, self._w_lg = gauss_legendre(self._params._npts, a = 0.0, b = 0.5)

        qmp_2d, z_2d = np.meshgrid(self._q_minus_p, self._z_lg, sparse=False, indexing='ij')
        self._cos_lg = np.cos(2 * qmp_2d * np.pi * z_2d)

        q_2d, p_2d = np.meshgrid(self._q, self._q, sparse=False, indexing='ij')
        self._index_qp = q_2d - p_2d + 2*self._params._q_max

        j_0 = 1j * self._params._inj_amplitude * np.sqrt(1.0 - self._params._r_1) / (1.0 - np.sqrt(self._params._r) * np.exp(1j * self._params._shift))
        q_2d, z_2d = np.meshgrid(self._q, self._z_lg, sparse=False, indexing='ij')
        beta = 1j * (2 * q_2d * np.pi - self._params._shift) + np.log(1.0/np.sqrt(self._params._r))
#        self._j_lg = np.sqrt(1.0/self._params._r) * np.exp(-beta * self._z_lg) + np.exp(1j * self._params._shift) * np.exp(beta * self._z_lg)
        self._j_lg = j_0 * ( np.sqrt(1.0/self._params._r) * np.exp(-beta * z_2d) + np.exp(1j * self._params._shift) * np.exp(beta * z_2d) )

        z_2d, q_2d = np.meshgrid(self._z_lg, self._q, sparse=False, indexing='ij')
        self._sat_p = np.sqrt(self._params._r_1) * np.exp(np.log(1.0/np.sqrt(self._params._r))*z_2d) * np.exp(2j * q_2d * np.pi * z_2d)
        self._sat_m = np.exp(-np.log(1.0/np.sqrt(self._params._r))*z_2d) * np.exp(-2j * q_2d * np.pi * z_2d)

        self._omega = self._delta_omega(self._q) * self._params._tau_prp

        self._q_3d, self._m_3d, self._n_3d = np.meshgrid(self._q, self._q, self._q, sparse=False, indexing='ij')
        self._index_qmn = self._indx(self._q_3d, self._m_3d, self._n_3d) + 3*self._params._q_max
        
        self._i2d, self._q2d, self._p2d = np.meshgrid(self._regions, self._q, self._q, sparse=False, indexing='ij')
#        self._index_qp = self._q2d - self._p2d + 2*self._params._q_max
        self._b = self._b_q(self._q)

        self._kmc_iqmn()

        self._set_scale()
        self.reset()
    
    def __str__(self):
        return self._params.__str__()

    def _d_omega(self, q):
        '''
        Return total frequency shift of mode q in units of the group round-trip time self._tau_grp.
        
        Parameters
        ----------
        q : numpy.int32
            Mode number. Can be an array.
        
        Returns
        ----------
        retval : numpy.float64
            Total frequency shift of mode q in units of the group round-trip time self._tau_grp.
        '''
                  
        disp_sum = np.zeros(q.shape)
        if self._params._disp.size != 0:
            for m in range(self._params._disp.size):
                disp_sum += self._params._disp[m] * (2 * q * np.pi) ** (m + 2) / factorial(m + 2)
    
        d_omega = -( q * np.pi * self._params._tau_prp/self._params._tau_pho
                         + disp_sum ) / (1.0 + 0.5 * self._params._tau_prp/self._params._tau_pho)
        return d_omega

    def _delta_omega(self, q):
        return 2 * np.pi * q + self._d_omega(q)

    def _indx(self, q, m, n):
        return q - m + n

    def _net_gain(self, q):
        omega = self._delta_omega(q) * self._params._tau_prp
        net_gain = sum(self._params._g_0)/(1 + omega**2) - 1.0/self._params._tau_pho
        return net_gain

    def _fwm(self):
        tau, m, n = np.meshgrid(self._params._tau_par, self._q, self._q, sparse=False, indexing='ij')
        omega_m = 2 * m * np.pi * self._params._tau_prp
        omega_n = 2 * n * np.pi * self._params._tau_prp
    
        b = 0.5 / ( 1 - 1j*omega_m ) + 0.5 / ( 1 + 1j*omega_n )
        c = 1.0 / ( 1 - 2j * (m - n) * np.pi * tau )
        
        g_0 = (self._params._g_0).reshape(self._regions.size, 1, 1)
        g = np.sum(self._params._g_0)
        
        fwm = np.sum(g_0 * (b * c), (0))/g
        
        return fwm

    def _b_q(self, q):
        omega = self._delta_omega(q) * self._params._tau_prp
        return 1.0 / ( 1 - 1j * omega )

    def _c_q(self, q, tau_par):
        return 1.0 / ( 1 - 2j * np.pi * q * tau_par )

    def _kmc_iqmn(self):
        
        def kappa_qmn(q, m, n):
            def delta(q):
                return 1.0/( 1.0 - 2j * q * np.pi / np.log(self._params._r) )
            
            def delta_kappa(q):
                r_1 = self._params._r_1
                r_2 = self._params._r_2
                beta = (r_1 + r_2) * (1 + np.sqrt(r_1*r_2)) / ((np.sqrt(r_1) + np.sqrt(r_2)) * np.sqrt(r_1*r_2))
                norm = 0.5 * np.sqrt(r_1*r_2) / ( (np.sqrt(r_1) + np.sqrt(r_2)) * (1 - np.sqrt(r_1*r_2)) )
        
                phi_0 = np.fmax( 1.0e-06 * np.ones(q.shape), norm * self._net_gain(q) )
                sqrt_phi = (3.0/(2.0*beta)) * (np.sqrt(1.0 + 8.0*beta*phi_0/9.0) - 1.0)
                dkappa = sqrt_phi / (phi_0 - 0.5*sqrt_phi)
                
                return dkappa
        
            def deltap(q, z):
                norm = ( self._params._c2 / np.log(1.0/(self._params._r)) ) * delta(q)
                arg =  ( 2j * q * np.pi - np.log(self._params._r) ) * z
                return norm * ( (np.exp(arg) - 1.0) + (1.0 - np.exp(-arg)) / self._params._r_1 )
            
            def kappa0(q, m, n):
                if (self._config == Config.URL):
                    kappa = 1.0
                elif (self._config == Config.SWL):
                    kappa = 1.0 + delta(2*(m - n))
                elif (self._config == Config.SHB):
                    kappa = 1.0 + delta(2*(m - n)) + delta(2*(q - m)) * delta_kappa(q)
                return kappa

            def kappap(q, m, n, z):
                if (self._config == Config.URL):
                    kappa = deltap(0, z)
                elif (self._config == Config.SWL):
                    kappa = deltap(0, z) + deltap(2*(m - n), z)
                elif (self._config == Config.SHB):
                    kappa = deltap(0, z) + deltap(2*(m - n), z) + deltap(2*(q - m), z) * delta_kappa(q)
                return kappa
            
            for region in self._regions:
                if self._params._z_list[region].size != 0:
                    z = self._params._z_list[region]
                    kappa = kappap(q, m, n, z[1]) - kappap(q, m, n, z[0])
                else:
                    kappa = kappa0(q, m, n)
                    
            return kappa

        b = 0.5 * self._b_q(self._q_3d)
        kappa = kappa_qmn(self._q_3d, self._m_3d, self._n_3d)
        kmc = np.zeros((self._regions.size, self._q.size, self._q.size, self._q.size ), dtype = np.complex128)
        for region in self._regions:
            c = self._c_q(self._m_3d - self._n_3d, self._params._tau_par[region])
            kmc[region] = b * kappa * c
            
        self._kmc = kmc

    def _set_scale(self):
        disp_sum = np.zeros(self._q.shape)
        if self._params._disp.size != 0:
            for m in range(self._params._disp.size):
                disp_sum += self._params._disp[m] * (2 * self._q * np.pi) ** (m + 1) / factorial(m + 1)
        self._scale = 1.0 / (1.0 + disp_sum)
    
    def _init_field(self, init_field_norm):
        cq0 = init_field_norm * np.sqrt( np.fmax(0.0, self._net_gain(self._q)) )
        phiq0 = 2 * np.pi * np.random.random_sample(self._q.shape)
        
        return cq0, phiq0

    def _e_qmn(self, e):
        ez = np.zeros(3*e.size - 2, dtype=e.dtype)
        ez[e.size - 1 : 2*e.size - 1] = e
        return ez[self._index_qmn]

    def _f(self, e, g, t):
        x = np.exp(2j * np.pi * t)
        g_qmp = np.dot(self._cos_lg, self._w_lg * g) * x**self._q_minus_p
        ge_q = np.dot(g_qmp[self._index_qp], e)

        gj_q = np.exp(-1j * self._params._shift * t) * np.dot(self._j_lg, self._w_lg * g) * x**self._q
        
        return ge_q + gj_q

    def _sat(self, e, t):
        x = np.exp(-2j * np.pi * t)
        e *= x**self._q
        e_p = np.dot(self._sat_p, e)
        e_m = np.dot(self._sat_m, e)
        
        return np.abs(e_p)**2 + np.abs(e_m)**2

    def _deriv(self, y, t):
        v = np.hsplit( y, (self._q.size, 2 * self._q.size, 2 * self._q.size + self._z_lg.size) )
        e = v[0] + 1j*v[1]
        g = v[2]

        f = self._f(e, g, t)
        dedt = ( -0.5/self._params._tau_pho ) * e + f
        
        sat = self._sat(e, t)
        dgdt = self._params._g_0[0] * self._p(t) - g * sat
        
        self._calls += 1
        
        return np.hstack((dedt.real, dedt.imag, dgdt))

    def _deriv_p(self, y, t):
        c, phi = np.hsplit(y, 2)
        e = c * np.exp(1j * phi)

        f = np.exp(-1j * phi) * self._f(e, t)
        
        dcdt = self._scale * (-c + f.real)
        dpdt = self._scale * (-self._omega + f.imag / np.fmax(1.0e-9 * np.ones(c.shape), c))
        self._calls += 1
    
        return np.hstack((dcdt, dpdt))

    def _e_out(self, eq_f, dpqdt_f, t_rt):
        t_mg, q_mg = np.meshgrid(t_rt, self._q, sparse=False, indexing='ij')
        
        dw = self._delta_omega(q_mg) - dpqdt_f
        phasor = np.exp(-1j*dw*t_mg)
        norm = np.sqrt( (1 - self._params._r_1) * np.sqrt(self._params._r_2) * np.log(1.0/self._params._r) 
                        / ( (np.sqrt(self._params._r_1) + np.sqrt(self._params._r_2)) * (1.0 - np.sqrt(self._params._r)) ) )
        eout = norm * np.dot(phasor, eq_f)
        
        self._eout = eout
    
        return eout

    def _psd(self, ef):
        freq = self._q - self._q[0]
        ec = np.conj(ef)

        a = np.zeros_like(ef)
        a[0] = np.dot(ef, ec)
        for f in freq[1:]:
            a[f] = np.dot(ef[f:], ec[:(-f)])
            
        psd = 10*np.log10(np.abs(a[:]/a[0]))
        
        return freq, psd

    def _uq(self, npts):
        #zn = np.linspace(0.0, 1.0, npts, endpoint = True)
        
        #q, z = np.meshgrid(self._q, zn, sparse=False, indexing='ij')
        
        #c = np.sqrt(self._params._c2)
        #cz = c #* ( 1.0 - (1 + np.sqrt(self._params._r_2)) * (z > 0.5) )

        #uq = cz * np.exp((2j*q*np.pi - 0.5*np.log(self._params._r)) * z)
        zn = np.linspace(0.0, 1.0, npts, endpoint = True)
    
        z, q = np.meshgrid(zn, self._q, sparse=False, indexing='ij')
    
        c = np.sqrt(self._params._c2)
        cz = c #* ( 1.0 - (1 + np.sqrt(r_2)) * (z > 0.5) )
    
        uq = cz * np.exp((2j*q*np.pi - 0.5*np.log(self._params._r)) * z)

        return zn, uq        

    def _get_iz_tm(self, indx, npts):
        zn, uq = self._uq(npts)
        eq_t = self._eq[indx] * np.exp(-2j * self._q * np.pi * self._t[indx])
        ez_t = np.dot(uq, eq_t)
        iz_t = np.abs(ez_t)**2
        
        return zn, uq, iz_t

    def _guardband(self, e, gb):
        if gb > 0:
            e[:,:gb] = 0.0 + 1j * 0.0
            e[:,-gb:] = 0.0 + 1j * 0.0
        
        return e

    def _savepath(self, dirpath, job_index, format_string):
        template = '{}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'
        lt = time.localtime()
        model_name = self.__class__.__name__
        if (self._config == Config.URL):
            model_name += '_URL_'
        elif (self._config == Config.SWL):
            model_name += '_SWL_'
        elif (self._config == Config.SHB):
            model_name += '_SHB_'
        name_str = dirpath + model_name + template.format(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec)

        if job_index:
            savepath = name_str + '_{:04d}.{}'.format(job_index, format_string)
        else:
            savepath = name_str + '.{}'.format(format_string)

        return savepath

    def _report(self, t, cq, phiq, eq_f, elapsed, usetex, show, dirpath, job_index, format_string):
        #dcqdt_t, dpqdt_t = np.hsplit(self._deriv_p(np.hstack((cq[-1], phiq[-1])), t[-1]), 2)

        if show:
            print("Elapsed time: {:.{prec}} sec\nDeriv function calls: {} ({:.{prec}} calls/sec)"
                  .format(elapsed, self._calls, self._calls/elapsed, prec = 3))
    
        #self._simplot(t, cq, phiq, eq_f, dpqdt_t, usetex, show, dirpath, job_index, format_string)
        self._simplot(t, cq, phiq, eq_f, 0.0, usetex, show, dirpath, job_index, format_string)

    def _simplot(self, t, cq, phiq, eq_f, dpqdt_f, usetex, show, dirpath, job_index, format_string):
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
        
        q = self._q
        q_ticks = self._q_ticks
        t_f = t[-1]
 
        dwq = self._d_omega(self._q)
        fpq = -( q * np.pi * self._params._tau_prp/self._params._tau_pho ) / (1.0 + .5 * self._params._tau_prp/self._params._tau_pho)
    
        npts = 1024
        t_2d, q_2d = np.meshgrid(t, self._q, sparse=False, indexing='ij')
        e_out = np.sum(self._eq * np.exp(-2j * np.pi * q_2d * t_2d), axis=1)
        j_out = self._params._inj_amplitude * np.exp(-1j * self._params._shift * t)
#        phi_out = savgol_filter(np.gradient(np.unwrap(np.arctan2((e_out+j_out).imag, (e_out+j_out).real)), t[1] - t[0]), 51, 3)
        phi_out = -savgol_filter(np.gradient(np.unwrap(np.arctan2((e_out).imag, (e_out).real)), t[1] - t[0]), 51, 3)
        i_out = np.abs(e_out)**2
        indx_mx = np.argmax(i_out)
        print('t_max = {} at index = {}\n'.format(t[indx_mx], indx_mx))
        i_out_f = np.abs(np.fft.rfft(np.abs(e_out)**2))
        freq = np.fft.rfftfreq(e_out.size)
        zn, uq, iz_tm = self._get_iz_tm(indx_mx, npts)
        
        fig, ax = plt.subplots(4, 2, figsize=(16, 24))
        fig.subplots_adjust(hspace=0.25, wspace = 0.25)
        for axis_row in ax:
            for axis in axis_row:
                axis.tick_params(axis='both', labelsize=labelsize)
                axis.grid(True)
        ax[0,0].set_xlabel(label_str('t', usetex), fontdict=font)
        ax[0,0].set_ylabel(label_str('|E_q(t)|', usetex), fontdict=font)
        ax[0,0].set_xlim(t[0], t[-1])
        ax[0,0].plot(t, cq)
        #ax[0,1].set_xlabel(label_str('t', usetex), fontdict=font)
        #ax[0,1].set_ylabel(label_str('\phi_q(t)', usetex), fontdict=font)
        #ax[0,1].set_xlim(t[0], t[-1])
        #ax[0,1].plot(t, phiq)
        ax[0,1].set_xlabel(label_str('t', usetex), fontdict=font)
        ax[0,1].set_ylabel(label_str('g(z,t)', usetex), fontdict=font)
        ax[0,1].set_xlim(t[0], t[-1])
        ax[0,1].plot(t, self._gz)
        ax[1,0].set_xlabel(label_str('t', usetex), fontdict=font)
        ax[1,0].set_ylabel(label_str('I_1(t)', usetex), fontdict=font)
        ax[1,0].set_xlim(t[0], t[-1])
        ax[1,0].plot(t, np.abs(e_out)**2)
        ax[1,1].set_xlabel(label_str('t', usetex), fontdict=font)
        ax[1,1].set_ylabel(label_str('\omega_1(t)', usetex), fontdict=font)
        ax[1,1].set_xlim(t[0], t[-1])
        ax[1,1].set_ylim(-1.0, 1.0)
#        ax[1,1].plot(t[1:], np.unwrap(np.angle(j_out[1:] + e_out[1:]))/t[1:])
        #ax[1,1].plot(t, np.gradient(np.unwrap(np.angle(j_out + e_out)), t[1] - t[0]))
#        ax[1,1].plot(t, np.gradient(np.unwrap(np.arctan2(e_out.imag, e_out.real)), t[1] - t[0]))
        ax[1,1].plot(t, phi_out)
        #ax[1,1].set_xlabel(label_str('f', usetex), fontdict=font)
        #ax[1,1].set_ylabel(label_str('I_1(f)', usetex), fontdict=font)
        #ax[1,1].plot(freq, i_out_f)
        ax[2,0].set_xlabel(label_str('q', usetex), fontdict=font)
        ax[2,0].set_ylabel(label_str('|E_q(t_m)|', usetex), fontdict=font)
        ax[2,0].set_xlim(q[0], q[-1])
        ax[2,0].set_xticks(q_ticks)
        markerline, stemlines, baseline = ax[2,0].stem(q, cq[indx_mx], '-')
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.setp(baseline, 'color','r', 'linewidth', 2)
        ax[2,1].set_xlabel(label_str('z', usetex), fontdict=font)
        ax[2,1].set_ylabel(label_str('I(z, t_m)', usetex), fontdict=font)
        ax[2,1].set_xlim(zn[0], zn[-1])
        ax[2,1].set_ylim(0.0, y_max(iz_tm))
        ax[2,1].plot(zn, iz_tm)
        
        ax[3,0].axis('off')
        ax[3,1].axis('off')
        self._params._fig(ax[3,0], usetex, font)
    
        if dirpath:
            savepath = self._savepath(dirpath, job_index, format_string)
            fig.savefig(savepath, bbox_inches='tight')
            print("Saved {0}".format(savepath))
        if show:
            plt.show()
        else:
            plt.close()

    def reset(self):
        self._calls = 0

    def get_stats(self):
        i_out = np.abs(self._eout) ** 2
        i_m = i_out.mean()
        i_s = i_out.std()
        
        return i_m, i_s

    def integrate(self, t_max, t_steps, p, j = None, init_field = [0.0], guard_band = 0, usetex = True, show = True, dirpath = False, job_index = False, format_string = 'pdf'):
        self.reset()

        self._p = p
        
        if len(init_field) == 1:
            cq0, phiq0 = self._init_field(init_field[0])
            eq0 = cq0 * np.exp(1j * phiq0)
        else:
            assert len(init_field) == len(self._q), "Inconsistent initial field and mode count."
            eq0 = init_field
            
        gz0 = np.zeros(self._params._npts)
        #if j is None:
            #injected_field = InjectedFieldConst(self._b * eq0)
            #self._j = injected_field.j
        #else:
##            assert j(0).shape == self._q.shape, "Inconsistent injected field and mode count."
            #self._j = j
            
        y0 = np.hstack((eq0.real, eq0.imag, gz0))
        t = np.linspace(0, t_max, t_steps)
    
        start = time.perf_counter()
        y = odeint(self._deriv, y0, t)
        finish = time.perf_counter()
        elapsed = finish - start
    
        v = np.hsplit( y, (self._q.size, 2 * self._q.size, 2 * self._q.size + self._z_lg.size) )
        eq = v[0] + 1j*v[1]
        gz = v[2]
        eq = self._guardband(eq, guard_band)

        self._t = t
        self._eq = eq
        self._gz = gz
        
        cq = np.abs(eq)
        phiq = np.angle(eq)
        eq_f = eq[-1] * np.exp(-1j * self._delta_omega(self._q) * t[-1])
        self._eq_f = eq_f

        self._report(t, cq, phiq, eq_f, elapsed, usetex, show, dirpath, job_index, format_string)

    def modplot(self):
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
        q_ticks = self._q_ticks
        
        fwm = self._fwm()
        fwm_r = fwm.real
        fwm_i = fwm.imag
        q_max = np.max(q)
        
        fig, ax = plt.subplots(2, 2, figsize=(16, 12))
        for axis_row in ax:
            for axis in axis_row:
                axis.tick_params(axis='both', labelsize=labelsize)
                axis.set_xticks(q_ticks)
                axis.grid(True) 

        ax[0,0].set_xlabel(r'$q$', fontdict=font)
        ax[0,0].set_ylabel('net unsaturated gain', fontdict=font)
        ax[0,0].set_xlim(q[0], q[-1])
        y = self._net_gain(q)
        ax[0,0].set_xlim(q[0], q[-1])
        ax[0,0].set_ylim(y_min(y), y_max(y))
        ax[0,0].plot(q, y)

        ax[0,1].set_xlabel(r'$q$', fontdict=font)
        ax[0,1].set_ylabel(r'$BC$', fontdict=font)
        ax[0,1].set_xlim(q[0], q[-1])
        ax[0,1].set_ylim(y_min(np.hstack((fwm_r, fwm_i))), y_max(np.hstack((fwm_r, fwm_i))))
        ax[0,1].plot(q, np.diag(np.fliplr(fwm.real)), '-', label = r'$\textrm{Re}(BC)$')
        ax[0,1].plot(q, np.diag(np.fliplr(fwm.imag)), linestyle = 'dashed', label = r'$\textrm{Im}(BC)$')
        ax[0,1].legend(fontsize=labelsize)
        
        ax[1,0].set_title(r'$\textrm{Re}(BC)$', fontdict=font)
        ax[1,0].set_yticks(q_ticks)
        cax = ax[1,0].imshow(fwm_r, interpolation='bilinear', cmap=cm.Spectral,
                    origin='lower', extent=[-q_max, q_max, -q_max, q_max],
                    vmax=fwm_r.max(), vmin=fwm_r.min())
        cbar = fig.colorbar(cax, ax=ax[1,0])
        cbar.ax.tick_params(axis='y', labelsize=labelsize)
    
        ax[1,1].set_title(r'$\textrm{Im}(BC)$', fontdict=font)
        ax[1,1].set_yticks(q_ticks)
        cax = ax[1,1].imshow(fwm_i, interpolation='bilinear', cmap=cm.Spectral,
                    origin='lower', extent=[-q_max, q_max, -q_max, q_max],
                    vmax=fwm_i.max(), vmin=fwm_i.min())
        cbar = fig.colorbar(cax, ax=ax[1,1])
        cbar.ax.tick_params(axis='y', labelsize=labelsize)

        plt.tight_layout(pad=2.0)
        plt.show()

class QSwitchedLaserModel0(object):
    def __init__(self, config:Config, params:LaserParameters):
        self._config = config
        self._params = params

        self._regions = np.arange(self._params._g_0.size)

        self._q     = np.arange(-self._params._q_max, self._params._q_max + 1)
        self._q_minus_p = np.arange(-2 * self._params._q_max, 2 * self._params._q_max + 1)
        if self._params._q_max < 20:
            self._q_ticks = self._q[0::2]
        else:
            self._q_ticks = self._q[0::4]

        if self._params._z_list.size != 0:
            z = self._params._z_list[0]
            self._z_lg, self._w_lg = gauss_legendre(self._params._npts, a = z[0], b = z[1])
        else:
            self._z_lg, self._w_lg = gauss_legendre(self._params._npts, a = 0.0, b = 0.5)

        qmp_2d, z_2d = np.meshgrid(self._q_minus_p, self._z_lg, sparse=False, indexing='ij')
        self._cos_lg = np.cos(2 * qmp_2d * np.pi * z_2d)

        q_2d, p_2d = np.meshgrid(self._q, self._q, sparse=False, indexing='ij')
        self._index_qp = q_2d - p_2d + 2*self._params._q_max

        j_0 = 1j * self._params._inj_amplitude * np.sqrt(1.0 - self._params._r_1) / (1.0 - np.sqrt(self._params._r) * np.exp(1j * self._params._shift))
        q_2d, z_2d = np.meshgrid(self._q, self._z_lg, sparse=False, indexing='ij')
        beta = 1j * (2 * q_2d * np.pi - self._params._shift) + np.log(1.0/np.sqrt(self._params._r))
#        self._j_lg = np.sqrt(1.0/self._params._r) * np.exp(-beta * self._z_lg) + np.exp(1j * self._params._shift) * np.exp(beta * self._z_lg)
        self._j_lg = j_0 * ( np.sqrt(1.0/self._params._r) * np.exp(-beta * z_2d) + np.exp(1j * self._params._shift) * np.exp(beta * z_2d) )

        z_2d, q_2d = np.meshgrid(self._z_lg, self._q, sparse=False, indexing='ij')
        self._sat_p = np.sqrt(self._params._r_1) * np.exp(np.log(1.0/np.sqrt(self._params._r))*z_2d) * np.exp(2j * q_2d * np.pi * z_2d)
        self._sat_m = np.exp(-np.log(1.0/np.sqrt(self._params._r))*z_2d) * np.exp(-2j * q_2d * np.pi * z_2d)

        self._omega = self._delta_omega(self._q) * self._params._tau_prp

        self._q_3d, self._m_3d, self._n_3d = np.meshgrid(self._q, self._q, self._q, sparse=False, indexing='ij')
        self._index_qmn = self._indx(self._q_3d, self._m_3d, self._n_3d) + 3*self._params._q_max
        
        self._i2d, self._q2d, self._p2d = np.meshgrid(self._regions, self._q, self._q, sparse=False, indexing='ij')
#        self._index_qp = self._q2d - self._p2d + 2*self._params._q_max
        self._b = self._b_q(self._q)

        self._kmc_iqmn()

        self._set_scale()
        self.reset()
    
    def __str__(self):
        return self._params.__str__()

    def _d_omega(self, q):
        '''
        Return total frequency shift of mode q in units of the group round-trip time self._tau_grp.
        
        Parameters
        ----------
        q : numpy.int32
            Mode number. Can be an array.
        
        Returns
        ----------
        retval : numpy.float64
            Total frequency shift of mode q in units of the group round-trip time self._tau_grp.
        '''
                  
        disp_sum = np.zeros(q.shape)
        if self._params._disp.size != 0:
            for m in range(self._params._disp.size):
                disp_sum += self._params._disp[m] * (2 * q * np.pi) ** (m + 2) / factorial(m + 2)
    
        d_omega = -( q * np.pi * self._params._tau_prp/self._params._tau_pho
                         + disp_sum ) / (1.0 + 0.5 * self._params._tau_prp/self._params._tau_pho)
        return d_omega

    def _delta_omega(self, q):
        return 2 * np.pi * q + self._d_omega(q)

    def _indx(self, q, m, n):
        return q - m + n

    def _net_gain(self, q):
        omega = self._delta_omega(q) * self._params._tau_prp
        net_gain = sum(self._params._g_0)/(1 + omega**2) - 1.0/self._params._tau_pho
        return net_gain

    def _fwm(self):
        tau, m, n = np.meshgrid(self._params._tau_par, self._q, self._q, sparse=False, indexing='ij')
        omega_m = 2 * m * np.pi * self._params._tau_prp
        omega_n = 2 * n * np.pi * self._params._tau_prp
    
        b = 0.5 / ( 1 - 1j*omega_m ) + 0.5 / ( 1 + 1j*omega_n )
        c = 1.0 / ( 1 - 2j * (m - n) * np.pi * tau )
        
        g_0 = (self._params._g_0).reshape(self._regions.size, 1, 1)
        g = np.sum(self._params._g_0)
        
        fwm = np.sum(g_0 * (b * c), (0))/g
        
        return fwm

    def _b_q(self, q):
        omega = self._delta_omega(q) * self._params._tau_prp
        return 1.0 / ( 1 - 1j * omega )

    def _c_q(self, q, tau_par):
        return 1.0 / ( 1 - 2j * np.pi * q * tau_par )

    def _kmc_iqmn(self):
        
        def kappa_qmn(q, m, n):
            def delta(q):
                return 1.0/( 1.0 - 2j * q * np.pi / np.log(self._params._r) )
            
            def delta_kappa(q):
                r_1 = self._params._r_1
                r_2 = self._params._r_2
                beta = (r_1 + r_2) * (1 + np.sqrt(r_1*r_2)) / ((np.sqrt(r_1) + np.sqrt(r_2)) * np.sqrt(r_1*r_2))
                norm = 0.5 * np.sqrt(r_1*r_2) / ( (np.sqrt(r_1) + np.sqrt(r_2)) * (1 - np.sqrt(r_1*r_2)) )
        
                phi_0 = np.fmax( 1.0e-06 * np.ones(q.shape), norm * self._net_gain(q) )
                sqrt_phi = (3.0/(2.0*beta)) * (np.sqrt(1.0 + 8.0*beta*phi_0/9.0) - 1.0)
                dkappa = sqrt_phi / (phi_0 - 0.5*sqrt_phi)
                
                return dkappa
        
            def deltap(q, z):
                norm = ( self._params._c2 / np.log(1.0/(self._params._r)) ) * delta(q)
                arg =  ( 2j * q * np.pi - np.log(self._params._r) ) * z
                return norm * ( (np.exp(arg) - 1.0) + (1.0 - np.exp(-arg)) / self._params._r_1 )
            
            def kappa0(q, m, n):
                if (self._config == Config.URL):
                    kappa = 1.0
                elif (self._config == Config.SWL):
                    kappa = 1.0 + delta(2*(m - n))
                elif (self._config == Config.SHB):
                    kappa = 1.0 + delta(2*(m - n)) + delta(2*(q - m)) * delta_kappa(q)
                return kappa

            def kappap(q, m, n, z):
                if (self._config == Config.URL):
                    kappa = deltap(0, z)
                elif (self._config == Config.SWL):
                    kappa = deltap(0, z) + deltap(2*(m - n), z)
                elif (self._config == Config.SHB):
                    kappa = deltap(0, z) + deltap(2*(m - n), z) + deltap(2*(q - m), z) * delta_kappa(q)
                return kappa
            
            for region in self._regions:
                if self._params._z_list[region].size != 0:
                    z = self._params._z_list[region]
                    kappa = kappap(q, m, n, z[1]) - kappap(q, m, n, z[0])
                else:
                    kappa = kappa0(q, m, n)
                    
            return kappa

        b = 0.5 * self._b_q(self._q_3d)
        kappa = kappa_qmn(self._q_3d, self._m_3d, self._n_3d)
        kmc = np.zeros((self._regions.size, self._q.size, self._q.size, self._q.size ), dtype = np.complex128)
        for region in self._regions:
            c = self._c_q(self._m_3d - self._n_3d, self._params._tau_par[region])
            kmc[region] = b * kappa * c
            
        self._kmc = kmc

    def _set_scale(self):
        disp_sum = np.zeros(self._q.shape)
        if self._params._disp.size != 0:
            for m in range(self._params._disp.size):
                disp_sum += self._params._disp[m] * (2 * self._q * np.pi) ** (m + 1) / factorial(m + 1)
        self._scale = 1.0 / (1.0 + disp_sum)
    
    def _init_field(self, init_field_norm):
        cq0 = init_field_norm * np.sqrt( np.fmax(0.0, self._net_gain(self._q)) )
        phiq0 = 2 * np.pi * np.random.random_sample(self._q.shape)
        
        return cq0, phiq0

    def _e_qmn(self, e):
        ez = np.zeros(3*e.size - 2, dtype=e.dtype)
        ez[e.size - 1 : 2*e.size - 1] = e
        return ez[self._index_qmn]

    def _f(self, e, g, t):
        g_qmp = np.dot(self._cos_lg, self._w_lg * g)
        ge_q = np.dot(g_qmp[self._index_qp], e)

        gj_q = np.exp(-1j * self._params._shift * t) * np.dot(self._j_lg, self._w_lg * g)
        
        return ge_q + gj_q

    def _sat(self, e):
        e_p = np.dot(self._sat_p, e)
        e_m = np.dot(self._sat_m, e)
        
        return np.abs(e_p)**2 + np.abs(e_m)**2

    def _deriv(self, y, t):
        v = np.hsplit( y, (self._q.size, 2 * self._q.size, 2 * self._q.size + self._z_lg.size) )
        e = v[0] + 1j*v[1]
        g = v[2]

        f = self._f(e, g, t)
        dedt = ( -0.5/self._params._tau_pho - 2j * self._q * np.pi ) * e + f
        
        sat = self._sat(e)
        dgdt = self._params._g_0[0] * self._p(t) - g * sat
        
        self._calls += 1
        
        return np.hstack((dedt.real, dedt.imag, dgdt))

    def _deriv_p(self, y, t):
        c, phi = np.hsplit(y, 2)
        e = c * np.exp(1j * phi)

        f = np.exp(-1j * phi) * self._f(e, t)
        
        dcdt = self._scale * (-c + f.real)
        dpdt = self._scale * (-self._omega + f.imag / np.fmax(1.0e-9 * np.ones(c.shape), c))
        self._calls += 1
    
        return np.hstack((dcdt, dpdt))

    def _e_out(self, eq_f, dpqdt_f, t_rt):
        t_mg, q_mg = np.meshgrid(t_rt, self._q, sparse=False, indexing='ij')
        
        dw = self._delta_omega(q_mg) - dpqdt_f
        phasor = np.exp(-1j*dw*t_mg)
        norm = np.sqrt( (1 - self._params._r_1) * np.sqrt(self._params._r_2) * np.log(1.0/self._params._r) 
                        / ( (np.sqrt(self._params._r_1) + np.sqrt(self._params._r_2)) * (1.0 - np.sqrt(self._params._r)) ) )
        eout = norm * np.dot(phasor, eq_f)
        
        self._eout = eout
    
        return eout

    def _psd(self, ef):
        freq = self._q - self._q[0]
        ec = np.conj(ef)

        a = np.zeros_like(ef)
        a[0] = np.dot(ef, ec)
        for f in freq[1:]:
            a[f] = np.dot(ef[f:], ec[:(-f)])
            
        psd = 10*np.log10(np.abs(a[:]/a[0]))
        
        return freq, psd

    def _uq(self, npts):
        #zn = np.linspace(0.0, 1.0, npts, endpoint = True)
        
        #q, z = np.meshgrid(self._q, zn, sparse=False, indexing='ij')
        
        #c = np.sqrt(self._params._c2)
        #cz = c #* ( 1.0 - (1 + np.sqrt(self._params._r_2)) * (z > 0.5) )

        #uq = cz * np.exp((2j*q*np.pi - 0.5*np.log(self._params._r)) * z)
        zn = np.linspace(0.0, 1.0, npts, endpoint = True)
    
        z, q = np.meshgrid(zn, self._q, sparse=False, indexing='ij')
    
        c = np.sqrt(self._params._c2)
        cz = c #* ( 1.0 - (1 + np.sqrt(r_2)) * (z > 0.5) )
    
        uq = cz * np.exp((2j*q*np.pi - 0.5*np.log(self._params._r)) * z)

        return zn, uq        

    def _get_iz_tm(self, indx, npts):
        zn, uq = self._uq(npts)
        eq_t = self._eq[indx]
        ez_t = np.dot(uq, eq_t)
        iz_t = np.abs(ez_t)**2
        
        return zn, uq, iz_t

    def _guardband(self, e, gb):
        if gb > 0:
            e[:,:gb] = 0.0 + 1j * 0.0
            e[:,-gb:] = 0.0 + 1j * 0.0
        
        return e

    def _savepath(self, dirpath, job_index, format_string):
        template = '{}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'
        lt = time.localtime()
        model_name = self.__class__.__name__
        if (self._config == Config.URL):
            model_name += '_URL_'
        elif (self._config == Config.SWL):
            model_name += '_SWL_'
        elif (self._config == Config.SHB):
            model_name += '_SHB_'
        name_str = dirpath + model_name + template.format(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec)

        if job_index:
            savepath = name_str + '_{:04d}.{}'.format(job_index, format_string)
        else:
            savepath = name_str + '.{}'.format(format_string)

        return savepath

    def _report(self, t, cq, phiq, eq_f, elapsed, usetex, show, dirpath, job_index, format_string):
        #dcqdt_t, dpqdt_t = np.hsplit(self._deriv_p(np.hstack((cq[-1], phiq[-1])), t[-1]), 2)

        if show:
            print("Elapsed time: {:.{prec}} sec\nDeriv function calls: {} ({:.{prec}} calls/sec)"
                  .format(elapsed, self._calls, self._calls/elapsed, prec = 3))
    
        #self._simplot(t, cq, phiq, eq_f, dpqdt_t, usetex, show, dirpath, job_index, format_string)
        self._simplot(t, cq, phiq, eq_f, 0.0, usetex, show, dirpath, job_index, format_string)

    def _simplot(self, t, cq, phiq, eq_f, dpqdt_f, usetex, show, dirpath, job_index, format_string):
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
        
        q = self._q
        q_ticks = self._q_ticks
        t_f = t[-1]
 
        dwq = self._d_omega(self._q)
        fpq = -( q * np.pi * self._params._tau_prp/self._params._tau_pho ) / (1.0 + .5 * self._params._tau_prp/self._params._tau_pho)
    
        npts = 1024
        e_out = np.sum(self._eq, axis=1)
        j_out = self._params._inj_amplitude * np.exp(-1j * self._params._shift * t)
#        phi_out = savgol_filter(np.gradient(np.unwrap(np.arctan2((e_out+j_out).imag, (e_out+j_out).real)), t[1] - t[0]), 51, 3)
        phi_out = -savgol_filter(np.gradient(np.unwrap(np.arctan2((e_out).imag, (e_out).real)), t[1] - t[0]), 51, 3)
        i_out = np.abs(e_out)**2
        indx_mx = np.argmax(i_out)
        i_out_f = np.abs(np.fft.rfft(np.abs(e_out)**2))
        freq = np.fft.rfftfreq(e_out.size)
        zn, uq, iz_tm = self._get_iz_tm(indx_mx, npts)
        
        fig, ax = plt.subplots(4, 2, figsize=(16, 24))
        fig.subplots_adjust(hspace=0.25, wspace = 0.25)
        for axis_row in ax:
            for axis in axis_row:
                axis.tick_params(axis='both', labelsize=labelsize)
                axis.grid(True)
        ax[0,0].set_xlabel(label_str('t', usetex), fontdict=font)
        ax[0,0].set_ylabel(label_str('|E_q(t)|', usetex), fontdict=font)
        ax[0,0].set_xlim(t[0], t[-1])
        ax[0,0].plot(t, cq)
        #ax[0,1].set_xlabel(label_str('t', usetex), fontdict=font)
        #ax[0,1].set_ylabel(label_str('\phi_q(t)', usetex), fontdict=font)
        #ax[0,1].set_xlim(t[0], t[-1])
        #ax[0,1].plot(t, phiq)
        ax[0,1].set_xlabel(label_str('t', usetex), fontdict=font)
        ax[0,1].set_ylabel(label_str('g(z,t)', usetex), fontdict=font)
        ax[0,1].set_xlim(t[0], t[-1])
        ax[0,1].plot(t, self._gz)
        ax[1,0].set_xlabel(label_str('t', usetex), fontdict=font)
        ax[1,0].set_ylabel(label_str('I_1(t)', usetex), fontdict=font)
        ax[1,0].set_xlim(t[0], t[-1])
        ax[1,0].plot(t, np.abs(e_out)**2)
        ax[1,1].set_xlabel(label_str('t', usetex), fontdict=font)
        ax[1,1].set_ylabel(label_str('\omega_1(t)', usetex), fontdict=font)
        ax[1,1].set_xlim(t[0], t[-1])
        ax[1,1].set_ylim(-1.0, 1.0)
#        ax[1,1].plot(t[1:], np.unwrap(np.angle(j_out[1:] + e_out[1:]))/t[1:])
        #ax[1,1].plot(t, np.gradient(np.unwrap(np.angle(j_out + e_out)), t[1] - t[0]))
#        ax[1,1].plot(t, np.gradient(np.unwrap(np.arctan2(e_out.imag, e_out.real)), t[1] - t[0]))
        ax[1,1].plot(t, phi_out)
        #ax[1,1].set_xlabel(label_str('f', usetex), fontdict=font)
        #ax[1,1].set_ylabel(label_str('I_1(f)', usetex), fontdict=font)
        #ax[1,1].plot(freq, i_out_f)
        ax[2,0].set_xlabel(label_str('q', usetex), fontdict=font)
        ax[2,0].set_ylabel(label_str('|E_q(t_m)|', usetex), fontdict=font)
        ax[2,0].set_xlim(q[0], q[-1])
        ax[2,0].set_xticks(q_ticks)
        markerline, stemlines, baseline = ax[2,0].stem(q, cq[indx_mx], '-')
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.setp(baseline, 'color','r', 'linewidth', 2)
        ax[2,1].set_xlabel(label_str('z', usetex), fontdict=font)
        ax[2,1].set_ylabel(label_str('I(z, t_m)', usetex), fontdict=font)
        ax[2,1].set_xlim(zn[0], zn[-1])
        ax[2,1].set_ylim(0.0, y_max(iz_tm))
        ax[2,1].plot(zn, iz_tm)
        
        ax[3,0].axis('off')
        ax[3,1].axis('off')
        self._params._fig(ax[3,0], usetex, font)
    
        if dirpath:
            savepath = self._savepath(dirpath, job_index, format_string)
            fig.savefig(savepath, bbox_inches='tight')
            print("Saved {0}".format(savepath))
        if show:
            plt.show()
        else:
            plt.close()

    def reset(self):
        self._calls = 0

    def get_stats(self):
        i_out = np.abs(self._eout) ** 2
        i_m = i_out.mean()
        i_s = i_out.std()
        
        return i_m, i_s

    def integrate(self, t_max, t_steps, p, j = None, init_field = [0.0], guard_band = 0, usetex = True, show = True, dirpath = False, job_index = False, format_string = 'pdf'):
        self.reset()

        self._p = p
        
        if len(init_field) == 1:
            cq0, phiq0 = self._init_field(init_field[0])
            eq0 = cq0 * np.exp(1j * phiq0)
        else:
            assert len(init_field) == len(self._q), "Inconsistent initial field and mode count."
            eq0 = init_field
            
        gz0 = np.zeros(self._params._npts)
        #if j is None:
            #injected_field = InjectedFieldConst(self._b * eq0)
            #self._j = injected_field.j
        #else:
##            assert j(0).shape == self._q.shape, "Inconsistent injected field and mode count."
            #self._j = j
            
        y0 = np.hstack((eq0.real, eq0.imag, gz0))
        t = np.linspace(0, t_max, t_steps)
    
        start = time.perf_counter()
        y = odeint(self._deriv, y0, t)
        finish = time.perf_counter()
        elapsed = finish - start
    
        v = np.hsplit( y, (self._q.size, 2 * self._q.size, 2 * self._q.size + self._z_lg.size) )
        eq = v[0] + 1j*v[1]
        gz = v[2]
        eq = self._guardband(eq, guard_band)

        self._t = t
        self._eq = eq
        self._gz = gz
        
        cq = np.abs(eq)
        phiq = np.angle(eq)
        eq_f = eq[-1] * np.exp(-1j * self._delta_omega(self._q) * t[-1])
        self._eq_f = eq_f

        self._report(t, cq, phiq, eq_f, elapsed, usetex, show, dirpath, job_index, format_string)

    def modplot(self):
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
        q_ticks = self._q_ticks
        
        fwm = self._fwm()
        fwm_r = fwm.real
        fwm_i = fwm.imag
        q_max = np.max(q)
        
        fig, ax = plt.subplots(2, 2, figsize=(16, 12))
        for axis_row in ax:
            for axis in axis_row:
                axis.tick_params(axis='both', labelsize=labelsize)
                axis.set_xticks(q_ticks)
                axis.grid(True) 

        ax[0,0].set_xlabel(r'$q$', fontdict=font)
        ax[0,0].set_ylabel('net unsaturated gain', fontdict=font)
        ax[0,0].set_xlim(q[0], q[-1])
        y = self._net_gain(q)
        ax[0,0].set_xlim(q[0], q[-1])
        ax[0,0].set_ylim(y_min(y), y_max(y))
        ax[0,0].plot(q, y)

        ax[0,1].set_xlabel(r'$q$', fontdict=font)
        ax[0,1].set_ylabel(r'$BC$', fontdict=font)
        ax[0,1].set_xlim(q[0], q[-1])
        ax[0,1].set_ylim(y_min(np.hstack((fwm_r, fwm_i))), y_max(np.hstack((fwm_r, fwm_i))))
        ax[0,1].plot(q, np.diag(np.fliplr(fwm.real)), '-', label = r'$\textrm{Re}(BC)$')
        ax[0,1].plot(q, np.diag(np.fliplr(fwm.imag)), linestyle = 'dashed', label = r'$\textrm{Im}(BC)$')
        ax[0,1].legend(fontsize=labelsize)
        
        ax[1,0].set_title(r'$\textrm{Re}(BC)$', fontdict=font)
        ax[1,0].set_yticks(q_ticks)
        cax = ax[1,0].imshow(fwm_r, interpolation='bilinear', cmap=cm.Spectral,
                    origin='lower', extent=[-q_max, q_max, -q_max, q_max],
                    vmax=fwm_r.max(), vmin=fwm_r.min())
        cbar = fig.colorbar(cax, ax=ax[1,0])
        cbar.ax.tick_params(axis='y', labelsize=labelsize)
    
        ax[1,1].set_title(r'$\textrm{Im}(BC)$', fontdict=font)
        ax[1,1].set_yticks(q_ticks)
        cax = ax[1,1].imshow(fwm_i, interpolation='bilinear', cmap=cm.Spectral,
                    origin='lower', extent=[-q_max, q_max, -q_max, q_max],
                    vmax=fwm_i.max(), vmin=fwm_i.min())
        cbar = fig.colorbar(cax, ax=ax[1,1])
        cbar.ax.tick_params(axis='y', labelsize=labelsize)

        plt.tight_layout(pad=2.0)
        plt.show()

class QSLModel(QSwitchedLaserModel):
    def __init__(self, config:Config, params:LaserParameters):
        QSwitchedLaserModel.__init__(self, config, params)
