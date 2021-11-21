import math
import numpy as np

import scipy.integrate
from scipy.special import cbrt
from scipy.special import gamma
from scipy.special import iv
from multipcc.utils import w_function, w_function_2


def integrand(x, a, b, c):  # integration I(a,b,c)
    return ((x ** a) * (np.sin(b * np.arctan(c * x)) ** 2)) / ((1 + (c * x) ** 2) ** b)


integrand_vec = np.vectorize(integrand)


def quad(func, a, b, c):
    return scipy.integrate.quad(func, 0, 1, (a, b, c))


quad_vec = np.vectorize(quad)


class Multiphonon:
    """
    Multiphonon capture class
    """

    def __init__(self, data) -> None:
        self.data = data

    def derived_parameters(self):
        phycon = self.data.physical_constants
        inputs = self.data.inputs
        derived = self.data.derived
        mpder = self.data.multiphonon_derived_parameters

        # radius of sphere with Brillouin zone volume in metre inverse
        mpder.q_D = cbrt(6 * np.pi ** 2) / inputs.a_0

        mpder.sa = 4 * math.sqrt(
            np.pi * derived.r_eh * phycon.eVJ / (phycon.kB * inputs.T)
        )  # sommerfiled factor
        mpder.pekar = (1 / inputs.epsilon_h) - (1 / inputs.epsilon_l)  # pekar factor
        mpder.V_0 = (inputs.a_0) ** 3  # volume of the unit cell in cubic meters
        mpder.omega = (inputs.Eph * phycon.eVJ) / phycon.hbar  # frequecy of the phonon

    def Huang_Rhys_Factor_deformation_potential_coupling(self):
        phycon = self.data.physical_constants
        inputs = self.data.inputs
        derived = self.data.derived
        mpder = self.data.multiphonon_derived_parameters
        egrid = self.data.energy_grids
        hrfd = self.data.deformation_potential_coupling

        hrfd.SHRD = (
            (inputs.Dij * phycon.eVJ * 100) / (inputs.Eph * phycon.eVJ)
        ) ** 2 / (
            2 * inputs.Mr * mpder.omega / phycon.hbar
        )  # deformation coupling
        print(hrfd.SHRD)
        # array of the three differnt values of mu depending on charge state. The value of mu for
        hrfd.mu = np.array([-egrid.nu, egrid.nu * 1e-6, egrid.nu])
        # neutral charge state was supposed to be zero but has been given a small value to avoid
        # division by zero error.
        hrfd.a = 0
        hrfd.b = 2 * hrfd.mu
        hrfd.c = (mpder.q_D * derived.a_ebr * egrid.nu) / 2
        hrfd.bcsqre = (hrfd.b * hrfd.c) ** 2
        # integral part of the Huang Rhys Factor
        hrfd.ans, hrfd.err = quad_vec(integrand_vec, hrfd.a, hrfd.b, hrfd.c)
        hrfd.I = hrfd.ans / hrfd.bcsqre
        # final values of Huang Rhys Factor. SHR is an array of size mu x Et. Each coloumn contains the
        hrfd.SHR_D = hrfd.SHRD * hrfd.I
        # values of SHR for every possible value of energy for a particula
        # r charge state

    def Huang_Rhys_Factor_polar_coupling(self):
        phycon = self.data.physical_constants
        inputs = self.data.inputs
        derived = self.data.derived
        mpder = self.data.multiphonon_derived_parameters
        egrid = self.data.energy_grids
        hrfp = self.data.polar_coupling

        hrfp.SHRP = (3 / (2 * ((inputs.Eph * phycon.eVJ) ** 2))) * (
            (phycon.Qe ** 2)
            * (inputs.Mr / mpder.V_0)
            * inputs.Eph
            * phycon.eVJ
            / (inputs.Mr * (mpder.q_D ** 2))
            * mpder.pekar
        )  # polar coupling
        print(hrfp.SHRP)
        # array of the three differnt values of mu depending on charge state. The value of mu for
        hrfp.mu = np.array([-egrid.nu, egrid.nu * 1e-6, egrid.nu])
        # neutral charge state was supposed to be zero but has been given a small value to avoid
        # division by zero error.
        hrfp.a = -2
        hrfp.b = 2 * hrfp.mu
        hrfp.c = (mpder.q_D * derived.a_ebr * egrid.nu) / 2
        hrfp.bcsqre = (hrfp.b * hrfp.c) ** 2
        # integral part of the Huang Rhys Factor
        hrfp.ans, hrfp.err = quad_vec(integrand_vec, hrfp.a, hrfp.b, hrfp.c)
        hrfp.I = hrfp.ans / hrfp.bcsqre
        # final values of Huang Rhys Factor. SHR is an array of size mu x Et. Each coloumn contains the
        hrfp.SHR_P = hrfp.SHRP * hrfp.I
        # values of SHR for every possible value of energy for a particula
        # r charge state

    def Huang_Rhys_factor(self):
        inputs = self.data.inputs
        egrid = self.data.energy_grids
        hrfd = self.data.deformation_potential_coupling
        hrfp = self.data.polar_coupling
        hrf = self.data.huang_rhys_factor

        # array of the three differnt values of mu depending on charge state. The value of mu for
        hrf.mu = np.array([-egrid.nu, egrid.nu * 1e-6, egrid.nu])
        # neutral charge state was supposed to be zero but has been given a small value to avoid
        # division by zero error.

        if inputs.dir == "dp":
            hrf.SHR = hrfd.SHR_D
        elif inputs.dir == "pc":
            hrf.SHR = hrfp.SHR_P
        elif inputs.dir == "com":
            hrf.SHR = hrfd.SHR_D + hrfp.SHR_P
        else:
            print("Please select Multiphonon coupling potential")

    def multiphonon_capture_coefficients(self):
        phycon = self.data.physical_constants
        inputs = self.data.inputs
        derived = self.data.derived
        mpder = self.data.multiphonon_derived_parameters
        egrid = self.data.energy_grids
        hrf = self.data.huang_rhys_factor
        mpcoef = self.data.multiphonon_capture_coefficients

        mpcoef.theta = (inputs.Eph * phycon.eVJ) / (2 * phycon.kB * inputs.T)
        # round to next highest integer
        mpcoef.p = np.ceil(egrid.ET / inputs.Eph)
        mpcoef.p_vec = np.ones(hrf.mu.shape) * mpcoef.p  # matching the shape of mu

        mpcoef.X = np.zeros(hrf.SHR.shape)
        mpcoef.X[hrf.SHR < mpcoef.p_vec] = hrf.SHR[hrf.SHR < mpcoef.p_vec] / (
            mpcoef.p_vec[hrf.SHR < mpcoef.p_vec] * math.sinh(mpcoef.theta)
        )
        mpcoef.X[hrf.SHR > mpcoef.p_vec] = mpcoef.p_vec[hrf.SHR > mpcoef.p_vec] / (
            hrf.SHR[hrf.SHR > mpcoef.p_vec] * math.sinh(mpcoef.theta)
        )
        mpcoef.sa = np.array([mpder.sa, 1, mpder.sa])

        mpcoef.Y = np.sqrt(1 + mpcoef.X ** 2)
        mpcoef.V_T = (
            (4 / 3) * np.pi * (derived.a_ebr * egrid.nu / 2) ** 3
        )  # volume of the wave function
        mpcoef.k1 = (
            (mpcoef.V_T)
            * ((mpcoef.p ** 2) * mpder.omega * math.sqrt(2 * np.pi))
            / (np.sqrt(mpcoef.p * mpcoef.Y))
        )
        mpcoef.k2 = (
            mpcoef.theta
            + mpcoef.Y
            - mpcoef.X * math.cosh(mpcoef.theta)
            - np.log((1 + mpcoef.Y) / mpcoef.X)
        )
        # recombination coefficients in m^3/s
        mpcoef.k = mpcoef.k1 * np.exp(mpcoef.p * mpcoef.k2)

        mpcoef.capt_cs = mpcoef.k / inputs.v_th  # capture cross section

    def trap_state_mp(self):
        inputs = self.data.inputs
        mpder = self.data.multiphonon_derived_parameters
        mpcoef = self.data.multiphonon_capture_coefficients
        tsm = self.data.trap_state_mp

        if inputs.trap_state == "don":
            # electron capture coefficient from the CB by donar [reversesed for the same reason as radiative]
            tsm.mp_sign = mpder.sa * mpcoef.k[2, :][::-1]
            # hole capture coefficent from VB by donar
            tsm.mp_sigp = mpcoef.k[1, :]

        else:
            # electron capture coefficient from CB by acceptor
            tsm.mp_sign = mpcoef.k[1, :][::-1]
            # hole capture coefficent from VB by acceptor
            tsm.mp_sigp = mpder.sa * mpcoef.k[0, :]


class Radiative:
    def __init__(self, data):
        self.data = data

    def photon_energy(self):
        """
        In photon energy grid all matrices are 2D. Along rows, ET remains fixed. along columns Ek remains fixed.
        Ept increases from left to right due to increase in Ek from left to right and increases from top to bottom due to increase of ET from top to bottom
        """
        mat = self.data.matrix
        egrid = self.data.energy_grids
        pgrid = self.data.photon_energy

        pgrid.Ept = (
            (np.zeros(mat.mat2D.shape) + egrid.Ek).T + egrid.ET
        ).T  # photon energy grid
        pgrid.theta = mat.Ek2D / mat.ET2D  # theta grid of the shape of Ept

    def charge_states(self):
        """
        In charge state grid, all matrices are 3D, formed by repeating the matrices of photon energy grid three times for three charge states. ET,Ek,deltaE and nu consist of three equal blocks. mu is -nu, 0 and nu for -ve, neutral and +ve charge state respectively.
        """
        mat = self.data.matrix
        pgrid = self.data.photon_energy
        cgrid = self.data.charge_states

        cgrid.mu = np.array([-mat.nu2D, mat.nu2D * 1e-6, mat.nu2D])  # mu is 3D matrix
        # Ept grid of the shape of mu
        cgrid.Ept = np.array([pgrid.Ept, pgrid.Ept, pgrid.Ept])
        # theta grid of the shape of mu
        cgrid.theta = np.array([pgrid.theta, pgrid.theta, pgrid.theta])

    def broadening_function(self):
        hrf = self.data.huang_rhys_factor
        phycon = self.data.physical_constants
        inputs = self.data.inputs
        bfunc = self.data.broadening_function

        bfunc.fB = 1 / (
            scipy.exp((inputs.Eph * phycon.eVJ) / (phycon.kB * inputs.T)) - 1
        )
        bfunc.bessel = iv(hrf.SHR, 2 * hrf.SHR * scipy.sqrt(bfunc.fB * (bfunc.fB + 1)))
        bfunc.broadening = (
            scipy.exp(-2 * hrf.SHR * (bfunc.fB + 1))
            * scipy.exp(hrf.SHR * inputs.Eph * phycon.eVJ / (phycon.kB * inputs.T))
            * bfunc.bessel
        )

    def photoionization_cross_section(self):
        """
        Photoionization cross section found using eq.5.91 from QPC by BKR. Unit is [m^2].
        For negative charege state the values will be invalid for nu >= 0.5. So nu is taken from 0.5.
        The remaining of the calculations are done keepting those values masked.
        The cross section is weighted to the unoccupied states to get rid of the final state energy dependence
        The weighted p_crosssection is multiplied by the thermal velocity to get the coefficient
        """
        phycon = self.data.physical_constants
        inputs = self.data.inputs
        derived = self.data.derived
        mat = self.data.matrix
        cgrid = self.data.charge_states
        pion = self.data.photoionization_cross_section
        # bfunc = self.data.broadening_function
        # wf = self.data.weighing_function

        # photoionization cross section (PCS)
        # constant part of PCS: PCS_C
        # Mu varrying part of PCS: PCS_vMU
        # E varrying part of PCS: PCS_vE

        pion.PCS_C = (
            (16 / 3)
            * phycon.alpha
            * phycon.a_br ** 2
            * phycon.r_h
            * phycon.eVJ
            * (1 / derived.eta_r)
            * (phycon.m_e / inputs.M_eff)
            * (2 * scipy.pi)
            * (np.sqrt(2 * inputs.M_eff) / phycon.hbar) ** 3
        )
        pion.Gamma = (gamma(cgrid.mu + 1)) ** 2 / gamma(2 * cgrid.mu + 1)
        pion.PCS_vMu = (
            (2 ** (2 * cgrid.mu)) * pion.Gamma * (mat.nu3D * derived.a_ebr) ** 3
        )
        pion.PCS_vE = (
            (np.sqrt(mat.Ek3D * phycon.eVJ)) ** 3 / (cgrid.Ept * phycon.eVJ)
        ) * (
            (np.sin((cgrid.mu + 1) * np.arctan(np.sqrt(cgrid.theta)))) ** 2
            / (cgrid.theta * (1 + cgrid.theta) ** (cgrid.mu + 1))
        )

        pion.PCS_E = pion.PCS_C * pion.PCS_vMu * pion.PCS_vE
        # photoionization cross section before weighing or summation overEk
        pion.PCS_E = np.ma.masked_less_equal(pion.PCS_E, 0)

        # weighed by density of unoccupied states per volume
        pion.PCS = w_function_2(self.data, pion.PCS_E, 2)
        pion.PCoeff = pion.PCS * inputs.v_th  # photoionization coefficient

        pion.sa = 4 * math.sqrt(
            scipy.pi * derived.r_eh * phycon.eVJ / (phycon.kB * inputs.T)
        )  # sommerfiled factor

        pion.sigma_k_c = (
            (16 / 3)
            * phycon.alpha
            * phycon.a_br ** 2
            * phycon.r_h
            * phycon.eVJ
            * (1 / derived.eta_r)
            * (phycon.m_e / inputs.M_eff)
            * (2 * scipy.pi)
            / (phycon.hbar * phycon.c / derived.eta_r) ** 3
        )
        pion.sigma_k_gamma = (gamma(cgrid.mu + 1)) ** 2 / gamma(2 * cgrid.mu + 1)
        pion.sigma_k_mu = (
            (2 ** (2 * cgrid.mu)) * pion.sigma_k_gamma * (mat.nu3D * derived.a_ebr) ** 3
        )
        pion.sigma_k_E = (
            mat.Ek3D
            * phycon.eVJ
            * (cgrid.Ept * phycon.eVJ) ** 2
            / (cgrid.Ept * phycon.eVJ)
        ) * (
            (np.sin((cgrid.mu + 1) * np.arctan(np.sqrt(cgrid.theta)))) ** 2
            / (cgrid.theta * (1 + cgrid.theta) ** (cgrid.mu + 1))
        )
        pion.sigma_k_Energy = pion.sigma_k_c * pion.sigma_k_mu * pion.sigma_k_E
        # weighed by density of unoccupied states per volume
        pion.Ccoeff = w_function(self.data, pion.sigma_k_Energy, 2) * inputs.v_th
        # pion.Ccoeff = pion.Ccoeff * bfunc.broadening

    def radiative_capture_cross_section(self):
        """
        Capture cross section is found using eq 5.30 of QPC by BKR. Unit is [m^2].
        The photon wave cector q in eq 5.30 is found using eq. 5.72 by replacing the numerator with our photon energy Ept.
        """
        phycon = self.data.physical_constants
        inputs = self.data.inputs
        derived = self.data.derived
        mat = self.data.matrix
        cgrid = self.data.charge_states
        pion = self.data.photoionization_cross_section
        # bfunc = self.data.broadening_function
        rcapt = self.data.radiative_capture_cross_section

        # capture cross section : CCS

        rcapt.factor = (cgrid.Ept * phycon.eVJ) ** 2 / (
            2 * inputs.M_eff * (phycon.c / derived.eta_r) ** 2 * mat.Ek3D * phycon.eVJ
        )  # DIMENSIONLESS
        rcapt.CCS_E = rcapt.factor * pion.PCS_E
        # capture cross section before weighing or summation overEk
        rcapt.CCS_E = np.ma.masked_array(rcapt.CCS_E, pion.PCS_E.mask)
        # weighed by density of unoccupied states per volume
        rcapt.CCS = w_function(self.data, rcapt.CCS_E, 2)
        rcapt.Ccoeff = rcapt.CCS * inputs.v_th  # capture coefficient
        rcapt.extrafactor = rcapt.Ccoeff / pion.Ccoeff

    def trap_state_rc(self):
        """
        Chosing the right capture coefficinets according to charge state.
        When the capture is from CB, the corresponding photo energy decreases as in the begining the trap is close to the VB. Thus the rates of capture
        from the CB are always reversed.
        when the state is charged, the coefficent of capture is enhanced by coulomb factor / sommerfeld factor
        """
        inputs = self.data.inputs
        rate = self.data.rate_of_capture
        rcapt = self.data.radiative_capture_cross_section
        pion = self.data.photoionization_cross_section
        derived = self.data.derived
        tsr = self.data.trap_state_rc

        if inputs.trap_state == "don":
            # electron capture coefficient from the CB by donar(initial charge state +ve)
            tsr.r_sign = derived.sa * pion.Ccoeff[2, :][::-1]
            # hole capture coefficent from VB by donar(initial charge state neu)
            tsr.r_sigp = pion.Ccoeff[1, :]

        else:
            # electron capture coefficient from the CB by acceptor(initial charge state neu)
            tsr.r_sign = pion.Ccoeff[1, :][::-1]
            # hole capture coefficent from the acceptor to VB(initial charge state -ve)
            tsr.r_sigp = derived.sa * pion.Ccoeff[0, :]
