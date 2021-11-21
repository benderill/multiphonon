import math
import numpy as np
import scipy.constants
from scipy.special import gamma
from scipy.special import iv
from multipcc.utils import w_function, w_function_2


def photon_energy(data):
    """
    In photon energy grid all matrices are 2D. Along rows, ET remains fixed. along columns Ek remains fixed.
    Ept increases from left to right due to increase in Ek from left to right and increases from top to bottom due to increase of ET from top to bottom
    """
    mat = data.matrix
    egrid = data.energy_grids
    pgrid = data.photon_energy

    pgrid.Ept = (
        (np.zeros(mat.mat2D.shape) + egrid.Ek).T + egrid.ET
    ).T  # photon energy grid
    pgrid.theta = mat.Ek2D / mat.ET2D  # theta grid of the shape of Ept


def charge_states(data):
    """
    In charge state grid, all matrices are 3D, formed by repeating the matrices of photon energy grid three times for three charge states. ET,Ek,deltaE and nu consist of three equal blocks. mu is -nu, 0 and nu for -ve, neutral and +ve charge state respectively.
    """
    mat = data.matrix
    pgrid = data.photon_energy
    cgrid = data.charge_states

    cgrid.mu = np.array([-mat.nu2D, mat.nu2D * 1e-6, mat.nu2D])  # mu is 3D matrix
    # Ept grid of the shape of mu
    cgrid.Ept = np.array([pgrid.Ept, pgrid.Ept, pgrid.Ept])
    # theta grid of the shape of mu
    cgrid.theta = np.array([pgrid.theta, pgrid.theta, pgrid.theta])


def broadening_function(data):
    hrf = data.huang_rhys_factor
    phycon = data.physical_constants
    inputs = data.inputs
    bfunc = data.broadening_function

    bfunc.fB = 1 / (scipy.exp((inputs.Eph * phycon.eVJ) / (phycon.kB * inputs.T)) - 1)
    bfunc.bessel = iv(hrf.SHR, 2 * hrf.SHR * scipy.sqrt(bfunc.fB * (bfunc.fB + 1)))
    bfunc.broadening = (
        scipy.exp(-2 * hrf.SHR * (bfunc.fB + 1))
        * scipy.exp(hrf.SHR * inputs.Eph * phycon.eVJ / (phycon.kB * inputs.T))
        * bfunc.bessel
    )


def photoionization_cross_section(data):
    """
    Photoionization cross section found using eq.5.91 from QPC by BKR. Unit is [m^2].
    For negative charege state the values will be invalid for nu >= 0.5. So nu is taken from 0.5.
    The remaining of the calculations are done keepting those values masked.
    The cross section is weighted to the unoccupied states to get rid of the final state energy dependence
    The weighted p_crosssection is multiplied by the thermal velocity to get the coefficient
    """
    phycon = data.physical_constants
    inputs = data.inputs
    derived = data.derived
    mat = data.matrix
    cgrid = data.charge_states
    pion = data.photoionization_cross_section
    # bfunc = data.broadening_function
    # wf = data.weighing_function

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
    pion.PCS_vMu = (2 ** (2 * cgrid.mu)) * pion.Gamma * (mat.nu3D * derived.a_ebr) ** 3
    pion.PCS_vE = ((np.sqrt(mat.Ek3D * phycon.eVJ)) ** 3 / (cgrid.Ept * phycon.eVJ)) * (
        (np.sin((cgrid.mu + 1) * np.arctan(np.sqrt(cgrid.theta)))) ** 2
        / (cgrid.theta * (1 + cgrid.theta) ** (cgrid.mu + 1))
    )

    pion.PCS_E = pion.PCS_C * pion.PCS_vMu * pion.PCS_vE
    # photoionization cross section before weighing or summation overEk
    pion.PCS_E = np.ma.masked_less_equal(pion.PCS_E, 0)

    # weighed by density of unoccupied states per volume
    pion.PCS = w_function_2(data, pion.PCS_E, 2)
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
        mat.Ek3D * phycon.eVJ * (cgrid.Ept * phycon.eVJ) ** 2 / (cgrid.Ept * phycon.eVJ)
    ) * (
        (np.sin((cgrid.mu + 1) * np.arctan(np.sqrt(cgrid.theta)))) ** 2
        / (cgrid.theta * (1 + cgrid.theta) ** (cgrid.mu + 1))
    )
    pion.sigma_k_Energy = pion.sigma_k_c * pion.sigma_k_mu * pion.sigma_k_E
    # weighed by density of unoccupied states per volume
    pion.Ccoeff = w_function(data, pion.sigma_k_Energy, 2) * inputs.v_th
    # pion.Ccoeff = pion.Ccoeff * bfunc.broadening


def radiative_capture_cross_section(data):
    """
    Capture cross section is found using eq 5.30 of QPC by BKR. Unit is [m^2].
    The photon wave cector q in eq 5.30 is found using eq. 5.72 by replacing the numerator with our photon energy Ept.
    """
    phycon = data.physical_constants
    inputs = data.inputs
    derived = data.derived
    mat = data.matrix
    cgrid = data.charge_states
    pion = data.photoionization_cross_section
    # bfunc = data.broadening_function
    rcapt = data.radiative_capture_cross_section

    # capture cross section : CCS

    rcapt.factor = (cgrid.Ept * phycon.eVJ) ** 2 / (
        2 * inputs.M_eff * (phycon.c / derived.eta_r) ** 2 * mat.Ek3D * phycon.eVJ
    )  # DIMENSIONLESS
    rcapt.CCS_E = rcapt.factor * pion.PCS_E
    # capture cross section before weighing or summation overEk
    rcapt.CCS_E = np.ma.masked_array(rcapt.CCS_E, pion.PCS_E.mask)
    # weighed by density of unoccupied states per volume
    rcapt.CCS = w_function(data, rcapt.CCS_E, 2)
    rcapt.Ccoeff = rcapt.CCS * inputs.v_th  # capture coefficient
    rcapt.extrafactor = rcapt.Ccoeff / pion.Ccoeff


def trap_state_rc(data):
    """
    Chosing the right capture coefficinets according to charge state.
    When the capture is from CB, the corresponding photo energy decreases as in the begining the trap is close to the VB. Thus the rates of capture
    from the CB are always reversed.
    when the state is charged, the coefficent of capture is enhanced by coulomb factor / sommerfeld factor
    """
    inputs = data.inputs
    rate = data.rate_of_capture
    rcapt = data.radiative_capture_cross_section
    pion = data.photoionization_cross_section
    derived = data.derived
    tsr = data.trap_state_rc

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
