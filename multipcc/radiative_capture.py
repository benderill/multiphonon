import numpy
import math
import scipy.constants
from scipy.special import gamma
from scipy.special import iv
from multipcc import weighing_function


# **********************************************************************************************************************************************
# In photon energy grid all matrices are 2D. Along rows, ET remains fixed. along columns Ek remains fixed.
# Ept increases from left to right due to increase in Ek from left to right and increases from top to bottom due to increase of ET from top to bottom
# **********************************************************************************************************************************************

def photon_energy(data):
    phycon = data.root.physical_constants
    derived = data.root.derived
    mat = data.root.matrix
    egrid = data.root.energy_grids
    pgrid = data.root.photon_energy

    pgrid.Ept = ((numpy.zeros(mat.mat2D.shape) + egrid.Ek).T +
                 egrid.ET).T  # photon energy grid
    pgrid.theta = mat.Ek2D/mat.ET2D  # theta grid of the shape of Ept


# **********************************************************************************************************************************************
# In charge state grid, all matrices are 3D, formed by repeating the matrices of photon energy grid three times for three charge states.
# ET,Ek,deltaE and nu consist of three equal blocks.
# mu is -nu, 0 and nu for -ve, neutral and +ve charge state respectively.
# **********************************************************************************************************************************************

def charge_states(data):
    phycon = data.root.physical_constants
    derived = data.root.derived
    egrid = data.root.energy_grids
    mat = data.root.matrix
    pgrid = data.root.photon_energy
    cgrid = data.root.charge_states

    cgrid.mu = numpy.array(
        [-mat.nu2D, mat.nu2D*1e-6, mat.nu2D])  # mu is 3D matrix
    # Ept grid of the shape of mu
    cgrid.Ept = numpy.array([pgrid.Ept, pgrid.Ept, pgrid.Ept])
    # theta grid of the shape of mu
    cgrid.theta = numpy.array([pgrid.theta, pgrid.theta, pgrid.theta])


def broadening_function(data):
    hrf = data.root.huang_rhys_factor
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    bfunc = data.root.broadening_function

    bfunc.fB = 1 / (scipy.exp((inputs.Eph * phycon.eVJ) /
                    (phycon.kB * inputs.T)) - 1)
    bfunc.bessel = iv(hrf.SHR, 2 * hrf.SHR *
                      scipy.sqrt(bfunc.fB * (bfunc.fB + 1)))
    bfunc.broadening = scipy.exp(-2 * hrf.SHR * (bfunc.fB+1)) * scipy.exp(
        hrf.SHR * inputs.Eph * phycon.eVJ / (phycon.kB * inputs.T)) * bfunc.bessel


# **********************************************************************************************************************************************
# Photoionization cross section found using eq.5.91 from QPC by BKR. Unit is [m^2].
# For negative charege state the values will be invalid for nu >= 0.5. So nu is taken from 0.5.
# The remaining of the calculations are done keepting those values masked.
# The cross section is weighted to the unoccupied states to get rid of the final state energy dependence
# The weighted p_crosssection is multiplied by the thermal velocity to get the coefficient
# **********************************************************************************************************************************************

def photoionization_cross_section(data):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    derived = data.root.derived
    mat = data.root.matrix
    cgrid = data.root.charge_states
    pion = data.root.photoionization_cross_section
    bfunc = data.root.broadening_function
    wf = data.root.weighing_function

    # photoionization cross section (PCS)
    # constant part of PCS: PCS_C
    # Mu varrying part of PCS: PCS_vMU
    # E varrying part of PCS: PCS_vE

    pion.PCS_C = (16 / 3) * phycon.alpha * phycon.a_br**2 * phycon.r_h * phycon.eVJ * (1 / derived.eta_r) * \
        (phycon.m_e / inputs.M_eff) * (2 * scipy.pi) * \
        (numpy.sqrt(2 * inputs.M_eff) / phycon.hbar)**3
    pion.Gamma = (gamma(cgrid.mu + 1))**2 / gamma(2 * cgrid.mu + 1)
    pion.PCS_vMu = (2**(2 * cgrid.mu)) * pion.Gamma * \
        (mat.nu3D * derived.a_ebr)**3
    pion.PCS_vE = ((numpy.sqrt(mat.Ek3D * phycon.eVJ))**3 / (cgrid.Ept * phycon.eVJ)) * ((numpy.sin((cgrid.mu + 1)
                                                                                                    * numpy.arctan(numpy.sqrt(cgrid.theta))))**2 / (cgrid.theta * (1 + cgrid.theta)**(cgrid.mu + 1)))

    pion.PCS_E = pion.PCS_C * pion.PCS_vMu * pion.PCS_vE
    # photoionization cross section before weighing or summation overEk
    pion.PCS_E = numpy.ma.masked_less_equal(pion.PCS_E, 0)

    # weighed by density of unoccupied states per volume
    pion.PCS = weighing_function.w_function_2(data, pion.PCS_E, 2)
    pion.PCoeff = pion.PCS * inputs.v_th  # photoionization coefficient

    pion.sa = 4 * math.sqrt(scipy.pi * derived.r_eh * phycon.eVJ /
                            (phycon.kB * inputs.T))  # sommerfiled factor

    pion.sigma_k_c = (16 / 3) * phycon.alpha * phycon.a_br**2 * phycon.r_h * phycon.eVJ * (1 / derived.eta_r) * \
        (phycon.m_e / inputs.M_eff) * (2 * scipy.pi) / \
        (phycon.hbar * phycon.c / derived.eta_r)**3
    pion.sigma_k_gamma = (gamma(cgrid.mu + 1))**2 / gamma(2 * cgrid.mu + 1)
    pion.sigma_k_mu = (2**(2 * cgrid.mu)) * \
        pion.sigma_k_gamma * (mat.nu3D * derived.a_ebr)**3
    pion.sigma_k_E = (mat.Ek3D * phycon.eVJ * (cgrid.Ept * phycon.eVJ)**2 / (cgrid.Ept * phycon.eVJ)) * ((numpy.sin(
        (cgrid.mu + 1) * numpy.arctan(numpy.sqrt(cgrid.theta))))**2 / (cgrid.theta * (1 + cgrid.theta)**(cgrid.mu + 1)))
    pion.sigma_k_Energy = pion.sigma_k_c * pion.sigma_k_mu * pion.sigma_k_E
    # weighed by density of unoccupied states per volume
    pion.Ccoeff = weighing_function.w_function(
        data, pion.sigma_k_Energy, 2) * inputs.v_th
    #pion.Ccoeff = pion.Ccoeff * bfunc.broadening

# **********************************************************************************************************************************************
# Capture cross section is found using eq 5.30 of QPC by BKR. Unit is [m^2].
# The photon wave cector q in eq 5.30 is found using eq. 5.72 by replacing the numerator with our photon energy Ept.
# **********************************************************************************************************************************************


def radiative_capture_cross_section(data):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    derived = data.root.derived
    mat = data.root.matrix
    cgrid = data.root.charge_states
    pion = data.root.photoionization_cross_section
    bfunc = data.root.broadening_function
    rcapt = data.root.radiative_capture_cross_section

    # capture cross section : CCS

    rcapt.factor = (cgrid.Ept * phycon.eVJ)**2 / (2 * inputs.M_eff *
                                                  (phycon.c / derived.eta_r)**2 * mat.Ek3D * phycon.eVJ)  # DIMENSIONLESS
    rcapt.CCS_E = rcapt.factor * pion.PCS_E
    # capture cross section before weighing or summation overEk
    rcapt.CCS_E = numpy.ma.masked_array(rcapt.CCS_E, pion.PCS_E.mask)
    # weighed by density of unoccupied states per volume
    rcapt.CCS = weighing_function.w_function(data, rcapt.CCS_E, 2)
    rcapt.Ccoeff = rcapt.CCS * inputs.v_th  # capture coefficient
    rcapt.extrafactor = rcapt.Ccoeff / pion.Ccoeff


# **********************************************************************************************************************************************
# Chosing the right capture coefficinets according to charge state.
# When the capture is from CB, the corresponding photo energy decreases as in the begining the trap is close to the VB. Thus the rates of capture
# from the CB are always reversed.
# when the state is charged, the coefficent of capture is enhanced by coulomb factor / sommerfeld factor
# **********************************************************************************************************************************************

def trap_state_rc(data):
    inputs = data.root.inputs
    rate = data.root.rate_of_capture
    rcapt = data.root.radiative_capture_cross_section
    pion = data.root.photoionization_cross_section
    derived = data.root.derived
    tsr = data.root.trap_state_rc

    if inputs.trap_state == 'don':
        # electron capture coefficient from the CB by donar(initial charge state +ve)
        tsr.r_sign = derived.sa * pion.Ccoeff[2, :][::-1]
        # hole capture coefficent from VB by donar(initial charge state neu)
        tsr.r_sigp = pion.Ccoeff[1, :]

    else:
        # electron capture coefficient from the CB by acceptor(initial charge state neu)
        tsr.r_sign = pion.Ccoeff[1, :][::-1]
        # hole capture coefficent from the acceptor to VB(initial charge state -ve)
        tsr.r_sigp = derived.sa * pion.Ccoeff[0, :]


