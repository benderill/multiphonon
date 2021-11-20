import numpy
import scipy.constants
from scipy.special import cbrt


def input_parameters(data, obj):
    phycon = data.root.physical_constants
    inputs = data.root.inputs

    inputs.mat = obj.mat  # Material
    inputs.T = obj.T  # Temperature in K
    # inputs.a_0 = obj.a_0 * cbrt(1/4) * 1E-10           #(1/4)^(1/3) of lattice constant in meters ( for face centered unit cell)
    inputs.a_0 = obj.a_0 * 1E-10  # for cubic unit cell
    inputs.Eg = obj.Eg  # bandgap in eV
    inputs.Ec = obj.Eg
    inputs.Ev = 0
    # relative permittivity at low frequency in F/m
    inputs.epsilon_l = obj.epsilon_l * phycon.eps_0
    # relative permittivity at high frequency in F/m
    inputs.epsilon_h = obj.epsilon_h * phycon.eps_0
    inputs.M_eff = obj.M_eff * phycon.m_e  # effective mass in kg
    inputs.Mr = obj.Mr * phycon.amu_kg  # Reduced mass in kg
    inputs.Eph = obj.Eph  # phonon energy in eV
    inputs.Dij = obj.Dij  # deformation potential constant in J/m
    inputs.Nt = obj.Nt  # trap denisty in cm^-3
    inputs.trap_state = obj.trap_state  # trap state
    inputs. v_th = obj.v_th / 100  # thermal velocity in m/s
    # fixed electron capture cross section in m2 to be used in simple SRH
    inputs.sign = obj.sign * 1e-4
    # fixed hole capture cross section in m2 to be used in simple SRH
    inputs.sigp = obj.sigp * 1e-4
    inputs.V_start = obj.V_start  # starting value of applied bias voltage
    inputs.V_stop = obj.V_stop  # last value of bias voltage
    inputs.dV = obj.dV  # step size of bias voltage
    inputs.dE = obj.dE  # step size for final energy mesh
    inputs.dir = obj.dir  # multiphonon coupling type


def derived_parameters(data):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    derived = data.root.derived

    derived.a_ebr = phycon.a_br * (inputs.epsilon_l / phycon.eps_0) / (
        inputs.M_eff / phycon.m_e)  # effective Bohr radius in meters
    # effective Rydberg energy in eV.Divide by e to get in eV. Value in meV=2.4
    derived.r_eh = phycon.Qe**2 / \
        (8 * scipy.pi * inputs.epsilon_l * derived.a_ebr * phycon.eVJ)
    # refractive index of the material
    derived.eta_r = scipy.sqrt(inputs.epsilon_h / phycon.eps_0)
    derived.sa = 4 * \
        scipy.sqrt(scipy.pi * derived.r_eh *
                   phycon.eVJ / (phycon.kB * inputs.T))


