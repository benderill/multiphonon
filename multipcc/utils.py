import numpy as np
import scipy.constants as sc
from scipy import sqrt

'''
Utility and support functions for multipcc.
'''


def physical_constants(data):
    phycon = data.root.physical_constants

    # eV to Joule conversion factor
    amu_kg, _, _ = sc.physical_constants['atomic mass unit-kilogram relationship']
    # eV to Joule conversion factor
    eVJ, _, _ = sc.physical_constants['electron volt-joule relationship']
    # Bohr Radius in m
    a_br, _, _ = sc.physical_constants['Bohr radius']
    # Rydberg constant in eV
    r_h, _, _ = sc.physical_constants['Rydberg constant times hc in eV']
    # planck constant over 2pi in Js
    hbar, _, _ = sc.physical_constants['Planck constant over 2 pi']
    # planck constant in Js
    hplanck, _, _ = sc.physical_constants['Planck constant']
    # charge of one electron
    Qe, _, _ = sc.physical_constants['elementary charge']
    # electron mass in kg
    m_e, _, _ = sc.physical_constants['electron mass']
    # boltzmann constant in JK^-1
    kBoltzmann, _, _ = sc.physical_constants['Boltzmann constant']
    # dimensionless
    alpha, _, _ = sc.physical_constants['fine-structure constant']
    # speed of light in vacuum in m/s
    c, _, _ = sc.physical_constants['speed of light in vacuum']
    epsilon_0 = 8.85418782E-12  # in F/m

    phycon.amu_kg = amu_kg
    phycon.eVJ = eVJ
    phycon.a_br = a_br
    phycon.r_h = r_h
    phycon.hbar = hbar
    phycon.hplanck = hplanck
    phycon.Qe = Qe
    phycon.m_e = m_e
    phycon.kB = kBoltzmann
    phycon.alpha = alpha
    phycon.c = c
    phycon.eps_0 = epsilon_0


def description(data):
    description = data.root.description
    description.current_simulation = 'Case 3: Directly calculate the radiative capture cross section by putting the photon density of states as 1/(2pi^2)((hbar omega)^2/(hbar c/eta_R)^3).'


def input_parameters(data, obj):
    phycon = data.root.physical_constants
    inputs = data.root.inputs

    inputs.mat = obj.mat  # Material Name (string)
    inputs.T = obj.T  # Temperature in kelvin (float)
    # inputs.a_0 = obj.a_0 * cbrt(1/4) * 1E-10           #(1/4)^(1/3) of lattice constant in meters ( for face centered unit cell)
    # lattice constant for cubic unit cell in meters (float)
    inputs.a_0 = obj.a_0 * 1E-10
    inputs.Eg = obj.Eg  # bandgap in eV (float)
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
        (8 * sc.pi * inputs.epsilon_l * derived.a_ebr * phycon.eVJ)
    # refractive index of the material
    derived.eta_r = sqrt(inputs.epsilon_h / phycon.eps_0)
    derived.sa = 4 * \
        sqrt(sc.pi * derived.r_eh *
             phycon.eVJ / (phycon.kB * inputs.T))


def bias_voltage(data):
    inputs = data.root.inputs
    VB = data.root.bias_voltage

    VB.Vb = np.arange(inputs.V_start, inputs.V_stop + inputs.dV, inputs.dV)
    VB.NVb = len(VB.Vb)


def coulomb_factor(data):
    phycon = data.root.physical_constants
    derived = data.root.derived
    egrid = data.root.energy_grids
    cf = data.root.coulomb_factor
    mat = data.root.matrix

    cf.CF_pos = np.zeros(mat.mat2D.shape)
    cf.CF_neg = np.zeros(mat.mat2D.shape)
    cf.X = np.sqrt(derived.r_eh / egrid.Ek)
    cf.CF_pos = (2 * sc.pi * np.sqrt(cf.X)) / \
        (1 - np.exp(-2 * sc.pi * cf.X))
    cf.CF_pos = np.broadcast_to(cf.CF_pos, (egrid.ET.size, egrid.Ek.size))
    cf.a = np.exp(-2 * sc.pi * cf.X)
    cf.b = (1 - np.exp(-2 * sc.pi * cf.X))

    cf.CF_neg = (-1 * 2 * sc.pi * np.sqrt(cf.X)) / \
        (1 - np.exp(-2 * sc.pi * -1 * cf.X))
    cf.CF_neg = np.broadcast_to(cf.CF_neg, (egrid.ET.size, egrid.Ek.size))
    cf.c = np.exp(-2 * sc.pi * -1 * cf.X)
    cf.d = (1 - np.exp(-2 * sc.pi * -1 * cf.X))
    cf.unity = np.ones(mat.mat2D.shape)
    cf.CF3D = np.array([cf.unity, cf.unity, cf.unity])

    cf.K = 2 * sc.pi * np.sqrt(derived.r_eh / egrid.Ek)


def w_function(data, arg, axis):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    mat = data.root.matrix
    wf = data.root.weighing_function

    wf.Gc = (8 * sqrt(2) * sc.pi / phycon.hplanck**3) * \
        sqrt(inputs.M_eff)**3 * \
        np.sqrt((mat.Ec - inputs.Eg) * phycon.eVJ)  # DOS
    wf.fE = np.exp((inputs.Eg/2 - mat.Ec) * phycon.eVJ /
                   (phycon.kB * inputs.T))  # fermidirac fucntion

    # carrier density per volume per energy
    wf.pear = wf.Gc * wf.fE * inputs.dE * phycon.eVJ

    # function weighted to occupied states
    wf.weighted_numerator = (arg * wf.pear).sum(axis=axis)
    # total number of carrier per volume, after integration over energy
    wf.denominator = wf.pear.sum(axis=0)
    wf.weighted_func = wf.weighted_numerator / wf.denominator  # weighted function
    return(wf.weighted_func)


def w_function_2(data, arg, axis):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    mat = data.root.matrix
    wf2 = data.root.weighing_function

    wf2.Gc = (8 * sqrt(2) * sc.pi / phycon.hplanck**3) * \
        sqrt(inputs.M_eff)**3 * \
        np.sqrt((mat.Ec - inputs.Eg) * phycon.eVJ)  # DOS
    wf2.fE = np.exp((inputs.Eg/2 - mat.Ec) * phycon.eVJ /
                    (phycon.kB * inputs.T))  # fermidirac fucntion

    # no of empty states per volume per energy
    wf2.pear = wf2.Gc * (1 - wf2.fE) * inputs.dE * phycon.eVJ

    # function weighted to unoccupied states
    wf2.weighted_numerator = (arg * wf2.pear).sum(axis=axis)
    # total number of empty states  volume, after integration over energy
    wf2.denominator = wf2.pear.sum(axis=0)
    wf2.weighted_func = wf2.weighted_numerator / \
        wf2.denominator  # weighted function
    return(wf2.weighted_func)


def energy_grids(data):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    derived = data.root.derived
    egrid = data.root.energy_grids

    # Distance of the defect from the band(the energy grids are so chosen to get the right nu)
    egrid.ET = np.linspace(4*derived.r_eh, inputs.Eg - 4*derived.r_eh, 1000)
    # Distance of the defect from the nearest band
    egrid.deltaE = np.minimum(egrid.ET, inputs.Eg - egrid.ET)
    egrid.nu = np.sqrt(derived.r_eh / egrid.deltaE)  # nu
    # the maximum final energy is 3KbT away
    egrid.Ekmax = 3 * phycon.kB * inputs.T / phycon.eVJ
    # the final energy mesh.
    egrid.Ek = np.arange(inputs.dE, egrid.Ekmax, inputs.dE)
    # the corresponding k values
    egrid.k = sqrt(egrid.Ek * phycon.eVJ * 2 *
                   inputs.M_eff / phycon.hbar**2)


def matrices(data):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    derived = data.root.derived
    egrid = data.root.energy_grids
    dos = data.root.effective_density_of_states
    VB = data.root.bias_voltage
    mat = data.root.matrix

    mat.mat2D = np.zeros([egrid.ET.size, egrid.Ek.size])  # xy

    # 2D matrix of trap energy [ET x Ek]
    mat.ET2D = np.broadcast_to(egrid.ET, (egrid.Ek.size, egrid.ET.size)).T
    # 2D matrix of final energy state [ET x Ek]
    mat.Ek2D = np.broadcast_to(egrid.Ek, (egrid.ET.size, egrid.Ek.size))
    # 3D matrix of final energy state [3 x ET x Ek]
    mat.Ek3D = np.broadcast_to(egrid.Ek, (3, egrid.ET.size, egrid.Ek.size))

    mat.Ec = egrid.Ek + inputs.Eg
    mat.Ev = -egrid.Ek

    # 2D matrix of nu [ET x Ek]
    mat.nu2D = np.broadcast_to(egrid.nu, (egrid.Ek.size, egrid.ET.size)).T
    mat.nu3D = np.tile(mat.nu2D, (3, 1)).reshape(
        3, egrid.ET.size, egrid.Ek.size)  # 3D matrix of nu [3 x ET x Ek]
