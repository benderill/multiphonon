import numpy
import scipy.constants


def physical_constants(data):
    phycon = data.root.physical_constants

    # physical constants
    # eV to Joule conversion factor
    (amu_kg, _,
     _) = scipy.constants.physical_constants['atomic mass unit-kilogram relationship']
    # eV to Joule conversion factor
    (eVJ, _,
     _) = scipy.constants.physical_constants['electron volt-joule relationship']
    # Bohr Radius in m
    (a_br, _, _) = scipy.constants.physical_constants['Bohr radius']
    # Rydberg constant in eV
    (r_h, _,
     _) = scipy.constants.physical_constants['Rydberg constant times hc in eV']
    # planck constant over 2pi in Js
    (hbar, _,
     _) = scipy.constants.physical_constants['Planck constant over 2 pi']
    # planck constant in Js
    (hplanck, _, _) = scipy.constants.physical_constants['Planck constant']
    # charge of one electron
    (Qe, _, _) = scipy.constants.physical_constants['elementary charge']
    # electron mass in kg
    (m_e, _, _) = scipy.constants.physical_constants['electron mass']
    # boltzmann constant in JK^-1
    (kBoltzmann, _,
     _) = scipy.constants.physical_constants['Boltzmann constant']
    # dimensionless
    (alpha, _,
     _) = scipy.constants.physical_constants['fine-structure constant']
    # speed of light in vacuum in m/s
    (c, _, _) = scipy.constants.physical_constants['speed of light in vacuum']
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


if __name__ == "__main__":
    data = h5.H5()
    data.filename = "C:/Users/basit/Simulations_and_results/Data/Paper1/SRH_deep_defects/Peros_Case_3.h5"
    with open('output.txt', 'w+') as output:
        main(data, input_peros, output)
