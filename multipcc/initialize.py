from addict import Dict
import numpy as np
import scipy.constants as sc
import json
from multipcc.utils import h5_load, h5_save


class Initialize:
    def __init__(self, input=None):
        """Initialize the class.
        TODO: write a validater function for the input parameters.
        TODO: load and save methods for the data
        """
        self.data = Dict()
        self.data.filename = "default.h5"
        self.data.description = "Case 3: Directly calculate the radiative capture cross section by putting the photon density of states as 1/(2pi^2)((hbar omega)^2/(hbar c/eta_R)^3)."

        if type(input) is str:
            try:
                with open(input, "r") as json_file:
                    inp_data = json.load(json_file)
                    self.data.inputs = Dict(inp_data)
            except FileNotFoundError:
                print("File does not exist")
            self.physical_constants()
            self.set_input_parameters()
            self.set_derived_parameters()
            self.set_energy_grids()
            self.set_matrices()

    def set_physical_constants(self):
        # eV to Joule conversion factor
        self.data.physical_constants.amu_kg = sc.physical_constants[
            "atomic mass unit-kilogram relationship"
        ][0]
        # eV to Joule conversion factor
        self.data.physical_constants.eVJ = sc.physical_constants[
            "electron volt-joule relationship"
        ][0]
        # Bohr Radius in m
        self.data.physical_constants.a_br = sc.physical_constants["Bohr radius"][0]
        # Rydberg constant in eV
        self.data.physical_constants.r_h = sc.physical_constants[
            "Rydberg constant times hc in eV"
        ][0]
        # planck constant over 2pi in Js
        self.data.physical_constants.hbar = sc.physical_constants[
            "Planck constant over 2 pi"
        ][0]
        # planck constant in Js
        self.data.physical_constants.hplanck = sc.physical_constants["Planck constant"][
            0
        ]
        # charge of one electron
        self.data.physical_constants.Qe = sc.physical_constants["elementary charge"][0]
        # electron mass in kg
        self.data.physical_constants.m_e = sc.physical_constants["electron mass"][0]
        # boltzmann constant in JK^-1
        self.data.physical_constants.kB = sc.physical_constants["Boltzmann constant"][0]
        # dimensionless
        self.data.physical_constants.alpha = sc.physical_constants[
            "fine-structure constant"
        ][0]
        # speed of light in vacuum in m/s
        self.data.physical_constants.c = sc.physical_constants[
            "speed of light in vacuum"
        ][0]
        self.data.physical_constants.eps_0 = 8.85418782e-12  # in F/m

        return self.data.physical_constants

    def get_physical_constants(self):
        return self.data.physical_constants

    def set_input_parameters(self):
        # self.data.inputs.mat = obj.mat  # Material Name (string)
        # self.data.inputs.T = obj.T  # Temperature in kelvin (float)
        # self.data.inputs.a_0 = obj.a_0 * cbrt(1/4) * 1E-10           #(1/4)^(1/3) of lattice constant in meters ( for face centered unit cell)
        # lattice constant for cubic unit cell in meters (float)
        self.data.inputs.a_0 *= 1e-10
        # self.data.inputs.Eg = obj.Eg  # bandgap in eV (float)
        self.data.inputs.Ec = self.data.inputs.Eg
        self.data.inputs.Ev = 0
        # relative permittivity at low frequency in F/m
        self.data.inputs.epsilon_l *= self.data.physical_constants.eps_0
        # relative permittivity at high frequency in F/m
        self.data.inputs.epsilon_h *= self.data.physical_constants.eps_0
        # effective mass in kg
        self.data.inputs.M_eff *= self.data.physical_constants.m_e
        self.data.inputs.Mr *= self.data.physical_constants.amu_kg  # Reduced mass in kg
        # self.data.inputs.Eph = obj.Eph  # phonon energy in eV
        # self.data.inputs.Dij = obj.Dij  # deformation potential constant in J/m
        # self.data.inputs.Nt = obj.Nt  # trap denisty in cm^-3
        # self.data.inputs.trap_state = obj.trap_state  # trap state
        self.data.inputs.v_th /= 100  # thermal velocity in m/s
        # fixed electron capture cross section in m2 to be used in simple SRH
        self.data.inputs.sign *= 1e-4
        # fixed hole capture cross section in m2 to be used in simple SRH
        self.data.inputs.sigp *= 1e-4
        # self.data.inputs.V_start = obj.V_start  # starting value of applied bias voltage
        # self.data.inputs.V_stop = obj.V_stop  # last value of bias voltage
        # self.data.inputs.dV = obj.dV  # step size of bias voltage
        # self.data.inputs.dE = obj.dE  # step size for final energy mesh
        # self.data.inputs.dir = obj.dir  # multiphonon coupling type

    def get_input_parameters(self):
        return self.data.inputs

    def set_derived_parameters(self):
        """
        Parameters that are derived from the input parameters.
        """
        self.data.derived.a_ebr = (
            self.data.physical_constants.a_br
            * (self.data.inputs.epsilon_l / self.data.physical_constants.eps_0)
            / (self.data.inputs.M_eff / self.data.physical_constants.m_e)
        )  # effective Bohr radius in meters
        # effective Rydberg energy in eV.Divide by e to get in eV. Value in meV=2.4
        self.data.derived.r_eh = self.data.physical_constants.Qe ** 2 / (
            8
            * sc.pi
            * self.data.inputs.epsilon_l
            * self.data.derived.a_ebr
            * self.data.physical_constants.eVJ
        )
        # refractive index of the material
        self.data.derived.eta_r = np.sqrt(
            self.data.inputs.epsilon_h / self.data.physical_constants.eps_0
        )
        self.data.derived.sa = 4 * np.sqrt(
            sc.pi
            * self.data.derived.r_eh
            * self.data.physical_constants.eVJ
            / (self.data.physical_constants.kB * self.data.inputs.T)
        )

    def get_derived_parameters(self):
        return self.data.derived

    def set_energy_grids(self):
        # Distance of the defect from the band(the energy grids are so chosen to get the right nu)
        self.data.energy_grids.ET = np.linspace(
            4 * self.data.derived.r_eh,
            self.data.inputs.Eg - 4 * self.data.derived.r_eh,
            1000,
        )
        # Distance of the defect from the nearest band
        self.data.energy_grids.deltaE = np.minimum(
            self.data.energy_grids.ET, self.data.inputs.Eg - self.data.energy_grids.ET
        )
        self.data.energy_grids.nu = np.sqrt(
            self.data.derived.r_eh / self.data.energy_grids.deltaE
        )  # nu
        # the maximum final energy is 3KbT away
        self.data.energy_grids.Ekmax = (
            3
            * self.data.physical_constants.kB
            * self.data.inputs.T
            / self.data.physical_constants.eVJ
        )
        # the final energy mesh.
        self.data.energy_grids.Ek = np.arange(
            self.data.inputs.dE, self.data.energy_grids.Ekmax, self.data.inputs.dE
        )
        # the corresponding k values
        self.data.energy_grids.k = np.sqrt(
            self.data.energy_grids.Ek
            * self.data.physical_constants.eVJ
            * 2
            * self.data.inputs.M_eff
            / self.data.physical_constants.hbar ** 2
        )

    def get_energy_grids(self):
        return self.data.energy_grids

    def set_matrices(self):
        et = self.data.energy_grids.ET
        ek = self.data.energy_grids.Ek
        self.data.matrix.mat2D = np.zeros([et.size, ek.size])  # xy

        # 2D matrix of trap energy [ET x Ek]
        self.data.matrix.ET2D = np.broadcast_to(et, (ek.size, et.size)).T
        # 2D matrix of final energy state [ET x Ek]
        self.data.matrix.Ek2D = np.broadcast_to(ek, (et.size, ek.size))
        # 3D matrix of final energy state [3 x ET x Ek]
        self.data.matrix.Ek3D = np.broadcast_to(ek, (3, et.size, ek.size))

        self.data.matrix.Ec = ek + self.data.inputs.Eg
        self.data.matrix.Ev = -ek

        # 2D matrix of nu [ET x Ek]
        self.data.matrix.nu2D = np.broadcast_to(
            self.data.energy_grids.nu, (ek.size, et.size)
        ).T
        self.data.matrix.nu3D = np.tile(self.data.matrix.nu2D, (3, 1)).reshape(
            3, et.size, ek.size
        )  # 3D matrix of nu [3 x ET x Ek]

    def get_matrices(self):
        return self.data.matrix

    def load(self):
        if self.data.filename is not None:
            self.data = h5_load(self.data.filename)
        else:
            print("Filename must be set before load can be used")

    def save(self):
        if self.data.filename is not None:
            h5_save(self.data.filename, self.data)
        else:
            print("Filename must be set before save can be used")
