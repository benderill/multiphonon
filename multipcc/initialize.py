import json
import numpy as np
from addict import Dict
import scipy.constants as sc
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
        self.isinput = False

        if type(input) is str:
            try:
                with open(input, "r") as json_file:
                    inp_data = json.load(json_file)
                    self.data.inputs = Dict(inp_data)
                    self.isinput = True
            except FileNotFoundError:
                print("File does not exist")

        if self.isinput:
            self.set_constants()
            self.set_parameters()
            self.set_energy_grids()
            self.set_matrices()

    def set_constants(self):
        # eV to Joule conversion factor
        self.data.constants.amu_kg = sc.physical_constants[
            "atomic mass unit-kilogram relationship"
        ][0]
        # eV to Joule conversion factor
        self.data.constants.eVJ = sc.physical_constants[
            "electron volt-joule relationship"
        ][0]
        # Bohr Radius in m
        self.data.constants.a_br = sc.physical_constants["Bohr radius"][0]
        # Rydberg constant in eV
        self.data.constants.r_h = sc.physical_constants[
            "Rydberg constant times hc in eV"
        ][0]
        # planck constant over 2pi in Js
        self.data.constants.hbar = sc.physical_constants["Planck constant over 2 pi"][0]
        # planck constant in Js
        self.data.constants.hplanck = sc.physical_constants["Planck constant"][0]
        # charge of one electron
        self.data.constants.Qe = sc.physical_constants["elementary charge"][0]
        # electron mass in kg
        self.data.constants.m_e = sc.physical_constants["electron mass"][0]
        # boltzmann constant in JK^-1
        self.data.constants.kB = sc.physical_constants["Boltzmann constant"][0]
        # dimensionless
        self.data.constants.alpha = sc.physical_constants["fine-structure constant"][0]
        # speed of light in vacuum in m/s
        self.data.constants.c = sc.physical_constants["speed of light in vacuum"][0]
        self.data.constants.eps_0 = 8.85418782e-12  # in F/m

        return self.data.constants

    def get_constants(self):
        return self.data.constants

    def set_parameters(self):
        """
        Set the parameters from the input file.
        """
        self.data.parameters.mat = self.data.inputs.mat
        self.data.parameters.T = self.data.inputs.T
        self.data.parameters.a_0 = self.data.inputs.a_0 * 1e-10
        self.data.parameters.Eg = self.data.inputs.Eg
        self.data.parameters.Ec = self.data.inputs.Eg
        self.data.parameters.Ev = 0.0
        self.data.parameters.epsilon_l = (
            self.data.inputs.epsilon_l * self.data.constants.eps_0
        )
        self.data.parameters.epsilon_h = (
            self.data.inputs.epsilon_h * self.data.constants.eps_0
        )
        self.data.parameters.M_eff = self.data.inputs.M_eff * self.data.constants.m_e
        self.data.parameters.Mr = self.data.inputs.Mr * self.data.constants.amu_kg
        self.data.parameters.Eph = self.data.inputs.Eph
        self.data.parameters.Dij = self.data.inputs.Dij
        self.data.parameters.Nt = self.data.inputs.Nt
        self.data.parameters.trap_state = self.data.inputs.trap_state
        self.data.parameters.v_th = self.data.inputs.v_th / 100.0
        self.data.parameters.sign = self.data.inputs.sign * 1e-4
        self.data.parameters.sigp = self.data.inputs.sigp * 1e-4
        self.data.parameters.V_start = self.data.inputs.V_start
        self.data.parameters.V_stop = self.data.inputs.V_stop
        self.data.parameters.dV = self.data.inputs.dV
        self.data.parameters.dE = self.data.inputs.dE
        self.data.parameters.dir = self.data.inputs.dir

        # Calculate the other parameters parameters
        self.data.parameters.a_ebr = (
            self.data.constants.a_br
            * (self.data.parameters.epsilon_l / self.data.constants.eps_0)
            / (self.data.parameters.M_eff / self.data.constants.m_e)
        )  # effective Bohr radius in meters
        # effective Rydberg energy in eV.Divide by e to get in eV. Value in meV=2.4
        self.data.parameters.r_eh = self.data.constants.Qe ** 2 / (
            8
            * sc.pi
            * self.data.parameters.epsilon_l
            * self.data.parameters.a_ebr
            * self.data.constants.eVJ
        )
        # refractive index of the material
        self.data.parameters.eta_r = np.sqrt(
            self.data.parameters.epsilon_h / self.data.constants.eps_0
        )
        self.data.parameters.sa = 4 * np.sqrt(
            sc.pi
            * self.data.parameters.r_eh
            * self.data.constants.eVJ
            / (self.data.constants.kB * self.data.parameters.T)
        )

        # Calculate multiphonon parameters
        # radius of sphere with Brillouin zone volume in metre inverse
        self.data.parameters.q_D = np.cbrt(6 * np.pi ** 2) / self.data.parameters.a_0

        self.data.parameters.sa = 4 * np.sqrt(
            np.pi
            * self.data.parameters.r_eh
            * self.data.constants.eVJ
            / (self.data.constants.kB * self.data.parameters.T)
        )  # sommerfiled factor
        self.data.parameters.pekar = (1 / self.data.parameters.epsilon_h) - (
            1 / self.data.parameters.epsilon_l
        )  # pekar factor
        self.data.parameters.V_0 = (
            self.data.parameters.a_0
        ) ** 3  # volume of the unit cell in cubic meters
        self.data.parameters.omega = (
            self.data.parameters.Eph * self.data.constants.eVJ
        ) / self.data.constants.hbar  # frequecy of the phonon

    def get_parameters(self):
        return self.data.parameters

    def get_input_parameters(self):
        return self.data.inputs

    def set_energy_grids(self):
        # Distance of the defect from the band(the energy grids are so chosen to get the right nu)
        self.data.energy_grids.ET = np.linspace(
            4 * self.data.parameters.r_eh,
            self.data.parameters.Eg - 4 * self.data.parameters.r_eh,
            1000,
        )
        # Distance of the defect from the nearest band
        self.data.energy_grids.deltaE = np.minimum(
            self.data.energy_grids.ET,
            self.data.parameters.Eg - self.data.energy_grids.ET,
        )
        self.data.energy_grids.nu = np.sqrt(
            self.data.parameters.r_eh / self.data.energy_grids.deltaE
        )  # nu
        # the maximum final energy is 3KbT away
        self.data.energy_grids.Ekmax = (
            3
            * self.data.constants.kB
            * self.data.parameters.T
            / self.data.constants.eVJ
        )
        # the final energy mesh.
        self.data.energy_grids.Ek = np.arange(
            self.data.parameters.dE,
            self.data.energy_grids.Ekmax,
            self.data.parameters.dE,
        )
        # the corresponding k values
        self.data.energy_grids.k = np.sqrt(
            self.data.energy_grids.Ek
            * self.data.constants.eVJ
            * 2
            * self.data.parameters.M_eff
            / self.data.constants.hbar ** 2
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

        self.data.matrix.Ec = ek + self.data.parameters.Eg
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
            print("File name must be set before load can be used")

    def save(self):
        if self.data.filename is not None:
            h5_save(self.data.filename, self.data)
        else:
            print("File name must be set before save can be used")
