import numpy
import scipy.constants


def energy_grids(data):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    derived = data.root.derived
    egrid = data.root.energy_grids

    # Distance of the defect from the band(the energy grids are so chosen to get the right nu)
    egrid.ET = numpy.linspace(4*derived.r_eh, inputs.Eg - 4*derived.r_eh, 1000)
    # Distance of the defect from the nearest band
    egrid.deltaE = numpy.minimum(egrid.ET, inputs.Eg - egrid.ET)
    egrid.nu = numpy.sqrt(derived.r_eh / egrid.deltaE)  # nu
    # the maximum final energy is 3KbT away
    egrid.Ekmax = 3 * phycon.kB * inputs.T / phycon.eVJ
    # the final energy mesh.
    egrid.Ek = numpy.arange(inputs.dE, egrid.Ekmax, inputs.dE)
    # the corresponding k values
    egrid.k = scipy.sqrt(egrid.Ek * phycon.eVJ * 2 *
                         inputs.M_eff / phycon.hbar**2)


def matrices(data):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    derived = data.root.derived
    egrid = data.root.energy_grids
    dos = data.root.effective_density_of_states
    VB = data.root.bias_voltage
    mat = data.root.matrix

    mat.mat2D = numpy.zeros([egrid.ET.size, egrid.Ek.size])  # xy

    # 2D matrix of trap energy [ET x Ek]
    mat.ET2D = numpy.broadcast_to(egrid.ET, (egrid.Ek.size, egrid.ET.size)).T
    # 2D matrix of final energy state [ET x Ek]
    mat.Ek2D = numpy.broadcast_to(egrid.Ek, (egrid.ET.size, egrid.Ek.size))
    # 3D matrix of final energy state [3 x ET x Ek]
    mat.Ek3D = numpy.broadcast_to(egrid.Ek, (3, egrid.ET.size, egrid.Ek.size))

    mat.Ec = egrid.Ek + inputs.Eg
    mat.Ev = -egrid.Ek

    # 2D matrix of nu [ET x Ek]
    mat.nu2D = numpy.broadcast_to(egrid.nu, (egrid.Ek.size, egrid.ET.size)).T
    mat.nu3D = numpy.tile(mat.nu2D, (3, 1)).reshape(
        3, egrid.ET.size, egrid.Ek.size)  # 3D matrix of nu [3 x ET x Ek]


