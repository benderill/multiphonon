material = "MAPI"
temperature = 300  # Temperature in K
lattice_constant = 6.3  # lattice constant in Angstrome
bandgap = 1.6  # bandgap in eV
# relative permittivity at low frequency # relative permittivity at high frequency
relative_permittivity = {
    "low_freq": 33.5,
    "high_freq": 5,
}
mass_effective = 0.2  # effective mass
mass_reduced = 78.7  # reduced mass
phonon_energy = 16.5e-3  # phonon energy in eV
deformation_potential = 6e8  # deformation potential constant in eV/cm
trap_density = 1e15  # trap density in cm^-3
thermal_velocity = 1e7  # thermal velocity in cm/s
trap_state = "don"  # trap state. 'don' for donar, 'acc' for acceptor
# electron capture cross section in cm2 for simple SRH,hole capture cross section in cm2 for simple SRH
capture_crossection = {
    "electron": 1e-16,
    "hole": 1e-16,
}
bias_voltage = {"start": 0, "stop": 1, "step": 0.1}
energy_mesh_step = 1e-3  # energy mesh step size
coupling_type = "pc"  # 'dp' for deformation potential coupling, 'pc' for polar optical coupling, 'com' for combined coupling
