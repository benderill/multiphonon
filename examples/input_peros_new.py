material = "MAPI"  # (mat)
temperature = 300  # (T) Temperature in K
lattice_constant = {"a0": 6.3}  # (a_0) lattice constant in Angstrome
bandgap = 1.6  # (Eg) bandgap in eV
# relative permittivity at low frequency # relative permittivity at high frequency
relative_permittivity = {
    "low_freq": 33.5,  # (epsilon_l)
    "high_freq": 5,  # (epsilon_h)
}
mass_effective = 0.2  # (M_eff) effective mass
mass_reduced = 78.7  # (Mr) reduced mass
phonon_energy = 16.5e-3  # (Eph) phonon energy in eV
deformation_potential = 6e8  # (Dij) deformation potential constant in eV/cm
trap_density = 1e15  # (Nt) trap density in cm^-3
thermal_velocity = 1e7  # (v_th) thermal velocity in cm/s
trap_state = "don"  # (trap_state) trap state. 'don' for donar, 'acc' for acceptor
# electron capture cross section in cm2 for simple SRH,hole capture cross section in cm2 for simple SRH
capture_crossection = {
    "electron": 1e-16,  # (sign)
    "hole": 1e-16,  # (sigp)
}
bias_voltage = {"start": 0, "stop": 1, "step": 0.1}  # (V_start,V_stop,dV)
energy_mesh_step = 1e-3  #  (dE) energy mesh step size
coupling_type = "pc"  # (dir) 'dp' for deformation potential coupling, 'pc' for polar optical coupling, 'com' for combined coupling
