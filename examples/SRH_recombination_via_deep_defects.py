import numpy
import math
import scipy.constants
from scipy.special import cbrt
from scipy.special import gamma
import matplotlib.pyplot as plt
from multipcc import h5 , description  , physical_constants , input_data , bias_voltage , taysim , mesh , coulomb_factor , weighing_function , radiative_capture , multiphonon , plot_data

# Warning: input_peros is an example it is nolonger imported as a module, check the main clause for import 

def main(data, obj, output):
    description.desciption(data)
    physical_constants.physical_constants(data)
    input_data.input_parameters(data, obj)
    input_data.derived_parameters(data)
    bias_voltage.bias_voltage(data)
    mesh.energy_grids(data)
    mesh.matrices(data)

    # calculation of capture cross section and capture coefficients due to multiphonon capture
    multiphonon.derived_parameters(data)
    multiphonon.Huang_Rhys_Factor_deformation_potential_coupling(data)
    multiphonon.Huang_Rhys_Factor_polar_coupling(data)
    multiphonon.Huang_Rhys_factor(data)
    multiphonon.multiphonon_capture_coefficients(data)
    multiphonon.trap_state_mp(data)

    # calculation of photoionization cross section, capture cross section & capture coefficients due to radiative capture
    radiative_capture.photon_energy(data)
    radiative_capture.charge_states(data)
    radiative_capture.broadening_function(data)
    radiative_capture.photoionization_cross_section(data)
    radiative_capture.radiative_capture_cross_section(data)
    radiative_capture.trap_state_rc(data)

    # calculation of defect occupation probability, recombination efficienecy and rate of SRH recombination
    taysim.effective_density_of_states(data)
    taysim.occupation_probability(data)
    taysim.eta_R_normalized(data)
    taysim.test(data)
    data.save()

    plot_data.plot_nu(data)
    plot_data.plot_HRF(data)
    plot_data.plot_radiative_capture_coefficients(data)
    plot_data.plot_multiphonon_capture_coefficients(data)
    plot_data.extrafactor(data)
    plot_data.twinx_rad(data)
    plot_data.twinx_mp(data)
    plot_data.twinx_combined(data)
    plot_data.twinx_constant(data)
    plot_data.plot_occupation_probability(data)
    plot_data.plot_recombination_efficiency(data)
    plot_data.plot_rate_SRH(data)


if __name__ == "__main__":
    import json
    from addict import Dict
    with open('MAPI.json', 'r') as json_file:
        input_peros= json.load(json_file)
    input_peros = Dict(input_peros)
    data = h5.H5()
    data.filename = "Peros_knkp=1.h5"
    with open('output.txt', 'w+') as output:
        main(data, input_peros, output)
