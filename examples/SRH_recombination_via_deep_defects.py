from multipcc.initialize import Initialize
from multipcc.capture import Multiphonon, Radiative
import sys

sys.path.append("./")
import plot_data as plot_data


def main(input_file, output_file):
    calc = Initialize(input_file)
    calc.filename = output_file

    multiphonon = Multiphonon(calc.data)
    # calculation of capture cross section and capture coefficients due to multiphonon capture
    multiphonon.derived_parameters()
    multiphonon.Huang_Rhys_Factor_deformation_potential_coupling()
    multiphonon.Huang_Rhys_Factor_polar_coupling()
    multiphonon.Huang_Rhys_factor()
    multiphonon.Huang_Rhys_factor()
    multiphonon.multiphonon_capture_coefficients()
    multiphonon.trap_state_mp()

    # calculation of photoionization cross section, capture cross section & capture coefficients due to radiative capture
    radiative = Radiative(calc.data)

    radiative.photon_energy()
    radiative.charge_states()
    radiative.broadening_function()
    radiative.photoionization_cross_section()
    radiative.radiative_capture_cross_section()
    radiative.trap_state_rc()

    # calculation of defect occupation probability, recombination efficienecy and rate of SRH recombination
    # taysim.effective_density_of_states(data)
    # taysim.occupation_probability(data)
    # taysim.eta_R_normalized(data)
    # taysim.test(data)
    calc.save()

    plot_data.plot_nu(calc.data)
    plot_data.plot_HRF(calc.data)
    plot_data.plot_radiative_capture_coefficients(calc.data)
    plot_data.plot_multiphonon_capture_coefficients(calc.data)
    # plot_data.extrafactor(data)
    # plot_data.twinx_rad(data)
    # plot_data.twinx_mp(data)
    # plot_data.twinx_combined(data)
    # plot_data.twinx_constant(data)
    # plot_data.plot_occupation_probability(data)
    # plot_data.plot_recombination_efficiency(data)
    # plot_data.plot_rate_SRH(data)


if __name__ == "__main__":
    import json
    from addict import Dict

    input_file = "MAPI.json"
    output_file = "mapi.h5"
    main(input_file, output_file)
