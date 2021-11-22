import math
import numpy as np


def effective_density_of_states(data):
    const = data.constants
    inputs = data.inputs
    VB = data.bias_voltage
    dos = data.effective_density_of_states

    # effective density of states in CB in [m^-3]
    dos.Nc = (
        2
        * np.sqrt((2 * np.pi * inputs.M_eff * const.kB * inputs.T) ** 3)
        / (const.hbar * 2 * np.pi) ** 3
    )
    dos.Nv = dos.Nc
    # intrinsic carrier concentration per m^3 [units m^-3]
    dos.ni = dos.Nc * math.exp(-inputs.Eg * const.eVJ / (2 * const.kB * inputs.T))
    # carrier density for different bias voltages
    dos.n = dos.ni * np.exp(VB.Vb * const.Qe / (2 * const.kB * inputs.T))
    dos.p = dos.n


def occupation_probability(data):
    const = data.constants
    inputs = data.inputs
    dos = data.effective_density_of_states
    egrid = data.energy_grids
    tsr = data.trap_state_rc
    tsm = data.trap_state_mp
    VB = data.bias_voltage
    ocp = data.occupation_probability

    # calculation of occupation probability, recombination efficiency and rate of recombination for simple SRH

    # matirx of carrier denisity [ET x NVb]
    ocp.n = np.broadcast_to(dos.n, (egrid.ET.size, VB.NVb)).T
    ocp.p = ocp.n

    # matrix of e-capture coefficient for simple SRH
    ocp.beta_n_s = np.broadcast_to(inputs.sign * inputs.v_th, (VB.NVb, egrid.ET.size))
    # matrix of h-capture coefficient for simple SRH
    ocp.beta_p_s = np.broadcast_to(inputs.sigp * inputs.v_th, (VB.NVb, egrid.ET.size))

    ocp.en_s = (
        ocp.beta_n_s
        * dos.Nc
        * np.exp((egrid.ET - inputs.Ec) * const.eVJ / (const.kB * inputs.T))
    )  # electrom emission rate
    ocp.ep_s = (
        ocp.beta_p_s
        * dos.Nv
        * np.exp((inputs.Ev - egrid.ET) * const.eVJ / (const.kB * inputs.T))
    )  # hole emissiom rate

    ocp.kn_s = ocp.n * ocp.beta_n_s  # electron capture rate
    ocp.kp_s = ocp.p * ocp.beta_p_s  # hole capture rate

    ocp.f_T_s = ((ocp.n * ocp.beta_n_s) + ocp.ep_s) / (
        (ocp.n * ocp.beta_n_s) + (ocp.p * ocp.beta_p_s) + ocp.en_s + ocp.ep_s
    )  # trap occupation probability

    ocp.eta_R_s = (ocp.n * ocp.beta_n_s * (1 - ocp.f_T_s)) - (
        ocp.en_s * ocp.f_T_s
    )  # recombination efficiency

    ocp.R_srh_s = inputs.Nt * ocp.eta_R_s  # rate of recombination

    # calculation of occupation probability, recombination efficiency and rate of recombination for radiative capture

    # matrix of e-capture coefficient for radiative SRH
    ocp.beta_n_r = np.broadcast_to(tsr.r_sign, (VB.NVb, egrid.ET.size))
    # matrix of h-capture coefficient for radiative SRH
    ocp.beta_p_r = np.broadcast_to(tsr.r_sigp, (VB.NVb, egrid.ET.size))

    ocp.en_r = (
        ocp.beta_n_r
        * dos.Nc
        * np.exp((egrid.ET - inputs.Ec) * const.eVJ / (const.kB * inputs.T))
    )  # electrom emission rate
    ocp.ep_r = (
        ocp.beta_p_r
        * dos.Nv
        * np.exp((inputs.Ev - egrid.ET) * const.eVJ / (const.kB * inputs.T))
    )  # hole emissiom rate

    ocp.kn_r = ocp.n * ocp.beta_n_r  # electron capture rate
    ocp.kp_r = ocp.p * ocp.beta_p_r  # hole capture rate

    ocp.f_T_r = ((ocp.n * ocp.beta_n_r) + ocp.ep_r) / (
        (ocp.n * ocp.beta_n_r) + (ocp.p * ocp.beta_p_r) + ocp.en_r + ocp.ep_r
    )  # trap occupation probability

    ocp.eta_R_r = (ocp.n * ocp.beta_n_r * (1 - ocp.f_T_r)) - (
        ocp.en_r * ocp.f_T_r
    )  # recombination efficiency

    ocp.R_srh_r = inputs.Nt * ocp.eta_R_r  # rate of recombination

    # calculation of occupation probability, recombination efficiency and rate of recombination for radiative capture

    # matrix of e-capture coefficient for multiphonon SRH
    ocp.beta_n_m = np.broadcast_to(tsm.mp_sign, (VB.NVb, egrid.ET.size))
    # matrix of h-capture coefficient for multiphonon SRH
    ocp.beta_p_m = np.broadcast_to(tsm.mp_sigp, (VB.NVb, egrid.ET.size))

    ocp.en_m = (
        ocp.beta_n_m
        * dos.Nc
        * np.exp((egrid.ET - inputs.Ec) * const.Qe / (const.kB * inputs.T))
    )  # electron emission rate
    ocp.ep_m = (
        ocp.beta_p_m
        * dos.Nv
        * np.exp((inputs.Ev - egrid.ET) * const.Qe / (const.kB * inputs.T))
    )  # electron emission rate

    ocp.kn_m = ocp.n * ocp.beta_n_m  # electron capture rate
    ocp.kp_m = ocp.p * ocp.beta_p_m  # hole capture rate

    # occupation probability of defect at any energy E dur to multiphonon transitions
    ocp.f_T_m = ((ocp.n * ocp.beta_n_m) + ocp.ep_m) / (
        (ocp.n * ocp.beta_n_m) + (ocp.p * ocp.beta_p_m) + ocp.en_m + ocp.ep_m
    )  # trap occupation probability

    ocp.eta_R_m_p = (ocp.kp_m * ocp.f_T_m) - (
        ocp.ep_m * (1 - ocp.f_T_m)
    )  # recombination efficiency
    eta_R_m_p = np.array_split(ocp.eta_R_m_p, 2, axis=1)
    ocp.eta_R_m_n = (ocp.n * ocp.beta_n_m * (1 - ocp.f_T_m)) - (
        ocp.en_m * ocp.f_T_m
    )  # recombination efficiency
    eta_R_m_n = np.array_split(ocp.eta_R_m_n, 2, axis=1)
    ocp.eta_R_m = np.concatenate((eta_R_m_n[0], eta_R_m_p[1]), axis=1)
    ocp.R_srh_m = inputs.Nt * ocp.eta_R_m  # rate of recombination

    ocp.f_T_m_1 = 1 - ocp.f_T_m

    # calculation of occupation probability, recombination efficiency and rate of recombination for radiative and multiphonon capture

    ocp.beta_n = ocp.beta_n_r + ocp.beta_n_m  # combined e-capture coefficient
    ocp.beta_p = ocp.beta_p_r + ocp.beta_p_m  # combined h-capture coefficient

    ocp.en = (
        ocp.beta_n
        * dos.Nc
        * np.exp((egrid.ET - inputs.Ec) * const.Qe / (const.kB * inputs.T))
    )  # electron emission rate
    ocp.ep = (
        ocp.beta_p
        * dos.Nv
        * np.exp((inputs.Ev - egrid.ET) * const.Qe / (const.kB * inputs.T))
    )  # electron emission rate

    ocp.kn = ocp.n * ocp.beta_n  # electron capture rate
    ocp.kp = ocp.p * ocp.beta_p  # hole capture rate

    ocp.f_T = ((ocp.n * ocp.beta_n) + ocp.ep) / (
        (ocp.n * ocp.beta_n) + (ocp.p * ocp.beta_p) + ocp.en + ocp.ep
    )  # trap occupation probability

    ocp.eta_R = (ocp.n * ocp.beta_n * (1 - ocp.f_T)) - (
        ocp.en * ocp.f_T
    )  # recombination efficiency

    ocp.R_srh = inputs.Nt * ocp.eta_R  # rate of recombination


def test(data):
    ocp = data.occupation_probability
    test = data.test

    test.kn_m = ocp.kn_m[-1, :]
    test.kp_m = ocp.kp_m[-1, :]
    test.en_m = ocp.en_m[-1, :]
    test.ep_m = ocp.ep_m[-1, :]
    test.f_T_m = ocp.f_T_m[-1, :]
    test.f_T_m_1 = 1 - test.f_T_m
    test.term1 = test.kn_m * test.f_T_m_1
    test.term2 = test.en_m * test.f_T_m
    test.term3 = test.kp_m * test.f_T_m
    test.term4 = test.ep_m * test.f_T_m_1
    test.eta_R_m_op = test.term3 - test.term4
    test.eta_R_m = test.term1 - test.term2
    test.eta_R_m_old = ocp.eta_R_m[-1, :]
    test.eta_R


def eta_R_normalized(data):
    ocp = data.occupation_probability
    eta_nr = data.eta_R_normalized

    eta_nr.eta_R_s_max = np.amax(ocp.eta_R_s, axis=1)
    eta_nr.eta_R_s_norm = (ocp.eta_R_s.T / eta_nr.eta_R_s_max).T

    eta_nr.eta_R_r_max = np.amax(ocp.eta_R_r, axis=1)
    eta_nr.eta_R_r_norm = (ocp.eta_R_r.T / eta_nr.eta_R_r_max).T

    eta_nr.eta_R_m_max = np.amax(ocp.eta_R_m, axis=1)
    eta_nr.eta_R_m_norm = (ocp.eta_R_m.T / eta_nr.eta_R_m_max).T

    eta_nr.eta_R_max = np.amax(ocp.eta_R, axis=1)
    eta_nr.eta_R_norm = (ocp.eta_R.T / eta_nr.eta_R_max).T
