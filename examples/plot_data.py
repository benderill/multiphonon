import matplotlib.pyplot as plt
import numpy as np

from multipcc.initialize import Initialize


def plot_nu(data):
    egrid = data.energy_grids
    hrf = data.huang_rhys_factor
    plt.xticks(np.arange(0, egrid.ET[-1], 0.1))
    plt.title("nu")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")
    plt.ylabel("nu")

    plt.plot(egrid.ET, egrid.nu, label=r"$\nu$")
    plt.legend()
    plt.show()


def plot_HRF(data):
    egrid = data.energy_grids
    hrf = data.huang_rhys_factor
    plt.xticks(np.arange(0, egrid.ET[-1], 0.1))
    plt.title(r"Huang Rhys Factor $S_{HR}$")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")
    plt.ylabel(r"Huang Rhys Factor $S_{HR}$")

    plt.plot(egrid.ET, hrf.SHR[0, :], label="negative")
    plt.plot(egrid.ET, hrf.SHR[1, :], label="neutral")
    plt.plot(egrid.ET, hrf.SHR[2, :], label="positive")
    plt.legend()
    plt.show()


def plot_radiative_capture_coefficients(data):
    tsr = data.trap_state_rc
    egrid = data.energy_grids

    plt.title("Radiative capture coefficients")
    plt.xticks(np.arange(0, egrid.ET[-1], 0.1))
    plt.ylabel(r"Capture coefficients $k_{n/p}$ [$cm^3/s$]")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")
    plt.semilogy(egrid.ET, tsr.r_sign * 1e6, label=r"$k_n$")
    plt.semilogy(egrid.ET, tsr.r_sigp * 1e6, label=r"$k_p$")
    plt.legend()
    plt.show()


def plot_multiphonon_capture_coefficients(data):
    tsm = data.trap_state_mp
    egrid = data.energy_grids

    # plt.ylim(1e-18,1e-3)
    plt.xticks(np.arange(0, egrid.ET[-1], 0.1))
    plt.title("Multiphonon capture coefficinets")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")
    plt.ylabel(r"Capture coefficients $k_{n/p}$ [$cm^3/s$]")
    plt.semilogy(egrid.ET, tsm.mp_sign * 1e6, label=r"$k_n$")
    plt.semilogy(egrid.ET, tsm.mp_sigp * 1e6, label=r"$k_p$")
    plt.legend()
    plt.show()


def extrafactor(data):
    rcapt = data.radiative_capture_cross_section
    egrid = data.energy_grids

    plt.title("Extra factor")
    plt.plot(egrid.ET, rcapt.extrafactor[0, :], label="neg")
    plt.plot(egrid.ET, rcapt.extrafactor[1, :], label="neu")
    plt.plot(egrid.ET, rcapt.extrafactor[2, :], label="pos")
    plt.legend()
    plt.show()


def twinx_rad(data):
    egrid = data.energy_grids
    ocp = data.occupation_probability
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # RADIATIVE CAPTURE

    # plt.title("Capture rate, emission rate and occupation probability of donar trap for radiative capture",loc='center')
    plt.title("RADIATIVE: nkn, pkp, en, ep fT and eta_R for VA =0.5V", loc="center")
    ax1.semilogy(
        egrid.ET, ocp.kn_r[5, :], label=r"$nkn$", color="#1f77b4", linestyle="-"
    )
    ax1.semilogy(
        egrid.ET, ocp.kp_r[5, :], label=r"$pkp$", color="#ff7f0e", linestyle="-"
    )
    ax1.semilogy(
        egrid.ET,
        ocp.eta_R_r[5, :],
        label=r"$\eta_{Rr}$",
        color="#296E01",
        linestyle="-",
    )
    ax1.semilogy(
        egrid.ET, ocp.en_r[5, :], label=r"$en$", color="#1f77b4", linestyle="--"
    )
    ax1.semilogy(
        egrid.ET, ocp.ep_r[5, :], label="$ep$", color="#ff7f0e", linestyle="--"
    )
    ax1.set_xlabel(r"Defect energy level $E_T$ [eV]")
    ax1.set_ylabel(r"Rate $[1/s]$")
    ax1.set_ylim(1e-14, 1e4)
    ax1.xaxis.set_ticks(np.arange(0, egrid.ET[-1], 0.1))
    for label in ax1.get_yticklabels():
        label.set_color("black")

    ax2 = ax1.twinx()
    plt.plot(
        egrid.ET, ocp.f_T_r[5, :], label=r"$f_{T}$", color="#000000", linestyle="-"
    )
    plt.plot(egrid.ET, 0.5 * np.ones(len(egrid.ET)), color="#000000", linestyle=":")
    ax2.set_xlabel(r"Defect energy level $E_T$ [eV]")
    ax2.set_ylabel(r"Occupation probability $f_T$")
    for label in ax2.get_yticklabels():
        label.set_color("black")
    ax1.legend(loc=2)  # upper left corner
    ax2.legend(loc=7)  # upper left corner
    plt.legend()
    plt.show()


def twinx_mp(data):
    egrid = data.energy_grids
    ocp = data.occupation_probability
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # MULTIPHONON CAPTURE

    # plt.title("Capture rate, emission rate and occupation probability of donar trap for multiphonon capture",loc='center')
    plt.title("MULTIPHONON: nkn, pkp, en, ep fT and eta_R for VA =0.5V", loc="center")
    ax1.semilogy(
        egrid.ET, ocp.kn_m[5, :], label=r"$nkn$", color="#1f77b4", linestyle="-"
    )
    ax1.semilogy(
        egrid.ET, ocp.kp_m[5, :], label=r"$pkp$", color="#ff7f0e", linestyle="-"
    )
    ax1.semilogy(
        egrid.ET, ocp.eta_R_m[5, :], label=r"$\eta_{R}$", color="#296E01", linestyle="-"
    )
    ax1.semilogy(
        egrid.ET, ocp.en_m[5, :], label=r"$en$", color="#1f77b4", linestyle="--"
    )
    ax1.semilogy(
        egrid.ET, ocp.ep_m[5, :], label="$ep$", color="#ff7f0e", linestyle="--"
    )
    ax1.set_xlabel("Trap energy [eV]")
    ax1.set_ylabel(r"Rate $[1/s]$")
    ax1.set_ylim(1e-7, 1e14)
    ax1.xaxis.set_ticks(np.arange(0, egrid.ET[-1], 0.1))
    for label in ax1.get_yticklabels():
        label.set_color("black")

    ax2 = ax1.twinx()
    plt.plot(
        egrid.ET, ocp.f_T_m[5, :], label=r"$f_{T}$", color="#000000", linestyle="-"
    )
    plt.plot(egrid.ET, 0.5 * np.ones(len(egrid.ET)), color="#000000", linestyle=":")
    ax2.set_xlabel(r"Defect energy level $E_T$ [eV]")
    ax2.set_ylabel(r"Occupation probability $f_{T}$")
    for label in ax2.get_yticklabels():
        label.set_color("black")
    ax1.legend(loc=2)  # upper left corner
    ax2.legend(loc=7)  # upper left corner
    plt.legend()
    plt.show()


def twinx_combined(data):
    egrid = data.energy_grids
    ocp = data.occupation_probability
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # COMBINED CAPTURE

    # plt.title("Capture rate, emission rate and occupation probability of donar trap for combined capture",loc='center')
    plt.title("COMBINED: nkn, pkp, en, ep fT and eta_R for VA =0.5V", loc="center")
    ax1.semilogy(egrid.ET, ocp.kn[5, :], label=r"$nkn$", color="#1f77b4", linestyle="-")
    ax1.semilogy(egrid.ET, ocp.kp[5, :], label=r"$pkp$", color="#ff7f0e", linestyle="-")
    ax1.semilogy(
        egrid.ET, ocp.eta_R[5, :], label=r"$\eta_R$", color="#296E01", linestyle="-"
    )
    ax1.semilogy(
        egrid.ET,
        ocp.eta_R_s[5, :],
        label=r"$\eta_{Rc}$",
        color="#8B0000",
        linestyle="-",
    )
    ax1.semilogy(egrid.ET, ocp.en[5, :], label=r"$en$", color="#1f77b4", linestyle="--")
    ax1.semilogy(egrid.ET, ocp.ep[5, :], label="$ep$", color="#ff7f0e", linestyle="--")
    ax1.set_xlabel(r"Defect energy level $E_T$ [eV]")
    ax1.set_ylabel(r"Rate $[1/s]$")
    ax1.set_ylim(1e-7, 1e14)
    ax1.xaxis.set_ticks(np.arange(0, egrid.ET[-1], 0.1))
    for label in ax1.get_yticklabels():
        label.set_color("black")

    ax2 = ax1.twinx()
    plt.plot(egrid.ET, ocp.f_T[5, :], label=r"$f_{T}$", color="#000000", linestyle="-")
    plt.plot(egrid.ET, 0.5 * np.ones(len(egrid.ET)), color="#000000", linestyle=":")
    ax2.set_xlabel(r"Defect energy level $E_T$ [eV]")
    ax2.set_ylabel(r"Occupation probability $f_T$")
    for label in ax2.get_yticklabels():
        label.set_color("black")
    ax1.legend(loc=2)  # upper left corner
    ax2.legend(loc=7)  # upper left corner
    plt.legend()
    plt.show()


def twinx_constant(data):
    egrid = data.energy_grids
    ocp = data.occupation_probability
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # CONSTANT CAPTURE COEFFICIENT

    # plt.title("Capture rate, emission rate and occupation probability of donar trap for constant capture coefficient",loc='center')
    plt.title("CONSTANT: nkn, pkp, en, ep fT and eta_R for VA =0.5V", loc="center")
    ax1.semilogy(
        egrid.ET, ocp.kn_s[10, :], label=r"$nkn$", color="#1f77b4", linestyle="-"
    )
    ax1.semilogy(
        egrid.ET, ocp.kp_s[10, :], label=r"$nkp$", color="#ff7f0e", linestyle="-"
    )
    ax1.semilogy(
        egrid.ET,
        ocp.eta_R_s[10, :],
        label=r"$\eta_{R}$",
        color="#296E01",
        linestyle="-",
    )
    ax1.semilogy(
        egrid.ET, ocp.en_s[10, :], label=r"$en$", color="#1f77b4", linestyle="--"
    )
    ax1.semilogy(
        egrid.ET, ocp.ep_s[10, :], label="$ep$", color="#ff7f0e", linestyle="--"
    )
    ax1.set_xlabel(r"Defect energy level $E_T$ [eV]")
    ax1.set_ylabel(r"Rate $[1/s]$")
    ax1.set_ylim(1e-7, 1e9)
    ax1.xaxis.set_ticks(np.arange(0, egrid.ET[-1], 0.1))
    for label in ax1.get_yticklabels():
        label.set_color("black")

    ax2 = ax1.twinx()
    plt.plot(
        egrid.ET, ocp.f_T_s[10, :], label=r"$f_{T}$", color="#000000", linestyle="-"
    )
    plt.plot(egrid.ET, 0.5 * np.ones(len(egrid.ET)), color="#000000", linestyle=":")
    ax2.set_xlabel(r"Defect energy level $E_T$ [eV]")
    ax2.set_ylabel(r"Occupation probability $f_Tc$")
    for label in ax2.get_yticklabels():
        label.set_color("black")
    ax1.legend(loc=2)  # upper left corner
    ax2.legend(loc=7)  # upper left corner
    plt.legend()
    plt.show()


def plot_occupation_probability(data):
    egrid = data.energy_grids
    ocp = data.occupation_probability

    plt.title("Trap occupation probability due to constant capture coefficient")
    plt.ylabel(r"Occupation probability $f_T$")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")
    plt.xticks(np.arange(0, egrid.ET[-1], 0.1))

    plt.plot(egrid.ET, ocp.f_T_s[0, :], label="0.0V")
    plt.plot(egrid.ET, ocp.f_T_s[1, :], label="0.1V")
    plt.plot(egrid.ET, ocp.f_T_s[2, :], label="0.2V")
    plt.plot(egrid.ET, ocp.f_T_s[3, :], label="0.3V")
    plt.plot(egrid.ET, ocp.f_T_s[4, :], label="0.4V")
    plt.plot(egrid.ET, ocp.f_T_s[5, :], label="0.5V")
    plt.plot(egrid.ET, ocp.f_T_s[6, :], label="0.6V")
    plt.plot(egrid.ET, ocp.f_T_s[7, :], label="0.7V")
    plt.plot(egrid.ET, ocp.f_T_s[8, :], label="0.8V")
    plt.plot(egrid.ET, ocp.f_T_s[9, :], label="0.9V")
    plt.plot(egrid.ET, ocp.f_T_s[10, :], label="1.0V")
    plt.legend()
    plt.show()

    plt.title("Trap occupation probability due to radiative capture")
    plt.ylabel(r"Occupation probability $f_T$")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")
    plt.xticks(np.arange(0, egrid.ET[-1], 0.1))
    plt.plot(egrid.ET, ocp.f_T_r[0, :], label="0.0V")
    plt.plot(egrid.ET, ocp.f_T_r[1, :], label="0.1V")
    plt.plot(egrid.ET, ocp.f_T_r[2, :], label="0.2V")
    plt.plot(egrid.ET, ocp.f_T_r[3, :], label="0.3V")
    plt.plot(egrid.ET, ocp.f_T_r[4, :], label="0.4V")
    plt.plot(egrid.ET, ocp.f_T_r[5, :], label="0.5V")
    plt.plot(egrid.ET, ocp.f_T_r[6, :], label="0.6V")
    plt.plot(egrid.ET, ocp.f_T_r[7, :], label="0.7V")
    plt.plot(egrid.ET, ocp.f_T_r[8, :], label="0.8V")
    plt.plot(egrid.ET, ocp.f_T_r[9, :], label="0.9V")
    plt.plot(egrid.ET, ocp.f_T_r[10, :], label="1.0V")
    plt.legend()
    plt.show()

    plt.title("Trap occupation probability due to multiphonon capture")
    plt.ylabel(r"Occupation probability $f_T$")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")
    plt.xticks(np.arange(0, egrid.ET[-1], 0.1))
    plt.plot(egrid.ET, ocp.f_T_m[0, :], label="0.0V")
    plt.plot(egrid.ET, ocp.f_T_m[1, :], label="0.1V")
    plt.plot(egrid.ET, ocp.f_T_m[2, :], label="0.2V")
    plt.plot(egrid.ET, ocp.f_T_m[3, :], label="0.3V")
    plt.plot(egrid.ET, ocp.f_T_m[4, :], label="0.4V")
    plt.plot(egrid.ET, ocp.f_T_m[5, :], label="0.5V")
    plt.plot(egrid.ET, ocp.f_T_m[6, :], label="0.6V")
    plt.plot(egrid.ET, ocp.f_T_m[7, :], label="0.7V")
    plt.plot(egrid.ET, ocp.f_T_m[8, :], label="0.8V")
    plt.plot(egrid.ET, ocp.f_T_m[9, :], label="0.9V")
    plt.plot(egrid.ET, ocp.f_T_m[10, :], label="1.0V")
    plt.legend()
    plt.show()

    plt.title("Trap occupation probability due to combined capture")
    plt.ylabel(r"Occupation probability $f_T$")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")
    plt.xticks(np.arange(0, egrid.ET[-1], 0.1))
    plt.plot(egrid.ET, ocp.f_T[0, :], label="0.0V")
    plt.plot(egrid.ET, ocp.f_T[1, :], label="0.1V")
    plt.plot(egrid.ET, ocp.f_T[2, :], label="0.2V")
    plt.plot(egrid.ET, ocp.f_T[3, :], label="0.3V")
    plt.plot(egrid.ET, ocp.f_T[4, :], label="0.4V")
    plt.plot(egrid.ET, ocp.f_T[5, :], label="0.5V")
    plt.plot(egrid.ET, ocp.f_T[6, :], label="0.6V")
    plt.plot(egrid.ET, ocp.f_T[7, :], label="0.7V")
    plt.plot(egrid.ET, ocp.f_T[8, :], label="0.8V")
    plt.plot(egrid.ET, ocp.f_T[9, :], label="0.9V")
    plt.plot(egrid.ET, ocp.f_T[10, :], label="1.0V")
    plt.legend()
    plt.show()


def plot_recombination_efficiency(data):
    egrid = data.energy_grids
    ocp = data.occupation_probability

    plt.title("Recombination effciency due to constant capture coefficient")
    plt.ylabel(r"recombination efficiency $\eta_R$ $[1/s]$")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")

    plt.xticks(np.arange(0, egrid.ET[-1], 0.1))
    # plt.semilogy(egrid.ET, ocp.eta_R_s[0,:],label='0.0V')
    plt.semilogy(egrid.ET, ocp.eta_R_s[1, :], label="0.1V")
    plt.semilogy(egrid.ET, ocp.eta_R_s[2, :], label="0.2V")
    plt.semilogy(egrid.ET, ocp.eta_R_s[3, :], label="0.3V")
    plt.semilogy(egrid.ET, ocp.eta_R_s[4, :], label="0.4V")
    plt.semilogy(egrid.ET, ocp.eta_R_s[5, :], label="0.5V")
    plt.semilogy(egrid.ET, ocp.eta_R_s[6, :], label="0.6V")
    plt.semilogy(egrid.ET, ocp.eta_R_s[7, :], label="0.7V")
    plt.semilogy(egrid.ET, ocp.eta_R_s[8, :], label="0.7V")
    plt.semilogy(egrid.ET, ocp.eta_R_s[9, :], label="0.9V")
    plt.semilogy(egrid.ET, ocp.eta_R_s[10, :], label="1.0V")
    plt.legend()
    plt.show()

    plt.title("Recombination effciency due to radaitive transitions")
    plt.ylabel(r"recombination efficiency $\eta_R$ $[1/s]$")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")
    plt.xticks(np.arange(0, egrid.ET[-1], 0.1))

    # plt.semilogy(egrid.ET, ocp.eta_R_r[0,:],label='0.0Vr')
    plt.semilogy(egrid.ET, ocp.eta_R_r[1, :], label="0.1V")
    plt.semilogy(egrid.ET, ocp.eta_R_r[2, :], label="0.2V")
    plt.semilogy(egrid.ET, ocp.eta_R_r[3, :], label="0.3V")
    plt.semilogy(egrid.ET, ocp.eta_R_r[4, :], label="0.4V")
    plt.semilogy(egrid.ET, ocp.eta_R_r[5, :], label="0.4V")
    plt.semilogy(egrid.ET, ocp.eta_R_r[6, :], label="0.6V")
    plt.semilogy(egrid.ET, ocp.eta_R_r[7, :], label="0.7V")
    plt.semilogy(egrid.ET, ocp.eta_R_r[8, :], label="0.8V")
    plt.semilogy(egrid.ET, ocp.eta_R_r[9, :], label="0.9V")
    plt.semilogy(egrid.ET, ocp.eta_R_r[10, :], label="1.0V")
    plt.legend()
    plt.show()

    plt.title("Recombination effciency due to multiphonon capture")
    plt.ylabel(r"recombination efficiency $\eta_R$ $[1/s]$")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")
    plt.xticks(np.arange(0, egrid.ET[-1], 0.1))
    # plt.ylim(1e-8,1e7)
    # plt.semilogy(egrid.ET, ocp.eta_R_m[0,:],label='0.0Vm')
    plt.semilogy(egrid.ET, ocp.eta_R_m[1, :], label="0.1V")
    plt.semilogy(egrid.ET, ocp.eta_R_m[2, :], label="0.2V")
    plt.semilogy(egrid.ET, ocp.eta_R_m[3, :], label="0.3V")
    plt.semilogy(egrid.ET, ocp.eta_R_m[4, :], label="0.4V")
    plt.semilogy(egrid.ET, ocp.eta_R_m[5, :], label="0.5V")
    plt.semilogy(egrid.ET, ocp.eta_R_m[6, :], label="0.6V")
    plt.semilogy(egrid.ET, ocp.eta_R_m[7, :], label="0.7V")
    plt.semilogy(egrid.ET, ocp.eta_R_m[8, :], label="0.8V")
    plt.semilogy(egrid.ET, ocp.eta_R_m[9, :], label="0.9V")
    plt.semilogy(egrid.ET, ocp.eta_R_m[10, :], label="1.0V")
    plt.legend()
    plt.show()

    plt.title("Recombination efficiency due to combined capture")
    plt.ylabel(r"recombination efficiency $\eta_R$ $[1/s]$")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")
    plt.xticks(np.arange(0, egrid.ET[-1], 0.1))
    # plt.ylim(1e-7,1e7)
    # plt.semilogy(egrid.ET, ocp.eta_R[0,:],label='0.0Vc')
    plt.semilogy(egrid.ET, ocp.eta_R[1, :], label="0.1V")
    plt.semilogy(egrid.ET, ocp.eta_R[2, :], label="0.2V")
    plt.semilogy(egrid.ET, ocp.eta_R[3, :], label="0.3V")
    plt.semilogy(egrid.ET, ocp.eta_R[4, :], label="0.4V")
    plt.semilogy(egrid.ET, ocp.eta_R[5, :], label="0.5V")
    plt.semilogy(egrid.ET, ocp.eta_R[6, :], label="0.6V")
    plt.semilogy(egrid.ET, ocp.eta_R[7, :], label="0.7V")
    plt.semilogy(egrid.ET, ocp.eta_R[8, :], label="0.8V")
    plt.semilogy(egrid.ET, ocp.eta_R[9, :], label="0.9V")
    plt.semilogy(egrid.ET, ocp.eta_R[10, :], label="1.0V")
    plt.legend()
    plt.show()


def plot_rate_SRH(data):
    egrid = data.energy_grids
    ocp = data.occupation_probability

    plt.title("Rate of SRH recombination due to radiative capture")
    plt.ylabel(r"Rate of recombination $R_{SRH}$ $[cm^{-3}s^{-1}]$")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")

    # plt.semilogy(egrid.ET, ocp.R_srh_r[0,:],label='0.0V')
    plt.semilogy(egrid.ET, ocp.R_srh_r[1, :], label="0.1V")
    plt.semilogy(egrid.ET, ocp.R_srh_r[2, :], label="0.2V")
    plt.semilogy(egrid.ET, ocp.R_srh_r[3, :], label="0.3V")
    plt.semilogy(egrid.ET, ocp.R_srh_r[4, :], label="0.4V")
    plt.semilogy(egrid.ET, ocp.R_srh_r[5, :], label="0.5V")
    plt.semilogy(egrid.ET, ocp.R_srh_r[6, :], label="0.6V")
    plt.semilogy(egrid.ET, ocp.R_srh_r[7, :], label="0.7V")
    plt.semilogy(egrid.ET, ocp.R_srh_r[8, :], label="0.8V")
    plt.semilogy(egrid.ET, ocp.R_srh_r[9, :], label="0.9V")
    plt.semilogy(egrid.ET, ocp.R_srh_r[10, :], label="1.0V")
    plt.legend()
    plt.show()

    plt.title("Rate of SRH recombination due to multiphonon capture")
    plt.ylabel(r"Rate of recombination $R_{SRH}$ $[cm^{-3}s^{-1}]$")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")
    plt.ylim(1e7, 1e22)
    # plt.semilogy(egrid.ET, ocp.R_srh_m[0,:],label='0.0V')
    plt.semilogy(egrid.ET, ocp.R_srh_m[1, :], label="0.1V")
    plt.semilogy(egrid.ET, ocp.R_srh_m[2, :], label="0.2V")
    plt.semilogy(egrid.ET, ocp.R_srh_m[3, :], label="0.3V")
    plt.semilogy(egrid.ET, ocp.R_srh_m[4, :], label="0.4V")
    plt.semilogy(egrid.ET, ocp.R_srh_m[5, :], label="0.5V")
    plt.semilogy(egrid.ET, ocp.R_srh_m[6, :], label="0.6V")
    plt.semilogy(egrid.ET, ocp.R_srh_m[7, :], label="0.7V")
    plt.semilogy(egrid.ET, ocp.R_srh_m[8, :], label="0.7V")
    plt.semilogy(egrid.ET, ocp.R_srh_m[9, :], label="0.9V")
    plt.semilogy(egrid.ET, ocp.R_srh_m[10, :], label="1.0V")
    plt.legend()
    plt.show()

    plt.title("Rate of SRH recombination due to combined capture")
    plt.ylabel(r"Rate of recombination $R_{SRH}$ $[cm^{-3}s^{-1}$")
    plt.xlabel(r"Defect energy level $E_T$ [eV]")
    # plt.ylim(1e7,1e22)
    # plt.semilogy(egrid.ET, ocp.R_srh[0,:],label='0.0V')
    plt.semilogy(egrid.ET, ocp.R_srh[1, :], label="0.1V")
    plt.semilogy(egrid.ET, ocp.R_srh[2, :], label="0.2V")
    plt.semilogy(egrid.ET, ocp.R_srh[3, :], label="0.3V")
    plt.semilogy(egrid.ET, ocp.R_srh[4, :], label="0.4V")
    plt.semilogy(egrid.ET, ocp.R_srh[5, :], label="0.5V")
    plt.semilogy(egrid.ET, ocp.R_srh[6, :], label="0.6V")
    plt.semilogy(egrid.ET, ocp.R_srh[7, :], label="0.7V")
    plt.semilogy(egrid.ET, ocp.R_srh[8, :], label="0.8V")
    plt.semilogy(egrid.ET, ocp.R_srh[9, :], label="0.9V")
    plt.semilogy(egrid.ET, ocp.R_srh[10, :], label="1.0V")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = Initialize()
    data.filename = "test.h5"
    data.load()
    twinx_mp(data)

    """plot_nu(data)
    plot_HRF(data)
    plot_capture_coefficients(data)
    plot_recombination_coefficients(data)"""
    """twinx_rad(data)
    twinx_mp(data)
    twinx_combined(data)"""
    # twinx_constant(data)
    plot_occupation_probability(data)
    plot_recombination_efficiency(data)
    plot_rate_SRH(data)
