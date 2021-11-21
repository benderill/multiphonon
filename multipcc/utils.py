import numpy as np
import scipy.constants as sc
from addict import Dict

'''
Utility and support functions for multipcc.
'''

def bias_voltage(start, stop, step):
    bias_volt = Dict()
    bias_volt.Vb = np.arange(start, stop + step, step)
    bias_volt.NVb = len(bias_volt.Vb)
    return bias_volt


def coulomb_factor(data):
    derived = data.derived
    egrid = data.energy_grids
    mat = data.matrix
    cf = data.coulomb_factor

    cf.CF_pos = np.zeros(mat.mat2D.shape)
    cf.CF_neg = np.zeros(mat.mat2D.shape)
    cf.X = np.sqrt(derived.r_eh / egrid.Ek)
    cf.CF_pos = (2 * sc.pi * np.sqrt(cf.X)) / \
        (1 - np.exp(-2 * sc.pi * cf.X))
    cf.CF_pos = np.broadcast_to(cf.CF_pos, (egrid.ET.size, egrid.Ek.size))
    cf.a = np.exp(-2 * sc.pi * cf.X)
    cf.b = (1 - np.exp(-2 * sc.pi * cf.X))

    cf.CF_neg = (-1 * 2 * sc.pi * np.sqrt(cf.X)) / \
        (1 - np.exp(-2 * sc.pi * -1 * cf.X))
    cf.CF_neg = np.broadcast_to(cf.CF_neg, (egrid.ET.size, egrid.Ek.size))
    cf.c = np.exp(-2 * sc.pi * -1 * cf.X)
    cf.d = (1 - np.exp(-2 * sc.pi * -1 * cf.X))
    cf.unity = np.ones(mat.mat2D.shape)
    cf.CF3D = np.array([cf.unity, cf.unity, cf.unity])

    cf.K = 2 * sc.pi * np.sqrt(derived.r_eh / egrid.Ek)


def w_function(data, arg, axis):
    phycon = data.physical_constants
    inputs = data.inputs
    mat = data.matrix
    wf = data.weighing_function

    wf.Gc = (8 * np.sqrt(2) * sc.pi / phycon.hplanck**3) * \
        np.sqrt(inputs.M_eff)**3 * \
        np.sqrt((mat.Ec - inputs.Eg) * phycon.eVJ)  # DOS
    wf.fE = np.exp((inputs.Eg/2 - mat.Ec) * phycon.eVJ /
                   (phycon.kB * inputs.T))  # fermidirac fucntion

    # carrier density per volume per energy
    wf.pear = wf.Gc * wf.fE * inputs.dE * phycon.eVJ

    # function weighted to occupied states
    wf.weighted_numerator = (arg * wf.pear).sum(axis=axis)
    # total number of carrier per volume, after integration over energy
    wf.denominator = wf.pear.sum(axis=0)
    wf.weighted_func = wf.weighted_numerator / wf.denominator  # weighted function
    return(wf.weighted_func)


def w_function_2(data, arg, axis):
    phycon = data.physical_constants
    inputs = data.inputs
    mat = data.matrix
    wf2 = data.weighing_function

    wf2.Gc = (8 * np.sqrt(2) * sc.pi / phycon.hplanck**3) * \
        np.sqrt(inputs.M_eff)**3 * \
        np.sqrt((mat.Ec - inputs.Eg) * phycon.eVJ)  # DOS
    wf2.fE = np.exp((inputs.Eg/2 - mat.Ec) * phycon.eVJ /
                    (phycon.kB * inputs.T))  # fermidirac fucntion

    # no of empty states per volume per energy
    wf2.pear = wf2.Gc * (1 - wf2.fE) * inputs.dE * phycon.eVJ

    # function weighted to unoccupied states
    wf2.weighted_numerator = (arg * wf2.pear).sum(axis=axis)
    # total number of empty states  volume, after integration over energy
    wf2.denominator = wf2.pear.sum(axis=0)
    wf2.weighted_func = wf2.weighted_numerator / \
        wf2.denominator  # weighted function
    return(wf2.weighted_func)
