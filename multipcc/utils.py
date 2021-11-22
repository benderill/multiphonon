import math
import numpy as np
import scipy.constants as sc
from addict import Dict

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


"""
Utility and support functions for multipcc.
"""


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
    cf.CF_pos = (2 * sc.pi * np.sqrt(cf.X)) / (1 - np.exp(-2 * sc.pi * cf.X))
    cf.CF_pos = np.broadcast_to(cf.CF_pos, (egrid.ET.size, egrid.Ek.size))
    cf.a = np.exp(-2 * sc.pi * cf.X)
    cf.b = 1 - np.exp(-2 * sc.pi * cf.X)

    cf.CF_neg = (-1 * 2 * sc.pi * np.sqrt(cf.X)) / (1 - np.exp(-2 * sc.pi * -1 * cf.X))
    cf.CF_neg = np.broadcast_to(cf.CF_neg, (egrid.ET.size, egrid.Ek.size))
    cf.c = np.exp(-2 * sc.pi * -1 * cf.X)
    cf.d = 1 - np.exp(-2 * sc.pi * -1 * cf.X)
    cf.unity = np.ones(mat.mat2D.shape)
    cf.CF3D = np.array([cf.unity, cf.unity, cf.unity])

    cf.K = 2 * sc.pi * np.sqrt(derived.r_eh / egrid.Ek)


def w_function(data, arg, axis):
    const = data.constants
    inputs = data.inputs
    mat = data.matrix
    wf = data.weighing_function

    wf.Gc = (
        (8 * np.sqrt(2) * sc.pi / const.hplanck ** 3)
        * np.sqrt(inputs.M_eff) ** 3
        * np.sqrt((mat.Ec - inputs.Eg) * const.eVJ)
    )  # DOS
    wf.fE = np.exp(
        (inputs.Eg / 2 - mat.Ec) * const.eVJ / (const.kB * inputs.T)
    )  # fermidirac fucntion

    # carrier density per volume per energy
    wf.pear = wf.Gc * wf.fE * inputs.dE * const.eVJ

    # function weighted to occupied states
    wf.weighted_numerator = (arg * wf.pear).sum(axis=axis)
    # total number of carrier per volume, after integration over energy
    wf.denominator = wf.pear.sum(axis=0)
    wf.weighted_func = wf.weighted_numerator / wf.denominator  # weighted function
    return wf.weighted_func


def w_function_2(data, arg, axis):
    const = data.constants
    inputs = data.inputs
    mat = data.matrix
    wf2 = data.weighing_function

    wf2.Gc = (
        (8 * np.sqrt(2) * sc.pi / const.hplanck ** 3)
        * np.sqrt(inputs.M_eff) ** 3
        * np.sqrt((mat.Ec - inputs.Eg) * const.eVJ)
    )  # DOS
    wf2.fE = np.exp(
        (inputs.Eg / 2 - mat.Ec) * const.eVJ / (const.kB * inputs.T)
    )  # fermidirac fucntion

    # no of empty states per volume per energy
    wf2.pear = wf2.Gc * (1 - wf2.fE) * inputs.dE * const.eVJ

    # function weighted to unoccupied states
    wf2.weighted_numerator = (arg * wf2.pear).sum(axis=axis)
    # total number of empty states  volume, after integration over energy
    wf2.denominator = wf2.pear.sum(axis=0)
    wf2.weighted_func = wf2.weighted_numerator / wf2.denominator  # weighted function
    return wf2.weighted_func


def h5_load(filename):
    if filename is not None:
        with h5py.File(filename, "r") as h5file:
            root = Dict(recursively_load(h5file, "/"))
        return root
    else:
        print("Filename must be set before load can be used")


def h5_save(filename, data):
    if filename is not None:
        with h5py.File(filename, "w") as h5file:
            recursively_save(h5file, "/", data)
    else:
        print("Filename must be set before save can be used")


def recursively_load(h5file, path):

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load(h5file, path + key + "/")
    return ans


def recursively_save(h5file, path, dic):

    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")
    # save items to the hdf5 file
    for key, item in dic.items():
        key = str(key)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # handle   int, float, string and ndarray of int32, int64, float64
        if isinstance(item, str):
            h5file[path + key] = np.array(item, dtype="S")

        elif isinstance(item, int):
            h5file[path + key] = np.array(item, dtype=np.int32)

        elif isinstance(item, float):
            h5file[path + key] = np.array(item, dtype=np.float64)

        elif isinstance(item, np.ndarray) and item.dtype == np.float64:
            h5file[path + key] = item

        elif isinstance(item, np.ndarray) and item.dtype == np.float32:
            h5file[path + key] = np.array(item, dtype=np.float64)

        elif isinstance(item, np.ndarray) and item.dtype == np.int32:
            h5file[path + key] = item

        elif isinstance(item, np.ndarray) and item.dtype == np.int64:
            h5file[path + key] = item.astype(np.int32)

        elif isinstance(item, np.ndarray) and item.dtype.kind == "S":
            h5file[path + key] = item

        elif isinstance(item, list) and all(isinstance(i, int) for i in item):
            h5file[path + key] = np.array(item, dtype=np.int32)

        elif isinstance(item, list) and any(isinstance(i, float) for i in item):
            h5file[path + key] = np.array(item, dtype=np.float64)

        elif isinstance(item, np.int32):
            h5file[path + key] = item

        elif isinstance(item, np.float64):
            h5file[path + key] = item

        elif isinstance(item, np.float32):
            h5file[path + key] = np.array(item, dtype=np.float64)

        elif isinstance(item, np.bytes_):
            h5file[path + key] = item

        elif isinstance(item, bytes):
            h5file[path + key] = item

        elif isinstance(item, list) and all(isinstance(i, str) for i in item):
            h5file[path + key] = np.array(item, dtype="S")

        # save dictionaries
        elif isinstance(item, dict):
            recursively_save(h5file, path + key + "/", item)
        # other types cannot be saved and will result in an error
        else:
            raise ValueError(
                "Cannot save %s/%s key with %s type." % (path, key, type(item))
            )
