import numpy
import scipy

def w_function(data, arg, axis):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    mat = data.root.matrix
    wf = data.root.weighing_function

    wf.Gc = (8 * scipy.sqrt(2) * scipy.pi / phycon.hplanck**3) * \
        scipy.sqrt(inputs.M_eff)**3 * \
        numpy.sqrt((mat.Ec - inputs.Eg) * phycon.eVJ)  # DOS
    wf.fE = numpy.exp((inputs.Eg/2 - mat.Ec) * phycon.eVJ /
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
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    mat = data.root.matrix
    wf2 = data.root.weighing_function

    wf2.Gc = (8 * scipy.sqrt(2) * scipy.pi / phycon.hplanck**3) * \
        scipy.sqrt(inputs.M_eff)**3 * \
        numpy.sqrt((mat.Ec - inputs.Eg) * phycon.eVJ)  # DOS
    wf2.fE = numpy.exp((inputs.Eg/2 - mat.Ec) * phycon.eVJ /
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

