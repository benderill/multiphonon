import numpy
import scipy.constants
from scipy.special import cbrt


def biasVoltage(data):
    inputs = data.root.inputs
    VB = data.root.bias_voltage

    VB.Vb = numpy.arange(inputs.V_start, inputs.V_stop + inputs.dV, inputs.dV)
    VB.NVb = len(VB.Vb)
