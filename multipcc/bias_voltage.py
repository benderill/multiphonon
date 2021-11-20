import numpy

def bias_voltage(data):
    inputs = data.root.inputs
    VB = data.root.bias_voltage

    VB.Vb = numpy.arange(inputs.V_start, inputs.V_stop + inputs.dV, inputs.dV)
    VB.NVb = len(VB.Vb)

