import numpy
import scipy.constants
from scipy.special import cbrt


def biasVoltage(data):
    inputs = data.root.inputs
    VB = data.root.bias_voltage
    
    VB.Vb = numpy.arange(inputs.V_start, inputs.V_stop + inputs.dV , inputs.dV)
    VB.NVb = len(VB.Vb)


if __name__ == "__main__":
    data = h5.H5()
    data.filename = "C:/Users/basit/Simulations_and_results/Data/Paper1/SRH_deep_defects/Peros_Case_3.h5"
    with open('output.txt', 'w+') as output:
        main(data, input_peros, output)
