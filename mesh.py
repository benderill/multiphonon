import numpy
import math
import scipy.constants

def energy_grids(data):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    derived = data.root.derived
    egrid = data.root.energy_grids

    
    egrid.ET = numpy.linspace(4*derived.r_eh, inputs.Eg - 4*derived.r_eh, 1000)       #Distance of the defect from the band(the energy grids are so chosen to get the right nu)
    egrid.deltaE = numpy.minimum(egrid.ET, inputs.Eg - egrid.ET)                      #Distance of the defect from the nearest band
    egrid.nu = numpy.sqrt(derived.r_eh / egrid.deltaE)                                #nu    
    egrid.Ekmax = 3 * phycon.kB * inputs.T / phycon.eVJ                               #the maximum final energy is 3KbT away
    egrid.Ek = numpy.arange(inputs.dE, egrid.Ekmax, inputs.dE)                        #the final energy mesh.
    egrid.k = scipy.sqrt(egrid.Ek * phycon.eVJ * 2 * inputs.M_eff / phycon.hbar**2)   #the corresponding k values
    
    


def matrices(data):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    derived = data.root.derived
    egrid = data.root.energy_grids
    dos = data.root.effective_density_of_states
    VB = data.root.bias_voltage
    mat = data.root.matrix

    mat.mat2D = numpy.zeros([egrid.ET.size, egrid.Ek.size])                                 #xy         

    mat.ET2D = numpy.broadcast_to(egrid.ET , (egrid.Ek.size, egrid.ET.size)).T              #2D matrix of trap energy [ET x Ek] 
    mat.Ek2D = numpy.broadcast_to(egrid.Ek , (egrid.ET.size, egrid.Ek.size))                #2D matrix of final energy state [ET x Ek] 
    mat.Ek3D = numpy.broadcast_to(egrid.Ek , (3, egrid.ET.size, egrid.Ek.size))             #3D matrix of final energy state [3 x ET x Ek]
    
    mat.Ec = egrid.Ek + inputs.Eg                                           
    mat.Ev = -egrid.Ek
     
    mat.nu2D = numpy.broadcast_to(egrid.nu , (egrid.Ek.size, egrid.ET.size)).T              #2D matrix of nu [ET x Ek] 
    mat.nu3D = numpy.tile(mat.nu2D,(3,1)).reshape(3, egrid.ET.size, egrid.Ek.size)          #3D matrix of nu [3 x ET x Ek]
        

if __name__ == "__main__":
    data = h5.H5()
    data.filename = "C:/Users/basit/Simulations_and_results/Data/Paper1/SRH_deep_defects/Peros_Case_3.h5"
    with open('output.txt', 'w+') as output:
        main(data, input_peros, output)