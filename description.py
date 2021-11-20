import numpy
import scipy


def desciption(data):
    description = data.root.description

    description.current_simulation = 'Case 3: Directly calculate the radiative capture cross section by putting the photon density of states as 1/(2pi^2)((hbar omega)^2/(hbar c/eta_R)^3).'


if __name__ == "__main__":
    data = h5.H5()
    data.filename = "C:/Users/basit/Simulations_and_results/Data/Paper1/SRH_deep_defects/Peros_Case_3.h5"
    with open('output.txt', 'w+') as output:
        main(data, input_peros, output)
