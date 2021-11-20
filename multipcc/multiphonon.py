import numpy
import scipy.constants
import math
import scipy.integrate
from scipy.special import cbrt


def integrand(x, a, b, c):  # integration I(a,b,c)
    return ((x**a)*(numpy.sin(b*numpy.arctan(c*x))**2))/((1+(c*x)**2)**b)


integrand_vec = numpy.vectorize(integrand)


def quad(func, a, b, c):
    return scipy.integrate.quad(func, 0, 1, (a, b, c))


quad_vec = numpy.vectorize(quad)


def derived_parameters(data):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    derived = data.root.derived
    mpder = data.root.multiphonon_derived_parameters

    # radius of sphere with Brillouin zone volume in metre inverse
    mpder.q_D = cbrt(6*scipy.pi**2) / inputs.a_0
    mpder.sa = 4 * math.sqrt(scipy.pi * derived.r_eh *
                             phycon.eVJ / (phycon.kB * inputs.T))  # sommerfiled factor
    mpder.pekar = (1 / inputs.epsilon_h) - \
        (1 / inputs.epsilon_l)  # pekar factor
    mpder.V_0 = (inputs.a_0)**3  # volume of the unit cell in cubic meters
    mpder.omega = (inputs.Eph * phycon.eVJ) / \
        phycon.hbar  # frequecy of the phonon


def Huang_Rhys_Factor_deformation_potential_coupling(data):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    derived = data.root.derived
    mpder = data.root.multiphonon_derived_parameters
    egrid = data.root.energy_grids
    hrfd = data.root.deformation_potential_coupling

    hrfd.SHRD = ((inputs.Dij * phycon.eVJ * 100) / (inputs.Eph * phycon.eVJ))**2 / \
        (2 * inputs.Mr * mpder.omega / phycon.hbar)  # deformation coupling
    print(hrfd.SHRD)
    # array of the three differnt values of mu depending on charge state. The value of mu for
    hrfd.mu = numpy.array([-egrid.nu, egrid.nu*1e-6, egrid.nu])
    # neutral charge state was supposed to be zero but has been given a small value to avoid
    # division by zero error.
    hrfd.a = 0
    hrfd.b = 2 * hrfd.mu
    hrfd.c = (mpder.q_D * derived.a_ebr * egrid.nu) / 2
    hrfd.bcsqre = (hrfd.b * hrfd.c)**2
    # integral part of the Huang Rhys Factor
    hrfd.ans, hrfd.err = quad_vec(integrand_vec, hrfd.a, hrfd.b, hrfd.c)
    hrfd.I = (hrfd.ans / hrfd.bcsqre)
    # final values of Huang Rhys Factor. SHR is an array of size mu x Et. Each coloumn contains the
    hrfd.SHR_D = hrfd.SHRD * hrfd.I
    # values of SHR for every possible value of energy for a particula
    # r charge state


def Huang_Rhys_Factor_polar_coupling(data):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    derived = data.root.derived
    mpder = data.root.multiphonon_derived_parameters
    egrid = data.root.energy_grids
    hrfp = data.root.polar_coupling

    hrfp.SHRP = (3 / (2 * ((inputs.Eph * phycon.eVJ)**2))) * ((phycon.Qe**2) * (inputs.Mr / mpder.V_0)
                                                              * inputs.Eph * phycon.eVJ / (inputs.Mr * (mpder.q_D**2)) * mpder.pekar)  # polar coupling
    print(hrfp.SHRP)
    # array of the three differnt values of mu depending on charge state. The value of mu for
    hrfp.mu = numpy.array([-egrid.nu, egrid.nu*1e-6, egrid.nu])
    # neutral charge state was supposed to be zero but has been given a small value to avoid
    # division by zero error.
    hrfp.a = -2
    hrfp.b = 2 * hrfp.mu
    hrfp.c = (mpder.q_D * derived.a_ebr * egrid.nu) / 2
    hrfp.bcsqre = (hrfp.b * hrfp.c)**2
    # integral part of the Huang Rhys Factor
    hrfp.ans, hrfp.err = quad_vec(integrand_vec, hrfp.a, hrfp.b, hrfp.c)
    hrfp.I = (hrfp.ans / hrfp.bcsqre)
    # final values of Huang Rhys Factor. SHR is an array of size mu x Et. Each coloumn contains the
    hrfp.SHR_P = hrfp.SHRP * hrfp.I
    # values of SHR for every possible value of energy for a particula
    # r charge state


def Huang_Rhys_factor(data):
    inputs = data.root.inputs
    egrid = data.root.energy_grids
    hrfd = data.root.deformation_potential_coupling
    hrfp = data.root.polar_coupling
    hrf = data.root.huang_rhys_factor

    # array of the three differnt values of mu depending on charge state. The value of mu for
    hrf.mu = numpy.array([-egrid.nu, egrid.nu*1e-6, egrid.nu])
    # neutral charge state was supposed to be zero but has been given a small value to avoid
    # division by zero error.

    if inputs.dir == 'dp':
        hrf.SHR = hrfd.SHR_D
    elif inputs.dir == 'pc':
        hrf.SHR = hrfp.SHR_P
    elif inputs.dir == 'com':
        hrf.SHR = hrfd.SHR_D + hrfp.SHR_P
    else:
        print('Please select Multiphonon coupling potential')


def multiphonon_capture_coefficients(data):
    phycon = data.root.physical_constants
    inputs = data.root.inputs
    derived = data.root.derived
    mpder = data.root.multiphonon_derived_parameters
    egrid = data.root.energy_grids
    hrf = data.root.huang_rhys_factor
    mpcoef = data.root.multiphonon_capture_coefficients

    mpcoef.theta = (inputs.Eph * phycon.eVJ) / (2 * phycon.kB * inputs.T)
    # round to next highest integer
    mpcoef.p = numpy.ceil(egrid.ET / inputs.Eph)
    mpcoef.p_vec = numpy.ones(hrf.mu.shape) * \
        mpcoef.p  # matching the shape of mu

    mpcoef.X = numpy.zeros(hrf.SHR.shape)
    mpcoef.X[hrf.SHR < mpcoef.p_vec] = hrf.SHR[hrf.SHR < mpcoef.p_vec] / \
        (mpcoef.p_vec[hrf.SHR < mpcoef.p_vec] * math.sinh(mpcoef.theta))
    mpcoef.X[hrf.SHR > mpcoef.p_vec] = mpcoef.p_vec[hrf.SHR > mpcoef.p_vec] / \
        (hrf.SHR[hrf.SHR > mpcoef.p_vec] * math.sinh(mpcoef.theta))
    mpcoef.sa = numpy.array([mpder.sa, 1, mpder.sa])

    mpcoef.Y = numpy.sqrt(1 + mpcoef.X**2)
    mpcoef.V_T = (4 / 3) * scipy.pi * (derived.a_ebr *
                                       egrid.nu / 2)**3  # volume of the wave function
    mpcoef.k1 = (mpcoef.V_T) * ((mpcoef.p**2) * mpder.omega *
                                math.sqrt(2 * scipy.pi)) / (numpy.sqrt(mpcoef.p * mpcoef.Y))
    mpcoef.k2 = mpcoef.theta + mpcoef.Y - mpcoef.X * \
        math.cosh(mpcoef.theta) - numpy.log((1 + mpcoef.Y) / mpcoef.X)
    # recombination coefficients in m^3/s
    mpcoef.k = mpcoef.k1 * numpy.exp(mpcoef.p * mpcoef.k2)

    mpcoef.capt_cs = mpcoef.k / inputs.v_th  # capture cross section


def trap_state_mp(data):
    inputs = data.root.inputs
    mpder = data.root.multiphonon_derived_parameters
    mpcoef = data.root.multiphonon_capture_coefficients
    tsm = data.root.trap_state_mp

    if inputs.trap_state == 'don':
        # electron capture coefficient from the CB by donar [reversesed for the same reason as radiative]
        tsm.mp_sign = mpder.sa * mpcoef.k[2, :][::-1]
        # hole capture coefficent from VB by donar
        tsm.mp_sigp = mpcoef.k[1, :]

    else:
        # electron capture coefficient from CB by acceptor
        tsm.mp_sign = mpcoef.k[1, :][::-1]
        # hole capture coefficent from VB by acceptor
        tsm.mp_sigp = mpder.sa * mpcoef.k[0, :]


