import numpy
import scipy.constants


def coulomb_factor(data):
    phycon = data.root.physical_constants
    derived = data.root.derived
    egrid = data.root.energy_grids
    cf = data.root.coulomb_factor
    mat = data.root.matrix

    cf.CF_pos = numpy.zeros(mat.mat2D.shape)
    cf.CF_neg = numpy.zeros(mat.mat2D.shape)
    cf.X = numpy.sqrt(derived.r_eh / egrid.Ek)
    cf.CF_pos = (2 * scipy.pi * numpy.sqrt(cf.X)) / \
        (1 - numpy.exp(-2 * scipy.pi * cf.X))
    cf.CF_pos = numpy.broadcast_to(cf.CF_pos, (egrid.ET.size, egrid.Ek.size))
    cf.a = numpy.exp(-2 * scipy.pi * cf.X)
    cf.b = (1 - numpy.exp(-2 * scipy.pi * cf.X))

    cf.CF_neg = (-1 * 2 * scipy.pi * numpy.sqrt(cf.X)) / \
        (1 - numpy.exp(-2 * scipy.pi * -1 * cf.X))
    cf.CF_neg = numpy.broadcast_to(cf.CF_neg, (egrid.ET.size, egrid.Ek.size))
    cf.c = numpy.exp(-2 * scipy.pi * -1 * cf.X)
    cf.d = (1 - numpy.exp(-2 * scipy.pi * -1 * cf.X))
    cf.unity = numpy.ones(mat.mat2D.shape)
    cf.CF3D = numpy.array([cf.unity, cf.unity, cf.unity])

    cf.K = 2 * scipy.pi * numpy.sqrt(derived.r_eh / egrid.Ek)



