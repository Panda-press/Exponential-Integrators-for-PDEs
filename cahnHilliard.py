import numpy as np
from ufl import *
from dune.ufl import Space, Constant
from dune.fem.function import gridFunction
from parabolicTest import model

dimR = 2
time = Constant(0,"time")
sourceTime = Constant(0,"sourceTime")
domain = [0, 0], [1, 1], [60, 60]

space = Space(2,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )


def test2(gridView):
    eps = Constant(0.01,"eps")
    M0 = Constant(0.5, "M0")
    M = M0*(1-u[0])*(1+u[0])
    Fdash = u[0] * (u[0]**2 - 1)
    a = (- M * inner(grad(u[1]), grad(v[0]))
        + (Fdash * v[1] + eps*eps * inner(grad(u[0]), grad(v[1]))))
        
    @gridFunction(gridView,name="random",order=1)
    def u0(x):
        return 1.8*np.random.rand(1)[0] -0.9
    np.random.seed(100)

    return -a * dx, 0.1, 0.001, [u0,0], None, None


