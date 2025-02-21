import numpy as np
from ufl import *
from dune.ufl import Space, Constant
from dune.fem.function import gridFunction
from parabolicTest import model

dimR = 1
time = Constant(0,"time")
sourceTime = Constant(0,"sourceTime")
domain = [0, 0], [1, 1], [10, 10]

space = Space(2,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )


def test1(gridView):
    tauFE = 1.25
    x_val = x[0]

    c = 10
    exact = lambda t: as_vector([1/(c-t)])

    #return model(exact, dtExact, lambda u: as_vector([0])), 8, tauFE, exact(0), exact
    return -dot(dot(u,u), v[0])*dx, 5, tauFE, exact(0), exact, None

def test2(gridView):
    end_time = 0.5
    TauFE = end_time

    c = Constant(1.1, "C")
    exact = lambda t: as_vector([sin(t) + c])

    return -dot(cos(sourceTime)/u[0] * u[0], v[0])*dx, end_time, TauFE, exact(0), exact, None
