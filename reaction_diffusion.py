import numpy as np
from ufl import *
from dune.ufl import Space, Constant
from dune.fem.function import gridFunction
from parabolicTest import model

dimR = 1
time = Constant(0,"time")
sourceTime = Constant(0,"sourceTime")
domain = [0, 0], [2, 1], [1024, 64]

space = Space(2,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )



"""Traveling Wave"""
def test1(gridView, epsilon=0.05):
    end_time = 3*1.414*epsilon/5
    tauFE = end_time / 32
    x_val = x[0]

    epsilon = Constant(epsilon,"epsilon")
    s = Constant(3/(sqrt(2) * epsilon))
    exact = lambda t: as_vector([0.5 * (1 - tanh((x[0] - s * sourceTime)/(2*sqrt(2)*epsilon)))])
    #dtExact = lambda t: -(1/2 - alpha) * as_vector([exact(t)]) * (as_vector([1]) - exact(t)) 

    boundary = lambda t: dot(dot(grad(exact(t)[0]),n),v[0])*ds
    potential = 1/(epsilon*epsilon) * dot(u[0] - 1, u[0] + 1) * dot(u[0], v[0]) * dx
    a = inner(grad(u),grad(v))*dx

    #return model(exact, dtExact, lambda u: as_vector([0])), 8, tauFE, exact(0), exact
    return a + potential - boundary(time), end_time, tauFE, exact(0), exact

