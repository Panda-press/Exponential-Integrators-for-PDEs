import numpy as np
from ufl import *
from dune.ufl import Space, Constant
from dune.fem.function import gridFunction

dimR = 1
time = Constant(0,"time")
sourceTime = Constant(0,"sourceTime")
domain = [-4, -1], [8, 1], [120, 10]

space = Space(2,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )


"""Traveling Wave"""
def test3(gridView, alpha=0.25):
    tauFE = 1
    x_val = x[0]

    exact = lambda t: as_vector([exp((-x_val/sqrt(2)+(0.5-alpha)*t))/(1+exp((-x_val/sqrt(2)+(0.5-alpha)*t)))])
    #dtExact = lambda t: -(1/2 - alpha) * as_vector([exact(t)]) * (as_vector([1]) - exact(t)) 

    boundary = lambda t: dot(dot(grad(exact(t)[0]),n),v[0])*ds
    potential = dot(u[0]-alpha,1-u[0]) * dot(u,v) * dx
    a = inner(grad(u),grad(v))*dx

    #return model(exact, dtExact, lambda u: as_vector([0])), 8, tauFE, exact(0), exact
    return a - potential - boundary(sourceTime), 16, tauFE, exact(0), exact, None

