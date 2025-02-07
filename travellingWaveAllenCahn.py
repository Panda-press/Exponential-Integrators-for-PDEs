import numpy as np
from ufl import *
from dune.ufl import Space, Constant, DirichletBC
from dune.fem.function import gridFunction

dimR = 1
time = Constant(0,"time")
sourceTime = Constant(0,"sourceTime")
domain = [-8, -1], [8, 1], [128, 8]

space = Space(2,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )


"""Traveling Wave"""
def test3(gridView, alpha=0.25):
    tauFE = 1
    x_val = x[0]

    exact = lambda t: as_vector([exp((-x_val/sqrt(2)+(0.5-alpha)*t))/(1+exp((-x_val/sqrt(2)+(0.5-alpha)*t)))])
    #dtExact = lambda t: -(1/2 - alpha) * as_vector([exact(t)]) * (as_vector([1]) - exact(t)) 
          
    rboundary = DirichletBC(space, [exact(sourceTime)[0]], x[0] > 8 - 1e-100)
    lboundary = DirichletBC(space, [exact(sourceTime)[0]], x[0] < -8 + 1e-100)
    potential = dot(u[0]-alpha,1-u[0]) * dot(u,v) * dx
    a = inner(grad(u),grad(v))*dx

    #return model(exact, dtExact, lambda u: as_vector([0])), 8, tauFE, exact(0), exact
    return a - potential, 8, tauFE, exact(0), exact, [rboundary, lboundary]

