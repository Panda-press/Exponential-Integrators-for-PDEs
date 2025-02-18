import numpy as np
from ufl import *
from dune.ufl import Space, Constant, DirichletBC
from dune.fem.function import gridFunction

dimR = 1
time = Constant(0,"time")
sourceTime = Constant(0,"sourceTime")
rhx = -0.5
lhx = 2.5
domain = [rhx, 0], [lhx, 1/8], [128, 8]

space = Space(2,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )


"""Traveling Wave"""
def test3(gridView, epsilon=0.05):
    end_time = 3*1.414*epsilon/5
    tauFE = end_time / 16
    x_val = x[0]

    epsilon = Constant(epsilon,"epsilon")
    s = Constant(3/(sqrt(2) * epsilon))
    exact = lambda t: as_vector([0.5 * (1 - tanh((x_val - s * t)/(2*sqrt(2)*epsilon)))])
    #dtExact = lambda t: -(1/2 - alpha) * as_vector([exact(t)]) * (as_vector([1]) - exact(t)) 


    rboundary = DirichletBC(space, [exact(sourceTime)[0]], x[0] < rhx + 1e-100)
    lboundary = DirichletBC(space, [exact(sourceTime)[0]], x[0] > lhx - 1e-100)

    potential = 1/(epsilon*epsilon) * dot(u[0] - 1, u[0] + 1) * dot(u[0], v[0]) * dx
    a = inner(grad(u),grad(v))*dx

    #return model(exact, dtExact, lambda u: as_vector([0])), 8, tauFE, exact(0), exact
    return a + potential, end_time, tauFE, exact(0), exact, [rboundary, lboundary]

