import numpy as np
from ufl import *
from dune.ufl import Space, Constant, DirichletBC
from dune.fem.function import gridFunction
from parabolicTest import model

dimR = 1
time = Constant(0,"time")
sourceTime = Constant(0,"sourceTime")
domain = [0.5, 0], [2.5, 1], [16, 16]


space = Space(2,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )

"""Reaction Diffusion"""
def test1(gridView):
    end_time = 1
    tauFE = end_time / 8
    x_val = x[0]
    y_val = x[1]

    def boundary_indicator(x):
        return x[0] < domain[0][0] or x[0] > domain[1][0] or x[1] < domain[0][1] or x[1] > domain[1][1]

    exact = lambda t: as_vector([exp(-pi * pi * t)*(sin(pi * x_val) - 1)*sin(pi * y_val)])
    #dtExact = lambda t: -(1/2 - alpha) * as_vector([exact(t)]) * (as_vector([1]) - exact(t)) 

    b = 0.5 * (-pi * pi * u[0] + pi * pi * exp(-pi * pi * sourceTime) * sin(pi * x_val) * sin(pi * y_val)) * v[0] * dx
    a = 0.5 * inner(grad(u),grad(v))*dx

    #return model(exact, dtExact, lambda u: as_vector([0])), 8, tauFE, exact(0), exact
    return a - b, end_time, tauFE, exact(0), exact, DirichletBC(space, [0.0])


"""Traveling Wave"""
def test2(gridView, epsilon=0.5):
    end_time = 3*1.414*epsilon/5
    tauFE = end_time / 32
    x_val = x[0]

    epsilon = Constant(epsilon,"epsilon")
    s = Constant(3/(sqrt(2) * epsilon))
    exact = lambda t: as_vector([0.5 * (1 - tanh((x_val - s * t)/(2*sqrt(2)*epsilon)))])
    #dtExact = lambda t: -(1/2 - alpha) * as_vector([exact(t)]) * (as_vector([1]) - exact(t)) 

    boundary = lambda t: dot(dot(grad(exact(t)[0]),n),v[0])*ds
    potential = 1/(epsilon*epsilon) * dot(u[0] - 1, u[0] + 1) * dot(u[0], v[0]) * dx
    a = inner(grad(u),grad(v))*dx

    #return model(exact, dtExact, lambda u: as_vector([0])), 8, tauFE, exact(0), exact
    return a + potential - boundary(sourceTime), end_time, tauFE, exact(0), exact

