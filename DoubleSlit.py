import numpy as np
from ufl import *
from dune.ufl import Space, Constant, DirichletBC
from dune.fem.function import gridFunction
from parabolicTest import model
from dune.grid import reader

dimR = 2
time = Constant(0,"time")
sourceTime = Constant(0,"sourceTime")
domain = (reader.gmsh, "wave_tank.msh")

space = Space(2,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )

g = 1/(10 * pi) * sin(10*pi*(sourceTime))

def test2(gridView):
    psi = u[0]
    p = u[1]
    form = (v[0] * (-p) + inner(grad(psi), grad(v[1]))) * dx
    bc   = DirichletBC(space, [None,g], x[0] < 1e-10)
    initial = [conditional(x[0]<0.3, 0.07*(-cos(10*pi*x[0])-1), 0), conditional(x[0]<0.3, 0.07*(-sin(10*pi*x[0])-1), 0)]
    #initial = [0,0]
    return [-form,bc], 3, 0.001, initial, None, None


