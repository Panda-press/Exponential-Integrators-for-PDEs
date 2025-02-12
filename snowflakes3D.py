import numpy as np
import ufl
from ufl import *
try:
    from ufl import atan2
except ImportError: # remain compatible with version 2022 of ufl
    from ufl import atan_2 as atan2
from dune.ufl import Space, Constant
from dune.fem.function import gridFunction
from parabolicTest import model

dimR = 2
time = Constant(0,"time")
sourceTime = Constant(0,"sourceTime")
domain = [-4, -4, -4], [4, 4, 4], [4, 4, 4]  # Use a grid refinement of at least two when applying the inital condition

space = Space(2,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )


r = sqrt( dot( x-as_vector([6,6]), x-as_vector([6,6])) )
initial = as_vector( [conditional(r>0.3,0,1), -0.5] )

def test1(gridView):
    Gamma = Constant(0.5, "Gamma")
    GammaVector = as_vector([1, 1, Gamma])
    epsilonxy = Constant(0.1, "Epsilonxy")
    epsilonz = Constant(0.05, "Epsilonz")
    Lambda = Constant(3.0, "Lambda")
    D = Constant(0.6267*Lambda.value, "D")
    Lsat = Constant(1.0, "L_sat")
    phi = u[0]
    u_ = u[1]

    n = - grad(phi)/norm(grad(phi))
    theta = atan(n[1]/n[0])
    psi = atan(sqrt(n[0] * n[0] + n[1] * n[1])/n[2])

    fDash = - phi + phi * phi * phi
    b = sqrt(dot(n * GammaVector, n * GammaVector))
    gDash = (1-phi)*(1-phi)
    A = 1 + epsilonxy * cos(6 * theta) + epsilonz * cos(2 * psi)
    q = 1 - phi

    dtPhiv0 = (-inner(GammaVector * grad(phi), GammaVector * grad(v[0]))
              + 1/(A*A)*(Lambda * B * gDash - fDash
                       + 0.5 * GammaVector * grad(inner(grad(phi), grad(phi)) * differentiate(A*A, grad(phi)))
                       )*v[0]) * dx

    dtPhiv1 = (-inner(GammaVector * grad(phi), GammaVector * grad(v[1]))
               + 1/(A*A)*(Lambda * B * gDash - fDash
                       + 0.5 * GammaVector * grad(inner(grad(phi), grad(phi)) * differentiate(A*A, grad(phi)))
                       )*v[1]) * dx

    dtUv = (D * GammaVector * grad(q * GammaVector * grad(u))) * v[1] - 0.5 * Lsat * B * dtPhiv1

    return dtPhiv0 + dtUv, 0.1, 5e-4, initial, None, [None]

    