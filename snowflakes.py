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
domain = [4, 4], [8, 8], [3, 3]  # Use a grid refinement of at least two when applying the inital condition

space = Space(2,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )


r = sqrt( dot( x-as_vector([6,6]), x-as_vector([6,6])) )
initial = as_vector( [conditional(r>0.3,0,1), -0.5] )


alpha        = 0.015
tau          = 3e-4
kappa1       = 0.9
kappa2       = 20.
c            = 0.02
N            = 6.

dt = Constant(0.0005, "dt")


psi        = pi/8.0 + atan2(grad(u[0])[1], (grad(u[0])[0]))
Phi        = tan(N / 2.0 * psi)
beta       = (1.0 - Phi*Phi) / (1.0 + Phi*Phi)
dbeta_dPhi = -2.0 * N * Phi / (1.0 + Phi*Phi)
fac        = 1.0 + c * beta
diag       = fac
offdiag    = -c * dbeta_dPhi
d0         = alpha*alpha*fac*as_vector([diag, offdiag])
d1         = alpha*alpha*fac*as_vector([-offdiag, diag])
m          = u[0] - 0.5 - kappa1 / pi*atan(kappa2*u[1])
a_im = (1 / tau * (-inner(dot(d0, grad(u[0])),grad(v[0])[0]) 
                   - inner(dot(d1, grad(u[0])), grad(v[0])[1]) 
                   + v[0]*u[0]*(1-u[0])*m)
        - 2.25 * inner(grad(u[1]), grad(v[1])) 
        - 1 / tau * (inner(dot(d0, grad(u[0])),grad(v[1])[0]) + inner(dot(d1, grad(u[0])), grad(v[1])[1]))
        + 1 / tau * v[1]*u[0]*(1-u[0])*m ) * dx

dtheta = (grad(dot(d0, grad(u[0])))[0] + grad(dot(d1, grad(u[0])))[1] + u[0]*(1-u[0])*m)

a_alt = (1/tau * (dtheta
                  )*v[0]
        +(2.25 * (div(grad(u[1])))
          +1/tau *dtheta
                  )*v[1]
        ) * dx


def test1(gridView):
    return -a_im, 0.1, 5e-4, initial, None, [None]

