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
width = 512
depth = 512
domain = [-0, -0, -0], [width, width, depth], [4, 4, 4]  

space = Space(3,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )


def test1(gridView):
    tol = Constant(1e-2, "tol")
    maxVal = Constant(10000, "max")
    offset = Constant(1, "offset")

    DL = Constant(1/12, "DL")
    muinf = Constant(0.04, "muinf")

    cL = Constant(0.9, "c_l")
    cS = Constant(0.5, "c_s")
    mu0 = Constant(1, "mu")
    a = Constant(4, "a")
    DS = 1e-4 * DL
    Rc = Constant(10, "Rc")
    R0 = Constant(20, "R0")
    delta = Constant(2, "delta")
    epsilon = Constant(0.02, "epsilon")
    alpha = 1 + (muinf - mu0)/(a*(cL-cS))
    cBar = alpha*cS + (1-alpha)*cL
    #a = (mu0 - muinf)/(cBar - cS)

    
    u_0 = Constant(0, "u_0")
    r = sqrt(x[0]**2 + x[1]**2 + x[2]**2) 
    initial0 = 1/(1+exp((r-R0)/delta))
    initial1 = mu0 - initial0*a*(cBar - cS)
    initial = [initial0, initial1]


    D = u[0]*DL + (1-u[0])*DS
    Deltac = cL - cS
    Lambda = 3 * Rc * Deltac**2/delta
    Lambda = Deltac**2

    gradu = grad(u[0])

    def Ai(var):
        return sqrt(var**2 + epsilon**2*(gradu[0]**2 + gradu[1]**2 + gradu[2]**2))

    A = Ai(gradu[0]) + Ai(gradu[1]) + Ai(gradu[2])

    dAdGP = as_vector([
        gradu[0]*((epsilon**2+1)/Ai(gradu[0]) + epsilon**2/Ai(gradu[1]) + epsilon**2/Ai(gradu[2])),
        gradu[1]*((epsilon**2+1)/Ai(gradu[1]) + epsilon**2/Ai(gradu[0]) + epsilon**2/Ai(gradu[2])),
        gradu[2]*((epsilon**2+1)/Ai(gradu[2]) + epsilon**2/Ai(gradu[0]) + epsilon**2/Ai(gradu[1]))
    ])

    omegaDash = u[0]*(1-2*u[0]**3)

    gDash = 6*u[0] - 6*u[0]**2

    dthedt = lambda test: (inner(grad(test), A * dAdGP) - test * omegaDash/(delta**2) - test * gDash*(mu0-u[1])*Deltac/(Lambda*delta**2))

    dmudt = (a * D * inner(grad(v[1]), grad(u[1])) - a * Deltac * gDash * dthedt(v[1]))

    return (dmudt + dthedt(v[0])) * dx, 100, 0.1, initial, None, None

    