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
width = 128
domain = [-0, -0, -0], [width, width, width], [4, 4, 4]  
#domain = [-width, -width, -depth], [width, width, depth], [3, 3, 3]  

space = Space(3,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )


def test1(gridView):
    tol = Constant(1e-5, "tol")


    DL = Constant(1/12, "DL")
    muinf = Constant(0.04, "muinf")


    cL = Constant(0.9, "c_l")
    cS = Constant(0.5, "c_s")
    mu0 = Constant(1, "mu")
    a = Constant(4, "a")
    DS = 1e-4 * DL
    Deltac = cL - cS
    Rc = Constant(10, "Rc")
    R0 = Constant(20, "R0")
    delta = Constant(2, "delta")
    epsilon = Constant(2e-2, "epsilon")
    cBar = (mu0 - muinf + a * cS)/a
    alpha = (cBar - cL)/(cS - cL)
    Lambda = 3 * Rc * Deltac**2/delta


    r = sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    #r = abs(x[0]) + abs(x[1]) + abs(x[2])
    initial0 = 1/(1+exp(-(r-R0)/delta))
    initial1 = mu0 - initial0*a*(cBar - cS)
    initial = [initial0, initial1]


    D = u[0]*DL + (1-u[0])*DS

    gradu = grad(u[0])
    
    if True:
        def Ai(var):
            return sqrt(var**2 + epsilon**2*(gradu[0]**2 + gradu[1]**2 + gradu[2]**2))

        A = Ai(gradu[0]) + Ai(gradu[1]) + Ai(gradu[2])

        dAdGP = as_vector([
            gradu[0]*((1+epsilon**2)/(Ai(gradu[0])+tol) + epsilon**2/(Ai(gradu[1])+tol) + epsilon**2/(Ai(gradu[2])+tol)),
            gradu[1]*((1+epsilon**2)/(Ai(gradu[1])+tol) + epsilon**2/(Ai(gradu[0])+tol) + epsilon**2/(Ai(gradu[2])+tol)),
            gradu[2]*((1+epsilon**2)/(Ai(gradu[2])+tol) + epsilon**2/(Ai(gradu[0])+tol) + epsilon**2/(Ai(gradu[1])+tol))
        ])
    else:
        A0 = 1-3*epsilon
        epsilonbar = 4*epsilon/(1-3*epsilon)
        phimag = sqrt(inner(gradu, gradu)) + tol
        normal = gradu / phimag

        A = A0*(1 + epsilonbar*(normal[0]**4 + normal[1]**4 + normal[2]**4))

        gradux = gradu[0]
        graduy = gradu[1]
        graduz = gradu[2]

        dAdGP = A0 * epsilonbar * as_vector([
            4 * gradux * (gradux**2 * (graduy**2 + graduz**2) - graduy**4 - graduz**4)/((gradux**2 + graduy**2 + graduz**2)**3+tol),
            4 * graduy * (graduy**2 * (gradux**2 + graduz**2) - gradux**4 - graduz**4)/((gradux**2 + graduy**2 + graduz**2)**3+tol),
            4 * graduz * (graduz**2 * (gradux**2 + graduy**2) - gradux**4 - graduy**4)/((gradux**2 + graduy**2 + graduz**2)**3+tol)

        ])    

    omegaDash = u[0]*(1-u[0])*(1-2*u[0])

    gDash = 6*u[0] - 6*u[0]**2

    dthedt = lambda test: (-inner(A * dAdGP, grad(test)) - test * omegaDash/(delta**2) - test * gDash*(mu0-u[1])*Deltac/(Lambda*delta**2))

    dmudt = (- a * D * inner(grad(v[1]), grad(u[1])) - a * Deltac * gDash * dthedt(v[1]))

    return -(dmudt + dthedt(v[0])) * dx, 10, 0.01, initial, None, None

    
