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
domain = [-width, -width, -depth], [width, width, depth], [8, 8, 8]  

space = Space(3,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )

u_0 = Constant(0, "u_0")
r = sqrt(x[0]**2 + x[1]**2 + x[2]**2) 
#initial = as_vector( [conditional(r>Constant(30), Constant(-1), conditional(abs(x[2]) > Constant(10), Constant(-1), 1)), u_initial] )
initial = as_vector( [1-1/(1+exp(8-r)), u_0])

def test1(gridView):
    tol = Constant(1e-2, "tol")
    maxVal = Constant(10000, "max")
    offset = Constant(1, "offset")

    #Tm = Constant(1, "T_m")
    Tc = Constant(200, "T_c")
    Tm0 = Constant(242, "T_m0")
    #d = Constant(1, "d")
    #D = Constant(1, "D")
    alpha_hat = Constant(0.658, "alpha_hat")
    K_hat = Constant(1.578)
    k0_hat = sqrt(Constant(0.916))
    #Gamma = D/(d**2)
    W = Constant(15.43, "W")
    k = Constant(0.5, "k")
    epsilon = Constant(0.06, "Epsilon")
    epsilon1 = Constant(0.03, "Epsilon1")
    epsilon2 = Constant(0.03, "Epsilon2")

    zeta0 = Constant(0.953, "Zeta0")
    zeta = Constant(0.167, "Zeta")
    #psi_hat = zeta0*(Tm0 - Tm)/(Tm0 - u[1]*(Tm - Tc) + Tc)
    #zeta_hat = (4*zeta0*psi_hat-3*psi_hat**2)/(6*zeta0 - 4*psi_hat)
    #zeta = (1 + 2/pi * atan(k*u[1])) * zeta_hat

    gradu = grad(u[0])

    n = - gradu/(sqrt(dot(gradu,gradu))+tol) 

    flocaldash = W * u[0] * (u[0] - zeta) * (u[0] - zeta0)

    beta = 1 - 3 * epsilon1 + (4 * epsilon1 * (gradu[0]**4 + gradu[1]**4) + (3 * epsilon1 + epsilon2) * gradu[2]**4)/(inner(gradu, gradu)**2 + tol)

    def betadash(du_i, i):
        if (i == 0):
            result = (4 * epsilon1 * (4 * gradu[0]**3 + gradu[1]**4) + (3 * epsilon1 + epsilon2) * gradu[2]**4)/(inner(gradu, gradu)**2 + tol)
        elif(i == 1):
            result = (4 * epsilon1 * (4 * gradu[1]**3 + gradu[0]**4) + (3 * epsilon1 + epsilon2) * gradu[2]**4)/(inner(gradu, gradu)**2 + tol)
        else:
            result = (4 * epsilon1 * (gradu[0]**4 + gradu[1]**4) + (3 * epsilon1 + epsilon2) * 4 * gradu[2]**3)/(inner(gradu, gradu)**2 + tol)
        result += -2 * du_i * 2 * inner(gradu, gradu) * (4 * epsilon1 * (gradu[0]**4 + gradu[1]**4) + (3 * epsilon1 + epsilon2) * gradu[2]**4)/(inner(gradu, gradu)**2 + tol)
        return result
    
    # beta = 1 - 3 * epsilon + 4 * epsilon * (gradu[0]**4 + gradu[1]**4 + gradu[2]**4)/(inner(gradu, gradu)**2 + tol)

    # def betadash(du_i, i):
    #     result = 4 * du_i**3 * 4 * epsilon * (gradu[0]**4 + gradu[1]**4 + gradu[2]**4)/(inner(gradu, gradu)**2 + tol)
    #     result += -2 * du_i * 2 * inner(gradu, gradu) * 4 * epsilon * (gradu[0]**4 + gradu[1]**4 + gradu[2]**4)/(inner(gradu, gradu)**4 + tol)
    #     return result

    beta2gradu2 = as_vector([
        2 * beta * betadash(gradu[0], 0) * inner(gradu, gradu) + 2 * beta ** 2 * gradu[0],
        2 * beta * betadash(gradu[1], 1) * inner(gradu, gradu) + 2 * beta ** 2 * gradu[1],
        2 * beta * betadash(gradu[2], 2) * inner(gradu, gradu) + 2 * beta ** 2 * gradu[2]
    ])

    dpsidt = lambda test: -(flocaldash * test + k0_hat * k0_hat / 2 * inner(grad(test), beta2gradu2))

    dUdt = (alpha_hat * inner(grad(u[1]),grad(v[1])))# + K_hat * dpsidt(v[1]))

    return -(dUdt + dpsidt(v[0])) * dx, 200, 0.1, initial, None, None

    