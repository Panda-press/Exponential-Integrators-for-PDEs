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
width = 200
depth = 25
domain = [-width, -width, -depth], [width, width, depth], [8, 8, 1]  

space = Space(3,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )

u_initial = Constant(0.5, "u_0")
r = sqrt(x[0]**2 + x[1]**2 + x[2]**2) 
#initial = as_vector( [conditional(r>Constant(30), Constant(-1), conditional(abs(x[2]) > Constant(10), Constant(-1), 1)), u_initial] )
initial = as_vector( [1-2/(1+exp(8-r)), u_initial])

def test1(gridView):
    eps = Constant(1e-2, "eps")
    maxVal = Constant(1000, "max")

    Gamma = Constant(0.5, "Gamma")
    epsilonz = Constant(0.3, "Epsilonz")
    Lsat = Constant(1.0, "L_sat")

    GammaVector = as_vector([1, 1, Gamma])
    GammaMatrix = as_matrix([[1, 0, 0], [0, 1, 0], [0, 0, Gamma]])
    epsilonxy = Constant(0.2, "Epsilonxy")
    Lambda = Constant(3.0, "Lambda") #Wrong value but isn't numerically unstable unlike 3.0
    D = Constant(0.6267*Lambda.value, "D")
    phi = u[0]
    u_ = u[1]

    gradPhi = grad(phi)
    #gradPhi = variable(gradPhi)

    n = - gradPhi/(sqrt(dot(gradPhi,gradPhi))+eps) # Added Small Value to prevent numerical error, feel free to change
    #n = variable(n)

    fDash = - phi + phi * phi * phi
    b = sqrt(dot(GammaMatrix * n, GammaMatrix * n))
    gDash = (1-phi*phi)*(1-phi*phi)
    xgp = gradPhi[0]
    ygp = gradPhi[1]
    zgp = gradPhi[2]

    q = 1 - phi

    safe_gradPhi0 = conditional(xgp>0, xgp + eps, xgp - eps)  # Keeps sign but ensures |n0| >= eps
    safe_gradPhi1 = conditional(ygp>0, ygp + eps, ygp - eps)
    safe_gradPhi2 = conditional(zgp>0, zgp + eps, zgp - eps)  # Same for n2
    safe_bottom = conditional(xgp**3 + xgp*ygp**2>0, xgp**3 + xgp*ygp**2 + eps, xgp**3 + xgp*ygp**2 - eps)

    A = 1 + epsilonxy * cos(6 * atan_2(ygp, safe_gradPhi0)) + epsilonz * cos(2 * atan_2(sqrt(xgp**2 + ygp**2), safe_gradPhi2))

    massA = lambda u: (1 + epsilonxy * cos(6*atan_2(grad(u[0])[1], conditional(abs(grad(u[0])[0])>0, grad(u[0])[0] + eps, grad(u[0])[0] - eps)))
                        + epsilonz * cos(2 * atan_2(sqrt(grad(u[0])[0]**2 + grad(u[0])[1]**2 + eps), conditional(abs(grad(u[0])[2])>0, grad(u[0])[2] + eps, grad(u[0])[2] - eps))))**2
    #There is no reason for this epsilon in the sqrt but if I don't have it the code doesn't work


    dAdGP = as_vector([
        epsilonxy  * (6 * ygp * inner(gradPhi, gradPhi) / ((xgp * xgp + ygp * ygp) + eps)) * (-sin(6 * atan_2(gradPhi[1], safe_gradPhi0)))
        + epsilonz * (2 * xgp * inner(gradPhi, gradPhi) * zgp / (inner(gradPhi, gradPhi) * sqrt(xgp * xgp + ygp * ygp) + eps)) * (-sin(2 * atan_2(sqrt(xgp**2 + ygp**2), safe_gradPhi2)))
,
        epsilonxy  * (6 * xgp * inner(gradPhi, gradPhi) / (xgp * xgp + ygp * ygp + eps)) * (-sin(pi/6 + 6 * atan_2(gradPhi[0], safe_gradPhi1)))
        #epsilonxy  * (6 * ygp * ygp / (safe_bottom)) * (-sin(6 * atan_2(gradPhi[1], safe_gradPhi0)))
        + epsilonz * (2 * ygp * zgp * inner(gradPhi, gradPhi) / (inner(gradPhi, gradPhi) * sqrt(xgp * xgp + ygp * ygp) + eps)) * (-sin(2 * atan_2(sqrt(xgp**2 + ygp**2), safe_gradPhi2)))
,
         epsilonz * (-2 * sqrt(xgp * xgp + ygp * ygp) * inner(gradPhi, gradPhi) / (inner(gradPhi, gradPhi) + eps)) * (-sin(2 * atan_2(sqrt(xgp**2 + ygp**2), safe_gradPhi2)))
    ])  

    if True:

        AAdtPhi = lambda v_:  (v_ * (Constant(1) * -fDash + Constant(1) * Lambda * gDash * u_)
                - Constant(1) * dot(GammaMatrix * grad(v_), inner(gradPhi, gradPhi) * 2 * A * dAdGP + A * A * GammaMatrix * gradPhi)
                )


        AAdtU = (- A * A * D * dot(q * GammaMatrix * grad(u_), GammaMatrix * grad(v[1])) - 0.5 * Lsat * b * AAdtPhi(v[1]))
        
        form = (AAdtPhi(v[0]) + AAdtU) * dx

    else:
        dtPhi = lambda v_:  (v_/(A*A) * (-fDash + Lambda * gDash * u_)
                - dot(GammaMatrix * (grad(v_)/(2*A) - v_*(GammaMatrix * grad(gradPhi))*dAdGP/(A*A)), 2 * dAdGP + A * GammaMatrix * gradPhi)
                )


        dtU = (D * div(GammaMatrix * q * GammaMatrix * grad(u_)) * v[1] - 0.5 * Lsat * b * dtPhi(v[1]))
        
        form = (dtPhi(v[0]) + dtU) * dx

        massA = None

    return -form, 0.1, 0.01, initial, None, massA

    