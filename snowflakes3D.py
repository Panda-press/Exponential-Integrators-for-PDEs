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
domain = [-width, -width, -depth], [width, width, depth], [25, 25, 4]  # Use a grid refinement of at least two when applying the inital condition

space = Space(3,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )

u_initial = Constant(0.7, "u_0")
r = sqrt(x[0]**2 + x[1]**2) 
initial = as_vector( [conditional(r>Constant(30), Constant(-1), conditional(abs(x[2]) > Constant(10), Constant(-1), 1)), u_initial] )
initial = as_vector( [1-2/(1+exp(8-r)), u_initial])

def test1(gridView):
    eps = Constant(1e-5, "eps")

    Gamma = Constant(0.5, "Gamma")
    GammaVector = as_vector([1, 1, Gamma])
    GammaMatrix = as_matrix([[1, 0, 0], [0, 1, 0], [0, 0, Gamma]])
    epsilonxy = Constant(0.1, "Epsilonxy")
    epsilonz = Constant(0.05, "Epsilonz")
    Lambda = Constant(3.0, "Lambda") #Wrong value but isn't numerically unstable unlike 3.0
    D = Constant(0.6267*Lambda.value, "D")
    Lsat = Constant(1.0, "L_sat")
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

    safe_gradPhi0 = max_value(abs(gradPhi[0]), eps) * sign(gradPhi[0])  # Keeps sign but ensures |n0| >= eps
    safe_gradPhi02 = max_value(gradPhi[0] * gradPhi[0], eps)
    safe_gradPhi1 = max_value(abs(gradPhi[1]), eps) * sign(gradPhi[1])
    safe_gradPhi12 = max_value(gradPhi[1] * gradPhi[1], eps)
    safe_gradPhi2 = max_value(abs(gradPhi[2]), eps) * sign(gradPhi[2])  # Same for n2

    bottom1 =  1/(xgp * xgp * xgp + xgp * ygp * ygp)
    safe_bottom1 = conditional(bottom1 < 100, 100, conditional(bottom1 > -100, -100, bottom1))
    safe_zsqrtx2y2 = max_value(abs(safe_gradPhi2 * sqrt(safe_gradPhi0**2 + safe_gradPhi1**2)), eps) * sign(safe_gradPhi2)

    A = 1 + epsilonxy * cos(6 * atan_2(ygp, safe_gradPhi0)) + epsilonz * cos(2 * atan2(sqrt(xgp**2 + ygp**2), safe_gradPhi2))

    dAdGP = as_vector([
        epsilonxy  * (6 * ygp / (xgp * xgp + ygp * ygp + eps)) * (-sin(6 * atan_2(gradPhi[1], safe_gradPhi0)))
        + epsilonz * (2 * xgp * zgp / (inner(gradPhi, gradPhi) * sqrt(xgp * xgp + ygp * ygp) + eps)) * (-sin(2 * atan_2(sqrt(safe_gradPhi0**2 + safe_gradPhi1**2), safe_gradPhi2)))
,
        epsilonxy  * (6 * xgp / (xgp * xgp + ygp * ygp + eps)) * (-sin(6 * atan_2(gradPhi[1], safe_gradPhi0)))
        + epsilonz * (2 * ygp * zgp / (inner(gradPhi, gradPhi) * sqrt(xgp * xgp + ygp * ygp) + eps)) * (-sin(2 * atan_2(sqrt(xgp**2 + ygp**2), safe_gradPhi2)))
,
          epsilonz * (-2 * sqrt(xgp * xgp + ygp * ygp) / (inner(gradPhi, gradPhi) + eps)) * (-sin(2 * atan_2(sqrt(xgp**2 + ygp**2), safe_gradPhi2)))
    ])

#     dA2dGP2 = as_matrix([[
#         epsilonxy * 12 * ygp * (xgp * sin(6 * atan_2(ygp, safe_gradPhi0)) + 3 * ygp * cos(6 * atan_2(y, safe_gradPhi0)))/((xgp**2 + ygp**2)**2 + eps)
        
# ,
#         epsilonxy * 6 * ((xgp**2 - ygp**2) * sin(6*atan_2(y, safe_gradPhi0)) + 6 * xgp * ygp * cos(6 * atan_2(y, safe_gradPhi0)))/((xgp**2 + ygp**2)**2 + eps)

# ,       
#         0

#     ],[

#     ]])
    

    if False:
                
        dtPhi = 1/(A*A)*(-fDash + Lambda * b * gDash * u_
                        +0.5*div(GammaMatrix * (inner(gradPhi, gradPhi) * dAdGP + A*A*GammaMatrix*gradPhi)))

        dtPhi = (1/(A*A))*(-fDash + Lambda * gDash * u_
                        +0.5 * (2 * dot(div(GammaMatrix * gradPhi) * gradPhi, 2 * A * dAdGP)
                                + inner(gradPhi, gradPhi) * 2 * dot(grad(GammaMatrix * gradPhi) * dAdGP, dAdGP)
                                + A * div(GammaMatrix * dAdGP)
                                + 2 * A * inner((grad(GammaMatrix * gradPhi) * dAdGP), (GammaMatrix * gradPhi))
                                + A*A * div(GammaMatrix * GammaMatrix * gradPhi))
                                )

        dtU = (D * div(GammaMatrix * q * GammaMatrix * grad(u_)) - 0.5 * Lsat * b * dtPhi)
        
        form = (dtPhi * v[0] + dtU * v[1])*dx

    else:
        dtPhi = lambda v_:  (v_/(A*A) * (-fDash + Lambda * gDash * u_)
                - dot(GammaMatrix * (grad(v_)/(2*A) - v_*(GammaMatrix * grad(gradPhi))*dAdGP/(A*A)), inner(gradPhi, gradPhi) * 2 * dAdGP + A * GammaMatrix * gradPhi)
                )


        dtU = (D * div(GammaMatrix * q * GammaMatrix * grad(u_)) * v[1] - 0.5 * Lsat * b * dtPhi(v[1]))
        
        form = (dtPhi(v[0]) + dtU) * dx

    return -form, 50, 0.01, initial, None, [None]

    