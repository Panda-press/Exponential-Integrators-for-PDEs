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
domain = [-160, -160, -25], [160, 160, 25], [25, 25, 4]  # Use a grid refinement of at least two when applying the inital condition

space = Space(3,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )


r = sqrt(x[0]**2 + x[1]**2) 
initial = as_vector( [conditional(r>Constant(10), -1, conditional(abs(x[2]) > Constant(5), -1, 1)), Constant(0.5)] )

def test1(gridView):
    eps = Constant(1e-10, "eps")

    Gamma = Constant(0.5, "Gamma")
    GammaVector = as_vector([1, 1, Gamma])
    GammaMatrix = as_matrix([[1, 0, 0], [0, 1, 0], [0, 0, Gamma]])
    epsilonxy = Constant(0.1, "Epsilonxy")
    epsilonz = Constant(0.05, "Epsilonz")
    Lambda = Constant(3.0, "Lambda")
    D = Constant(0.6267*Lambda.value, "D")
    Lsat = Constant(1.0, "L_sat")
    phi = u[0]
    u_ = u[1]

    gradPhi = grad(phi)
    #gradPhi = variable(gradPhi)

    n = - gradPhi/(sqrt(dot(gradPhi,gradPhi))+1e-3) # Added Small Value to prevent numerical error, feel free to change
    #n = variable(n)

    fDash = - phi + phi * phi * phi
    b = sqrt(dot(GammaMatrix * n, GammaMatrix * n))
    gDash = (1-phi)*(1-phi)

    xgp = gradPhi[0]
    ygp = gradPhi[1]
    zgp = gradPhi[2]

    A = 1 + epsilonxy * cos(6 * atan(n[1]/(conditional(n[0] > 0, n[0]+eps, n[0]-eps)))) + epsilonz * cos(2 * atan(sqrt(n[0] * n[0] + n[1] * n[1])/(conditional(n[2] > 0, n[2]+eps, n[2]-eps))))
    q = 1 - phi


    safe_gradPhi0 = max_value(abs(gradPhi[0]), eps) * sign(gradPhi[0])  # Keeps sign but ensures |n0| >= eps
    safe_gradPhi02 = max_value(gradPhi[0] * gradPhi[0], eps)
    safe_gradPhi1 = max_value(abs(gradPhi[1]), eps) * sign(gradPhi[1])
    safe_gradPhi12 = max_value(gradPhi[1] * gradPhi[1], eps)
    safe_gradPhi2 = max_value(abs(gradPhi[2]), eps) * sign(gradPhi[2])  # Same for n2

    safe_x3xy2 = max_value(abs(xgp **3 + xgp * ygp **2), eps) * sign(xgp **3 + xgp * ygp **2)
    safe_zsqrtx2y2 = max_value(abs(safe_gradPhi2 * sqrt(safe_gradPhi0**2 + safe_gradPhi1**2)), eps) * sign(safe_gradPhi2)

    A = 1 + epsilonxy * cos(6 * atan_2(ygp, safe_gradPhi0)) + epsilonz * cos(2 * atan2(sqrt(xgp**2 + ygp**2), safe_gradPhi2))

    dA2 = as_vector([
        epsilonxy  * (-gradPhi[1] / (safe_gradPhi02)) * (6 / (1 + (gradPhi[1]**2 / safe_gradPhi02))) * (-sin(6 * atan2(gradPhi[1], safe_gradPhi0)))
        + epsilonz * (2 * gradPhi[0]) * (0.5 / (safe_zsqrtx2y2)) * (2 / (1 + (safe_gradPhi0**2 + safe_gradPhi1**2) / (safe_gradPhi2**2))) * (-sin(2 * atan(sqrt(safe_gradPhi0**2 + safe_gradPhi1**2) / safe_gradPhi2)))
,
        epsilonxy * (1 / (safe_gradPhi0)) * (6 / (1 + (gradPhi[1]**2 / safe_gradPhi02))) * (-sin(6 * atan2(gradPhi[1], safe_gradPhi0)))
        + epsilonz * (2 * gradPhi[1]) * (0.5 / (safe_gradPhi2 * sqrt(safe_gradPhi0**2 + safe_gradPhi1**2))) * (2 / (1 + (safe_gradPhi0**2 + safe_gradPhi1**2) / (safe_gradPhi2**2))) * (-sin(2 * atan(sqrt(safe_gradPhi0**2 + n[1]**2) / safe_gradPhi2)))
,
        epsilonz * (-sqrt(gradPhi[0]**2 + gradPhi[1]**2) / (safe_gradPhi2**2)) * (2 / (1 + (gradPhi[0]**2 + gradPhi[1]**2) / (safe_gradPhi2**2))) * (-sin(2 * atan(sqrt(gradPhi[0]**2 + gradPhi[1]**2) / safe_gradPhi2)))
    ])    

    dA2 = as_vector([
        epsilonxy  * (6 * ygp / (xgp * xgp + ygp * ygp + eps)) * (-sin(6 * atan_2(gradPhi[1], safe_gradPhi0)))
        + epsilonz * (2 * xgp * zgp / (inner(gradPhi, gradPhi) * sqrt(xgp * xgp + ygp * ygp) + eps)) * (-sin(2 * atan_2(sqrt(safe_gradPhi0**2 + safe_gradPhi1**2), safe_gradPhi2)))
,
        epsilonxy * (6 /(xgp * xgp * xgp / safe_gradPhi12 + xgp + eps)) * (-sin(6 * atan2(gradPhi[1], safe_gradPhi0)))
        + epsilonz * (2 * ygp * zgp / (inner(gradPhi, gradPhi) * sqrt(xgp * xgp + ygp * ygp) + eps)) * (-sin(2 * atan_2(sqrt(xgp**2 + ygp**2), safe_gradPhi2)))
,
          epsilonz * (-2 * sqrt(xgp * xgp + ygp * ygp) / (inner(gradPhi, gradPhi) + eps)) * (-sin(2 * atan_2(sqrt(xgp**2 + ygp**2), safe_gradPhi2)))
    ])


    dtPhi = 1/(A*A)*(-fDash + Gamma * b *gDash*u_
                       +0.5*div(GammaMatrix * (inner(gradPhi, gradPhi) * dA2 + A*A*GammaMatrix*gradPhi)))

    dtPhi = 1/(A*A)*(-fDash + Gamma * b * gDash * u_
                       +0.5 * (2 * dot(div(GammaMatrix * gradPhi) * gradPhi, 2 * A * dA2)
                               + inner(gradPhi, gradPhi) * 2 * dot(grad(GammaMatrix * gradPhi) * dA2, dA2)
                               #+ A * div(GammaMatrix * dA2)))
                               + 2 * A * inner((grad(GammaMatrix * gradPhi) * dA2), (GammaMatrix * gradPhi))
                               + A*A * div(GammaMatrix * GammaMatrix * gradPhi)))
    dtU = D * div(GammaMatrix * q * GammaMatrix * grad(u_)) - 0.5 * Lsat * b * dtPhi
    
    #dtU = dtPhi

    form = (dtPhi * v[0] + dtU * v[1]) * dx
    
    form = (dtPhi * v[0] + dtU * v[1]) * dx

    return form, 10, 0.001, initial, None, [None]

    