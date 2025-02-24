# %% [markdown]
#
# # Exponential Integrators
#
# ## Setup

# %%
import sys
import argparse
import time as tm

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import lgmres, gmres, LinearOperator, aslinearoperator, expm_multiply
from scipy.sparse import identity, diags
from scipy.optimize import newton_krylov, KrylovJacobian
from scipy.special import factorial
import scipy.integrate as scipyintegrate
import random

from scipy.sparse.linalg import expm_multiply
expm_sci = [lambda A,x,m: expm_multiply(A,x),"Scipy"]
from Stable_Lanzcos import LanzcosExp, Lanzcos
expm_lanzcos = [lambda A,x,m: LanzcosExp(A,x,m),"Lanzcos"]
from NBLA import NBLAExp
expm_nbla = [lambda A,x,m: NBLAExp(A,x,m),"NBLA"]
from Arnoldi import ArnoldiExp, Arnoldi
expm_arnoldi = [lambda A,x,m: ArnoldiExp(A,x,m),"Arnoldi"]
from kiops import KiopsExp
expm_kiops = [lambda A,x,m: KiopsExp(A,x),"Kiops"]

from dune.grid import cartesianDomain, OutputType
from dune.alugrid import aluCubeGrid as leafGridView
from dune.fem.space import lagrange
from dune.fem import integrate, threading, globalRefine, mark, adapt, loadBalance
from dune.fem.view import adaptiveLeafGridView as view
from dune.fem.operator import galerkin
from dune.fem.function import gridFunction
from ufl import dx, dot, inner, grad, TestFunction, TrialFunction
# %% [markdown]
#
# Simple utility function to show the result of a simulation

# %%
def printResult(time,error,*args):
    print(time,'L2:{:0.5e}, H1:{:0.5e}'.format(
                 *[ np.sqrt(e) for e in integrate([error**2,inner(grad(error),grad(error))]) ]),
          *args, " # gnuplot", flush=True)

# %% [markdown]
# ## A base class for the steppers
#
# Assume the non-linear operator is given by `N` so we are solving
# $u_t + N[u] = 0$
# 

# %%
class BaseStepper:
    # method is 'explicit', 'approx', 'exact'
    # mass is 'lumped', 'exact', 'identitiy'
    # Change:
    # - define class that wraps an operator 'N' and returns object with same
    #   interface but including M^{-1}
    # - put N on right hand side
    def __init__(self, N, *, method="approx", mass="lumped", grid="fixed", massWeight=None):
        self.N = N
        self.method = method
        self.mass = mass
        self.grid = grid
        self.spc = N.domainSpace
        self.un = self.spc.zero.copy()  # previous time step
        self.res = self.un.copy()       # residual
        self.shape = (self.spc.size,self.spc.size)

        if massWeight == None:
            self.massWeight = lambda u: 1
            self.hasMassWeigth = False
        else:
            self.massWeight = massWeight
            self.hasMassWeigth = True

        self.getMass()

        # time step to use
        self.tau = None

        if self.method == "exact":
            self.Nprime = self.N.linear()

        self.I = identity(self.shape[0])
        self.tmp = self.un.copy()
        self.countN = 0
        self.linIter = 0

    def getMass(self):
        if self.mass == 'identity':
            # This is hack!
            # Issue: the dgoperator is set to be on the right so we need
            # the action of -N. Since this is the only operator corrently
            # using the identity mass matrix we change the sign here
            # NEEDS FIXING
            self.Minv = -identity(self.shape[0]) 
        else:
            # inverse (lumped) mass matrix (see the explanation given in 'wave.py' tutorial example)
            # u^n v dx = sum_j u^n_j phi_j phi_i dx = M u^n
            # Mu^{n+1} = Mu^n - dt N(u^n) (for FE)
            # u^{n+1} = u^n - dt M^{-1}N(u^n)
            # bug: u,v = TrialFunction(N.domainSpace), TestFunction(N.rangeSpace)
            u,v = TrialFunction(self.N.domainSpace.as_ufl()), TestFunction(self.N.rangeSpace.as_ufl())
            
            M = galerkin(self.massWeight(self.un) * dot(u,v)*dx).linear().as_numpy

            if self.mass == 'lumped':
                Mdiag = M.sum(axis=1) # sum up the entries onto the diagonal
                self.Minv = diags( 1/Mdiag.A1, shape=(np.shape(Mdiag)[0], np.shape(Mdiag)[0]) )

    # call before computing the next time step
    def setup(self,un,tau):
        self.un.assign(un)
        self.tau = tau
        if self.grid != "fixed" or self.hasMassWeigth:
            self.getMass()
        if not "expl" in self.method:
            self.linearize(self.un)

    # evaluate w = tau M^{-1} N[u]
    def evalN(self,x):
        xh = self.N.domainSpace.function("tmp", dofVector=x)
        self.N(xh, self.tmp)
        self.countN += 1
        return self.tmp.as_numpy * (self.tau * self.Minv)

    # compute matrix D = tau M^{-1}DN[u] + I
    # e.g. BE: u^{n+1} + tau M^{-1}N[u^{n+1}] = u^n
    # lineaaized around u^n_k: (I + tau M^{-1}DN[u^n_k])u = u^n
    def linearize(self,u):
        assert not self.method == "expl"
        if self.method == "approx":
            self.A = aslinearoperator(self.Aapprox(self, u))
            self.D = aslinearoperator(self.Dapprox(self, u))
            # self.test(u)
        else:
            # assert False # we are not considering 'exact' at the moment
            self.N.jacobian(u,self.Nprime)
            self.A = self.tau * self.Minv @ self.Nprime.as_numpy
            self.D = self.A + self.I

    def test(self,u):
        self.N.jacobian(u,self.Nprime)
        self.A_ = self.tau * self.Minv @ self.Nprime.as_numpy
        x = 0*self.tmp.as_numpy
        maxVal = 0
        for k in range(10):
            i = random.randrange(0,len(x),1)
            x[i] = random.random()
            y = self.A@x - self.A_@x
            maxVal = max(maxVal,y.dot(y))
            # print(i,maxVal)
        print("difference:",maxVal)

    class Aapprox:
        def __init__(self,stepper, u):
            self.krylovJacobian = KrylovJacobian()
            self.shape = (u.as_numpy.shape[0],u.as_numpy.shape[0])
            self.dtype = u.as_numpy.dtype
            f = stepper.evalN(u.as_numpy)
            self.krylovJacobian.setup(u.as_numpy, f, stepper.evalN)
        # issue with x having shape (N,1) needed by exponential method and
        # not handled by KrylovJacobian
        def matvec(self,x):
            if len(x.shape) > 1:
                y = x.copy()
                y[:,0] = self.krylovJacobian.matvec(x[:,0])
            else:
                y = self.krylovJacobian.matvec(x)
            return y
        # assume problem is self-adjoint - how does this hold for
        # the linearization and needs fixing for non self-adjoint PDEs
        def rmatvec(self,x):
            return self.matvec(x)
        def solve(self, rhs, tol=0):
            return self.krylovJacobian.solve(rhs,tol)

    class Dapprox:
        def __init__(self,stepper, u):
            self.krylovJacobian = KrylovJacobian()
            self.stepper = stepper
            self.shape = (u.as_numpy.shape[0],u.as_numpy.shape[0])
            self.dtype = u.as_numpy.dtype
            f = stepper.evalN(u.as_numpy)
            self.krylovJacobian.setup(u.as_numpy, f, stepper.evalN)
        # issue with x having shape (N,1) needed by exponential method and
        # not handled by KrylovJacobian
        def matvec(self,x):
            if len(x.shape) > 1:
                y = x.copy()
                y[:,0] = self.krylovJacobian.matvec(x[:,0]) + x
            else:
                y = self.krylovJacobian.matvec(x) + x
            return y
        # assume problem is self-adjoint - how does this hold for
        # the linearization and needs fixing for non self-adjoint PDEs
        def rmatvec(self,x):
            return self.matvec(x)

    # the below should be the same - need to test to make sure
    """
    class Dapprox:
        def __init__(self,A):
            self.shape = A.shape
            self.dtype = A.dtype
            self.A = A
        def matvec(self,x):
            y = self.A@x
            print(y.dot(y))
            return self.A@x + x
    """

# %% [markdown]
# ## Forward Euler method
# solve d_t u + N(u) = 0 using u^{n+1} = u^n - tau N(u^n)
class FEStepper(BaseStepper):
    def __init__(self, N, *, mass='lumped', **kwargs):
        BaseStepper.__init__(self,N, method="explicit", mass=mass, **kwargs)
        self.name = "FE"

    def __call__(self, target, tau):
        self.setup(target,tau)
        target.as_numpy[:] -= self.evalN(self.un.as_numpy[:])
        return {"iterations":1, "linIter":0}

# %% [markdown]
# ## Backward Euler method
# solve d_t u + N(u) = 0 using u^{n+1} = u^n - tau N(u^{n+1})
class BEStepper(BaseStepper):
    def __init__(self, N, *, method="approx", mass='lumped',
                 ftol=1e-6, verbose=False, **kwargs):
        BaseStepper.__init__(self,N, method=method, mass=mass, **kwargs)
        self.name = f"BE({self.method})"
        self.verbose = self.callback if verbose else None
        self.ftol = ftol

    # non linear function (non-linear problem want f(x)=0)
    def f(self, x):
        return self.evalN(x) + ( x - self.un.as_numpy )
    def callback(self,x,Fx): 
        print(self.countN, max(abs(Fx)),flush=True)
        self.linIter += 1

    def __call__(self, target, tau):
        try:
            self.N.model.sourceTime += tau
        except AttributeError:
            pass
        self.setup(target,tau)
        # get numpy vectors for target, residual, search direction
        sol_coeff = target.as_numpy
        sol_coeff[:] = newton_krylov(self.f, xin=sol_coeff, f_tol=self.ftol,
                                     callback=self.verbose)
        return {"iterations":0, "linIter":self.linIter}

# %% [markdown]
# ## A semi-implicit approach (with a linear implicit part)
# solve d_t u + N(u) = 0 using u^{n+1} = u^n - tau A^n u^{n+1} - tau R^n(u^n)
# with A^n = DN(u^n) and R^n(u^n) = N(u^n) - A^n u^n, i.e., rewritting
# N(u) = DN(u)u + (N(u) - DN(u)u)
# So in each step we need to solve
# ( I + tau A^n ) u^{n+1} = u^n - tau R^n(u^n)
#                         = ( I + tau A^n ) u^n - tau N(u^n)
# u^{n+1} = u^n - tau ( I + tau A^n )^{-1} N(u^n)
class SIStepper(BaseStepper):
    def __init__(self, N, *, method="approx", mass='lumped', **kwargs):
        BaseStepper.__init__(self,N, method=method, mass=mass, **kwargs)
        self.name = f"SI({self.method})"

    def explStep(self, target, tau):
        # this sets D = I + tau A^n and initialized u^n
        self.setup(target, tau)
        res_coeff = self.res.as_numpy
        # compute N(u^n)
        res_coeff[:] = self.evalN(target.as_numpy)
        # subtract Du^n
        res_coeff[:] -= self.D@target.as_numpy
        res_coeff[:] *= -1

    def callback(self,x): 
        y = self.D@x - self.res.as_numpy
        print(self.linIter,y.dot(y), self.countN,flush=True)
        self.linIter += 1

    def __call__(self, target, tau):
        self.explStep(target, tau)

        sol_coeff = target.as_numpy
        res_coeff = self.res.as_numpy
        self.linIter = 0
        # solve linear system
        sol_coeff[:],_ = lgmres(self.D, res_coeff, # x0=self.un.as_numpy,
                                rtol=1e-2,
                                # callback=lambda x: self.callback(x),
                                # callback_type='x'
                               )
        return {"iterations":0, "linIter":self.linIter}

# %%
# solve d_t u + N(u) = 0 using exponential integrator for
# d_t u + A^n u + R^n(u) = 0 with A^n = DN(u^n) and R^n(u) = N(u) - A^n u
# Set v = e^{A^n(t-t^n)} u then
# d_t v = e^{A^n(t-t^n)) A^n u + e^{A^n(t-t^n)} d_t u
#       = e^{A^n(t-t^n)) A^n u - e^{A^n(t-t^n)} (A^n u + R^n(u))
#       = - e^{A^n(t-t^n)) R^n(u)
#       = - e^{A^n(t-t^n)) R^n( e^{-A^n(t-t^n)}v )
# Then using FE:
# v^{n+1} = v^n - tau R^n(v^n)
# e^{A^n tau)u^{n+1} = u^n - tau R^n(u^n) since u(t^n) = v(t^n)
# u^{n+1} = e^{-A^n tau} ( u^n - tau R^n(u^n) )
#         = e^{-A^n tau} ( u^n - tau (N(u^n) - A^n u^n) )
#         = e^{-A^n tau} ( (I + tau A^n)u^n - tau N(u^n) )

class ExponentialStepper(SIStepper):
    def __init__(self, N, exp_v, *, expv_args, method='approx', mass='lumped', **kwargs):
        SIStepper.__init__(self,N, method=method, mass=mass, **kwargs)
        self.name = f"ExpInt({self.method},{exp_v[1]},{expv_args})"
        self.exp_v = exp_v[0]
        self.expv_args = expv_args

    def __call__(self, target, tau):
        # u^* = (I + tau A^n)u^n - tau N(u^n)
        # Note: the call method on the base class calls 'setup' which computes the right A^n
        self.explStep(target,tau)
        # Compute e^{-tau A^n}u^*
        target.as_numpy[:] = self.exp_v(- self.A, self.res.as_numpy, **self.expv_args)
        return {"iterations":0, "linIter":0}


class FristOrderExponentialStepper(ExponentialStepper):
    def __init__(self, N, exp_v, krylovMethod, *, integration='simple', **kwargs):
        ExponentialStepper.__init__(self,N,exp_v=exp_v, **kwargs)
        self.name = f"ExpIntFirstOrder({self.method},{exp_v[1]},{self.expv_args})"
        self.krylovMethod = krylovMethod
        self.integration = integration

    def __call__(self, target, tau):
        self.setup(target, tau)

        self.res.as_numpy[:] = self.exp_v(- self.A, target.as_numpy, **self.expv_args)

        R = self.evalN(target.as_numpy) - self.A@target.as_numpy[:]

        H, V, beta = self.krylovMethod(self.A, -R, self.expv_args["m"])
        
        target.as_numpy[:] = self.res.as_numpy[:] + V@self.phi_k(-H, 1)*beta
        
        return {"iterations":0, "linIter":0}

    def phi_k(self, H, k):
        e_1 = np.zeros((self.expv_args["m"]))
        e_1[0] = 1
        func = lambda t: expm_multiply((1-t) * H, e_1) * (t)**(k-1)/factorial(k-1)
        return self.integrate(func, 0, 1)

    def integrate(self, func, a, b):
        if self.integration == 'simple':
            n = 10
            xs, dx = np.linspace(a+(b-a)/n,b,n, retstep=True) # might not work for all a and b
            xs -= dx/2
            xs = [func(x) for x in xs]
            return np.sum(xs, 0)*dx
        else:
            return scipyintegrate.quad_vec(func, a, b)[0]


# Seems to work?
class SecondOrderExponentialStepper(FristOrderExponentialStepper):
    def __init__(self, N, exp_v, *, integration='simple', c=0.5, **kwargs):
        FristOrderExponentialStepper.__init__(self, N=N, exp_v=exp_v, **kwargs)
        self.name = f"ExpIntSecondOrder({self.method},{exp_v[1]},{self.expv_args})"
        self.c = c

    def __call__(self, target, tau):
        self.setup(target, tau)
        
        e_1 = np.zeros((self.expv_args["m"]))
        e_1[0] = 1
        #result = self.exp_v(- self.A, target.as_numpy, **self.expv_args) #This subspace can be reused (should fix)

        H1, V1, beta1 = self.krylovMethod(self.A, target.as_numpy, self.expv_args["m"])
        result = V1@expm_multiply(-H1, e_1) * beta1

        R = self.evalN(target.as_numpy[:]) - self.A@target.as_numpy[:]
        H2, V2, beta2 = self.krylovMethod(self.A, -R, self.expv_args["m"])
        result += V2@(self.phi_k(-H2, 1) - 1/self.c * self.phi_k(-H2, 2))*beta2

        #self.res.as_numpy[:] = self.exp_v(- self.c * self.A, target.as_numpy, **self.expv_args) #You can reuses the subspace generated above
        self.res.as_numpy[:] = V1@expm_multiply(-H1 * self.c, e_1) * beta1
        self.res.as_numpy[:] += self.c * V2@self.phi_k(-H2*self.c, 1)*beta2

        try: # If model has no sourceTime then model is not time dependant so can safly ignore
            temp = self.N.model.sourceTime
            self.N.model.sourceTime += self.c * tau
            self.linearize(target)
        except:
            pass
        R2 = self.evalN(self.res.as_numpy[:]) - self.A@self.res.as_numpy[:]
        try:
            self.N.model.sourceTime = temp
            self.linearize(target)
        except:
            pass

        H3, V3, beta3 = self.krylovMethod(self.A, -R2, self.expv_args["m"])
        result += V3 @ (self.phi_k(-H3, 2) * (1 / self.c)) * beta3

        target.as_numpy[:] = result
        return {"iterations":0, "linIter":0}

steppersDict = {"FE": (FEStepper,{}),
                "BE": (BEStepper,{}),
                "SI": (SIStepper,{}),
                "EXPSCI": (ExponentialStepper, {"exp_v":expm_sci}),
                "EXPLAN": (ExponentialStepper, {"exp_v":expm_lanzcos}),
                "EXP1LAN": (FristOrderExponentialStepper, {"exp_v":expm_lanzcos, "krylovMethod":Lanzcos}),
                "EXP2LAN": (SecondOrderExponentialStepper, {"exp_v":expm_lanzcos, "krylovMethod":Lanzcos}),
                "EXPNBLA": (ExponentialStepper, {"exp_v":expm_nbla}),
                "EXPARN": (ExponentialStepper, {"exp_v":expm_arnoldi}),
                "EXP1ARN": (FristOrderExponentialStepper, {"exp_v":expm_arnoldi, "krylovMethod":Arnoldi}),
                "EXP2ARN": (SecondOrderExponentialStepper, {"exp_v":expm_arnoldi, "krylovMethod":Arnoldi}),
                "EXPKIOPS": (ExponentialStepper, {"exp_v":expm_kiops}),
               }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Script to Run Stepper')
    parser.add_argument('--problem', help = "Problem To Run")
    parser.add_argument('--stepper', help = "Stepper To Use")
    parser.add_argument('--factor', help = "Time Step Multiplication Factor", nargs='*', default=[1])
    parser.add_argument('--krylovsize', help = "Dimention of the Kyrlov Subspace", nargs='*', default=[64])
    parser.add_argument('--refinement', help = "Refinement of the Grid", nargs='*', default=[0])
    parser.add_argument('--adaptive', help = "Is an adaptive grid begin used", action='store_true')
    sysargs = parser.parse_args()

    threading.use = max(8,threading.max)

    def adaptGrid(u_h):
        return

    kwargs = {}
    if sysargs.problem=="TravellingWaveAllenCahn":
        from travellingWaveAllenCahn import dimR, time, sourceTime, domain
        from travellingWaveAllenCahn import test3 as problem
        baseName = "TravellingWaveAllenCahn"
        order = 1
    elif sysargs.problem=="TravellingWaveAllenCahn3D":
        from travellingWaveAllenCahn3D import dimR, time, sourceTime, domain
        from travellingWaveAllenCahn3D import test3 as problem
        baseName = "TravellingWaveAllenCahn3D"
        order = 1
    elif sysargs.problem=="TravellingWaveAllenCahn2":
        from travellingWaveAllenCahn2 import dimR, time, sourceTime, domain
        from travellingWaveAllenCahn2 import test3 as problem
        baseName = "TravellingWaveAllenCahn2"
        order = 1
    elif sysargs.problem=="Test":
        from test_problem import dimR, time, sourceTime
        from test_problem import test1 as problem
        domain = [-1, -1], [1, 1], [10, 10]
        baseName = "Test"
        order = 1
    elif sysargs.problem=="ReactionDiffusion":
        from reaction_diffusion import dimR, time, sourceTime, domain
        from reaction_diffusion import test1 as problem
        baseName = "ReactionDiffusion"
        order = 1
    elif sysargs.problem=="Snowflake":
        from snowflakes import dimR, time, sourceTime, domain
        from snowflakes import test1 as problem
        from dune.alugrid import aluConformGrid as leafGridView
        baseName = "Snowflake"
        order = 1
        if sysargs.adaptive:
            kwargs = {"grid": "adaptive"}
            def adaptGrid(u_h):
                indicator = dot(grad(u_h[0]),grad(u_h[0]))
                mark(indicator,1.4,1.2,0,11)
                adapt(u_h)
    elif sysargs.problem=="Snowflake3D":
        from snowflakes3D import dimR, time, sourceTime, domain
        from snowflakes3D import test1 as problem
        from dune.alugrid import aluConformGrid as leafGridView
        baseName = "Snowflake3D"
        order = 1
        if sysargs.adaptive:
            kwargs = {"grid": "adaptive"}
            def adaptGrid(u_h):
                indicator = dot(grad(u_h[0]),grad(u_h[0]))
                mark(indicator,0.1,0.01,0,10, markNeighbors = False)
                adapt(u_h)
    elif sysargs.problem=="Parabolic":
        from parabolicTest import dimR, time, sourceTime, domain
        from parabolicTest import paraTest2 as problem
        baseName = "Parabolic Test2"
        order = 1
    else:
        print("No Valid Problem Provided")
        quit()

    # ## Setup grid, space, and operator
    gridView = view( leafGridView(cartesianDomain(*domain)) )
    space = lagrange(gridView, order=order, dimRange=dimR)

    model, T, tauFE, u0, exact, massWeight = problem(gridView)
    kwargs["massWeight"] = massWeight

    stepperFct, args = steppersDict[sysargs.stepper]
    if "exp_v" in args.keys():
        m = int(sysargs.krylovsize[0])
        args["expv_args"] = {"m":m}

    factor = float(sysargs.factor[0])
    tau = tauFE * factor

    # refinement
    level = int(sysargs.refinement[0])

    if level>0:
        gridView.hierarchicalGrid.globalRefine(level)
        #tau *= 0.25**level

    outputName = lambda n: f"{baseName}_{level}_{factor}_{n}_{sysargs.krylovsize[0]}.png"

    # initial condition
    u_h = space.interpolate(u0, name='u_h')

    # stepper
    op = galerkin([model], domainSpace=space, rangeSpace=space)
    stepper = stepperFct(N=op,**args,**kwargs)

    # time loop
    n = 0
    totalIter, linIter = 0, 0
    run = []

    plotTime = T/10
    nextTime = plotTime
    fileCount = 0

    
    if sysargs.adaptive:
        for i in range(10):
            print("adapting")
            adaptGrid(u_h)
            u_h.interpolate(u0)
    try:
        u_h.plot(block=False)
        plt.savefig(outputName(fileCount))
    except:
        gridView.writeVTK(outputName(fileCount), pointdata=[u_h[0]], outputType=OutputType.appendedraw)
        gridView.writeVTK(outputName(fileCount) + "T", pointdata=[u_h[1]], outputType=OutputType.appendedraw)
    fileCount += 1
    if exact is not None:
        printResult(time.value,u_h-exact(time),stepper.countN)

    computeTime = 0
    while time.value < T - tau/2:
        print(time.value)
        # this actually depends probably on the method we use, i.e., BE would
        # be + tau and the others without
        sourceTime.value = time.value
        if sysargs.adaptive:
            adaptGrid(u_h)
        computeTime -= tm.time()
        info = stepper(target=u_h, tau=tau)
        computeTime += tm.time()
        assert not np.isnan(u_h.as_numpy).any()
        time.value += tau
        totalIter += info["iterations"]
        linIter   += info["linIter"]
        n += 1
        
        if time.value >= plotTime - tau/2:
            print(f"[{fileCount}]: time step {n}, time {time.value}, N {stepper.countN}, iterations {info}, compute time {computeTime}",
                    flush=True)
            if exact is not None:
                printResult(time.value,u_h-exact(time),stepper.countN)
            run += [(stepper.countN,linIter)]
            try:
                u_h[0].plot(block=False)
                plt.savefig(outputName(fileCount))
            except:
                gridView.writeVTK(outputName(fileCount), pointdata=[u_h[0]], outputType=OutputType.appendedraw)
                gridView.writeVTK(outputName(fileCount) + "T", pointdata=[u_h[1]], outputType=OutputType.appendedraw)
            plotTime += nextTime
            fileCount += 1

    print(f"Final time step {n}, time {time.value}, N {stepper.countN}, iterations {info}, compute time {computeTime}")
    try:
        u_h.plot(gridLines=None, block=False)
        gridView.writeVTK(baseName, pointdata={"u_h": u_h}, subsampling=2)
        plt.savefig(outputName(fileCount))
        fileCount += 1
        u_h = space.interpolate(exact(T), name='u_h')
        u_h.plot(block=False)
        plt.savefig(outputName("Solution"))
    except:
        pass

    