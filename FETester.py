import numpy as np
import pickle as pickle
import os as os
import hashlib
import copy
import pandas as pd
import matplotlib.pyplot as plt
import sys
from ufl import *
from dune.ufl import Space, Constant
from dune.fem import integrate, threading, mark, adapt, loadBalance
from dune.fem.function import gridFunction
from dune.fem.space import lagrange
from dune.grid import cartesianDomain
from dune.fem.operator import galerkin
from dune.alugrid import aluCubeGrid as leafGridView
from dune.fem.view import adaptiveLeafGridView as view
from steppers import BEStepper, steppersDict


results_folder = "FETests"
# results_folder = "FETests2009"

class Tester():
    def __init__(self, 
                 initial_condition, 
                 op, 
                 problem_name, 
                 seed_time = 0, 
                 setup_tau = 1e-2,
                 target_tau = 1e-4,
                 setup_stepper = BEStepper,
                 exact = None,
                 gridAdaptFunc = lambda u_h: None,
                 stepper_args = {}):
        self.op = op
        self.N = self.op.domainSpace.gridView.size(0)
        self.initial_condition = initial_condition
        self.seed_time = seed_time
        self.setup_stepper = setup_stepper(op, **stepper_args)
        self.target_tau = target_tau
        self.folder = results_folder + "/Problem:{0}_Grid_Size:{4}_Setup_Tau:{2}_Setup_Stepper:{1}_Seed_Time:{3}".format(problem_name,self.setup_stepper.name,setup_tau,self.seed_time,self.N)
        self.exact = exact
        
        self.run_setup(setup_tau, initial_condition)

    def get_initial_file(self):
        return self.folder + "_Initial.pickle"
    
    def get_target_results_file(self, tau, end_time):
        return self.folder + "_Tau:{0}_End_Time:{1}_Target.pickle".format(tau, end_time)

    def get_test_results_file(self, tau, stepper_name, end_time):
        return self.folder + "_Tau:{0}_End_Time:{1}_Stepper:{2}.pickle".format(tau, end_time, stepper_name)

    def run_setup(self, tau, initial_condition):
        if self.exact is not None:
            return

        # File path to initial data
        self.intitial_file_name = self.get_initial_file()

        # Check if file alread exist
        if os.path.isfile(self.intitial_file_name):
            with open(self.intitial_file_name, 'rb') as file:
                self.initial_condition.as_numpy[:] = pickle.load(file)

        # If not generate it
        else:
            self.initial_condition, _ = self.run(tau, self.setup_stepper, initial_condition, 0, self.seed_time)
            with open(self.intitial_file_name, 'wb') as file:
                pickle.dump(self.initial_condition.as_numpy[:], file)
    
    def run_test(self, tau, test_stepper, stepper_args, end_time):
        
        # Load initial conditions
        if self.exact is not None:
            self.initial_condition.interpolate(self.exact(self.seed_time))
        else:
            with open(self.intitial_file_name, 'rb') as file:
                self.initial_condition.as_numpy[:] = pickle.load(file)

        # Generate target data if it doesn't exist
        if self.exact is None:
            target_file_name = self.get_target_results_file(self.target_tau, end_time)
            if not os.path.isfile(target_file_name):
                self.target,self.target_countN = self.run(self.target_tau, self.setup_stepper, self.initial_condition, self.seed_time, end_time)
                with open(target_file_name, 'wb') as file:
                    pickle.dump([self.target.as_numpy[:],self.target_countN], file)
            else:
                self.target = self.initial_condition.copy()
                with open(target_file_name, 'rb') as file:
                    self.target.as_numpy[:], self.target_countN = pickle.load(file)
        else:
            self.target = gridFunction(exact(end_time),
                               gridView=op.domainSpace.gridView,order=5)
            self.target_countN = 0

        # Generate test stepper data if it doesn't exist
        test_stepper = test_stepper(op, **stepper_args)
        temp = copy.deepcopy(stepper_args)
        try:
            temp['exp_v'][0] = 0
        except:
            pass
        test_stepper_name = test_stepper.name
        test_file_name = self.get_test_results_file(tau, test_stepper_name, end_time)
        if not os.path.isfile(test_file_name):
            print("Running Tests")
            self.test_results, self.test_countN = self.run(tau, test_stepper, self.initial_condition, self.seed_time, end_time)
            with open(test_file_name, 'wb') as file:
                pickle.dump([self.test_results.as_numpy[:],self.test_countN], file)
        else:
            self.test_results = self.initial_condition.copy()
            print("Load Data")
            with open(test_file_name, 'rb') as file:
                self.test_results.as_numpy[:], self.test_countN = pickle.load(file)
        

    def run(self, tau, stepper, initial_condition, start_time, end_time):
        # Runs for a given stepper
        current_step = initial_condition.copy()
        time.value = start_time
        while time.value < end_time - tau/2:
            print(time.value)
            sourceTime.value = time.value
            stepper(target=current_step, tau = tau)
            time.value+= tau
            adaptGrid(current_step)
        countN = stepper.countN
        stepper.countN = 0
        return current_step, countN

    def produce_results(self, tau, stepper, stepper_args, end_time):
        self.run_test(tau, stepper, stepper_args, end_time)

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description = 'Script to Run Tests')
    parser.add_argument('--problem', help = "Problem To Run")
    parser.add_argument('--debug', help = "Is this just to test the script or to produce results", action='store_true')
    sysargs = parser.parse_args()

    threading.use = threading.max
    adaptGrid = lambda gridView: None
    stepper_args = {}
    if sysargs.problem == "TravellingWaveAllenCahn":
        from travellingWaveAllenCahn import dimR, time, sourceTime, domain
        from travellingWaveAllenCahn import test3 as problem
        problemName = "Travelling Wave Allen Cahn"
        start_time = 0
        end_time = 8
        if sysargs.debug == True:
            krylovSizes = [64]
            tau0 = end_time / 16 # Any higher gives numerical issues
            taus = 2
            grids = [[512, 8]]
            exp_methods = ["BE", "CN", "EXP1ALTLAN"]
        else:
            tau0 = end_time / 1
            taus = 9
            grids = [[128, 8], [256, 8], [512, 8]]
            exp_methods = ["BE", "CN", "EXP1ALTLAN"]
            krylovSizes = [8, 16, 32, 64]
    elif sysargs.problem == "TravellingWaveAllenCahn2":
        from travellingWaveAllenCahn2 import dimR, time, sourceTime, domain
        from travellingWaveAllenCahn2 import test3 as problem
        problemName = "Travelling Wave2"
        start_time = 0
        end_time = 3*1.414*0.05/5
        tau0 = end_time / 16
        taus = 5
        krylovSizes = [16, 32, 64, 128]
        grids = [[1024, 64]]
        exp_methods = ["EXP1LAN", "EXP2LAN"]
    elif sysargs.problem == "AllenCahn":
        from allenCahn import dimR, time, sourceTime, domain
        from allenCahn import test2 as problem
        problemName = "Allen Cahn"
        start_time = 0
        end_time = 24
        tau0 = 1
        taus = 8
        krylovSizes = [4, 8, 16, 32]
        grids = [[60, 60]]
        exp_methods = ["BE", "EXP1LAN", "EXP2LAN", "EXP1ALTLAN"]
    elif sysargs.problem=="Snowflake":
        from snowflakes import dimR, time, sourceTime, domain
        from snowflakes import test1 as problem
        from dune.alugrid import aluCubeGrid as leafGridView
        problemName = "Snowflake"
        grids = [[256, 256]]
        start_time = 0
        end_time = 0.1
        tau0 = 5e-4
        taus = 4
        krylovSizes = [8,10,12,14,16]
        exp_methods = ["EXP1LAN", "EXP2LAN"]
        order = 1
    elif sysargs.problem=="ReactionDiffusion":
        from reaction_diffusion import dimR, time, sourceTime, domain
        from reaction_diffusion import test1 as problem
        
        start_time = 0.00
        end_time = 0.1
        tau0 = end_time / 1
        taus = 5
        start_time = 0
        problemName = "Reaction Diffusion"
        exp_methods = ["BE", "EXP1LAN", "EXP2LAN", "EXP1ALTLAN"]
        krylovSizes = [8, 16, 32, 64]
        grids = [[128, 64], [256, 128]]
    elif sysargs.problem=="Test":
        from test_problem import dimR, time, sourceTime, domain
        from test_problem import test1 as problem
        
        start_time = 0.00
        end_time = 5
        tau0 = end_time / 1
        taus = 6
        start_time = 0
        problemName = "TestProblem"
        exp_methods = ["FE", "EXP1LAN", "EXP2LAN"]
        krylovSizes = [16, 64]
        grids = [[8, 8]]
    elif sysargs.problem=="Test2":
        from test_problem import dimR, time, sourceTime, domain
        from test_problem import test2 as problem
        
        start_time = 0.00
        end_time = 0.5
        tau0 = 2**(-5)
        taus = 4
        start_time = 0
        problemName = "TestProblem2"
        exp_methods = ["FE", "EXP1LAN", "EXP2LAN"]
        krylovSizes = [1]
        grids = [[4, 4]]
    else:
        from parabolicTest import dimR, time, sourceTime, domain
        from parabolicTest import paraTest0 as problem
        problemName = "Parabolic Test0"
        tau0 = 2e-2  # 2e-3,N:32,Tau:0.002: compare m=5->m=10
        taus = 5
        start_time = 0.00
        end_time = 0.02
        grids = [[30, 30], [50, 50]]#, [100, 100]]
        krylovSizes = [8, 16]#, 32, 64, 128]
        exp_methods = ["BE", "EXP1LAN", "EXP2LAN"]

    results = []

    domain = list(domain)
    for tau in tau0*np.array([2**-i for i in range(0, taus)]):
        print("Tau:{0}".format(tau))
        for grid in grids:
            print("N:{0}".format(grid))
            for exp_method in exp_methods:
                print("Stepper method:{0}".format(exp_method))
                for kyrlovSize in krylovSizes:
                    print(f"Krylov size {kyrlovSize}")

                    exp_stepper, args = steppersDict[exp_method]
                    if "exp_v" in args.keys() and exp_method != "EXPKIOPS":
                        args["expv_args"] = {"m":kyrlovSize}
                        name = f"{exp_method} {kyrlovSize}"
                    else:
                        name = exp_method
                        if exp_method == "EXPKIOPS":
                            args["expv_args"] = {"m":None}
                        if kyrlovSize != krylovSizes[0]:
                            break



                    domain[2] = grid

                    gridView = view(leafGridView(cartesianDomain(*domain)) )
                    space = lagrange(gridView, order=1, dimRange=dimR)

                    model, T, tauFE, u0, exact, diriBC = problem(gridView)
                    op = galerkin(model, domainSpace=space, rangeSpace=space)

                    u_h = space.interpolate(u0, name='u_h')

                    for _ in range(0, 10):
                        adaptGrid(u_h)
                        u_h.interpolate(u0)

                    tester = Tester(u_h, op, problemName, start_time, exact=exact, gridAdaptFunc = adaptGrid, stepper_args=stepper_args)
                    
                    tester.produce_results(tau, exp_stepper, args, end_time)

                    #tester.test_results.plot()
                    #tester.target.plot()
                    error = tester.test_results - tester.target
                    ref = [ np.sqrt(r) for r in
                            integrate([tester.target**2,
                                    inner(grad(tester.target),grad(tester.target))]
                                    )]
                    H1err = [ np.sqrt(e)/r
                            for r,e in zip(ref,integrate([error**2,inner(grad(error),grad(error))])) ]

                    print(f"{name},{tau},{gridView.size(0)}: {H1err}")
                    # if exact is not None:
                    #     exact_error = ...

                    # write file self.test_results.plot()
                    results += [ [name,gridView.size(0),tau,H1err[0],H1err[1],
                                        tester.target_countN,tester.test_countN] ]

    # produce plots using 'results'

    results = pd.DataFrame(results)
    results.columns = ["Method", "Grid Size", "Tau", "Error L2", "Error H1", "Target N Count", "Test N Count"]
    print("plotting")
    # N count vs tau
    for grid_size in results["Grid Size"].unique():
        if "BE" not in exp_methods:
            plt.scatter(results["Tau"], results["Target N Count"], marker="x", label="BE Method")
        for i, exp_method in enumerate(results["Method"].unique()):
            trimmed_data = results[results["Method"] == exp_method]
            trimmed_data = trimmed_data[trimmed_data["Grid Size"] == grid_size]

            plt.scatter(trimmed_data["Tau"], trimmed_data["Test N Count"], marker="x", label=exp_method)

        plt.legend()
        plt.xlabel("Tau")
        plt.ylabel("Calls")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both")
        plt.title(f"Tau V Opperator Calls for {problemName} with grid size {grid_size}")
        plt.savefig(f"FEMethodPlots/Tau V Operator Calls for {problemName} with grid size {grid_size}.svg")
        plt.close()

    # N count v Error
    for grid_size in results["Grid Size"].unique():
        if "BE" not in exp_methods:
            plt.plot(results["Target N Count"], results["Error L2"], marker="x", label="BE Method")
        for i, exp_method in enumerate(results["Method"].unique()):
            trimmed_data = results[results["Method"] == exp_method]
            trimmed_data = trimmed_data[trimmed_data["Grid Size"] == grid_size]

            plt.plot(trimmed_data["Test N Count"], trimmed_data["Error L2"], marker="x", label=exp_method)

        plt.legend()
        plt.xlabel("Calls")
        plt.ylabel("L2 Error")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both")
        plt.title(f"Opperator Calls V L2 Error for {problemName} with grid size {grid_size}")
        plt.savefig(f"FEMethodPlots/Operator Calls V Error for {problemName} with grid size {grid_size}.svg")
        plt.close()

    # Tau v Error all on one graph
    for grid_size in results["Grid Size"].unique():
        if "BE" not in exp_methods:
            plt.plot(results["Tau"], results["Error L2"], marker="x", label="BE Method")
        for i, exp_method in enumerate(results["Method"].unique()):
            trimmed_data = results[results["Method"] == exp_method]
            trimmed_data = trimmed_data[trimmed_data["Grid Size"] == grid_size]

            plt.plot(trimmed_data["Tau"], trimmed_data["Error L2"], marker="x", label=exp_method)

        plt.legend()
        plt.xlabel("Tau")
        plt.ylabel("L2 Error")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both")
        plt.title(f"Tau V L2 Error for {problemName} with grid size {grid_size}")
        plt.savefig(f"FEMethodPlots/Tau V L2 Error for {problemName} with grid size {grid_size}.svg")
        plt.close()

    # Tau vs error split by method
    for exp_method in results["Method"].unique():
        trimmed_data = results[results["Method"] == exp_method]
        for grid_size in results["Grid Size"].unique():
            grid_data = trimmed_data[trimmed_data["Grid Size"] == grid_size]

            plt.plot(grid_data["Tau"], grid_data["Error L2"], marker=".", label=f"Grid Size {grid_size}")
        plt.title(f"Tau V L2 Error for {exp_method} for {problemName}")
        plt.xlabel("Tau")
        plt.ylabel("L2 Error")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both")
        plt.savefig(f"FEMethodPlots/Tau V L2 Error for {exp_method} for {problemName}.svg")
        plt.close()

    # EOC for each grid size:
    try:
        for grid_size in results["Grid Size"].unique():
            fig, ax = plt.subplots()
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            eocdata = []
            tablehead = []
            if "BE" not in exp_methods:
                next
            for i, exp_method in enumerate(results["Method"].unique()):
                tablehead.append(exp_method)
                trimmed_data = results[results["Method"] == exp_method]
                trimmed_data = trimmed_data[trimmed_data["Grid Size"] == grid_size]

                error1 = trimmed_data["Error L2"][:-1].to_numpy()
                error2 = trimmed_data["Error L2"][1:].to_numpy()
                tau1 = trimmed_data["Tau"][:-1].to_numpy()
                tau2 = trimmed_data["Tau"][1:].to_numpy()
                eoc = np.log(error1/error2)/np.log(tau1/tau2)
                eocdata.append(eoc.tolist())

            eocdata = pd.DataFrame(np.array(eocdata).T, columns=tablehead)

            ax.table(cellText=eocdata.values, colLabels=eocdata.columns, loc='center')

            fig.tight_layout()
            fig.savefig(f"FEMethodPlots/EOC for {problemName} on grid size: {grid_size}.svg", format='svg', dpi=1200)
    except:
        pass