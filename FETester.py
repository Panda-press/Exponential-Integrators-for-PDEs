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
from dune.fem import integrate, threading
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
                 **stepper_args):

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
        test_stepper_name = test_stepper.name +\
                            hashlib.sha1( repr(sorted(temp.items())).encode('utf-8') ).hexdigest()
        test_file_name = self.get_test_results_file(tau, test_stepper_name, end_time)
        if not os.path.isfile(test_file_name):
            self.test_results, self.test_countN = self.run(tau, test_stepper, self.initial_condition, self.seed_time, end_time)
            with open(test_file_name, 'wb') as file:
                pickle.dump([self.test_results.as_numpy[:],self.test_countN], file)
        else:
            self.test_results = self.initial_condition.copy()
            with open(test_file_name, 'rb') as file:
                self.test_results.as_numpy[:], self.test_countN = pickle.load(file)
        

    def run(self, tau, stepper, initial_condition, start_time, end_time):
        # Runs for a given stepper
        current_step = initial_condition.copy()
        time = start_time
        while time < end_time - tau/2:
            stepper.N.model.sourceTime = time
            stepper(target=current_step, tau = tau)
            time += tau
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

    threading.use = max(8,threading.max)
    if sysargs.problem == "TravellingWaveAllenCahn":
        from allenCahn import dimR, time, sourceTime, domain
        domain = [-4, -1], [8, 1], []
        from allenCahn import test3 as problem
        problemName = "Travelling Wave"
        start_time = 0
        end_time = 8
        if sysargs.debug == True:
            krylovSizes = [5, 10]
            tau0 = 2e-1
            taus = 3
            grids = [[5, 10], [10, 10], [30, 10]]
            exp_methods = ["EXPLAN", "BE", "EXPKIOPS", "EXP1LAN"]
        else:
            tau0 = 1e-1 # Using this value as a higher value such as 8e-2 causes numerical issues
            taus = 9
            grids = [[10, 10], [30, 10], [60, 10], [120, 10], [240, 10], [480,10]]
            exp_methods = ["EXPLAN", "EXPKIOPS", "BE", "EXP1LAN"]
            krylovSizes = [5, 10, 20]
    else:
        from parabolicTest import dimR, time, sourceTime, domain
        from parabolicTest import paraTest2 as problem
        problemName = "Parabolic Test2"
        tau0 = 1e-4  # 2e-3,N:32,Tau:0.002: compare m=5->m=10
        taus = 4
        start_time = 0.00
        end_time = 0.02
        grids = [[10, 10]]#, [30, 30], [60, 60]]
        exp_methods = ["EXPARN", "EXPLAN", "EXPKIOPS", "BE"]

    results = []

    domain = list(domain)
    for tau in tau0*np.array([2**-i for i in range(0, taus)]):
        print("Tau:{0}".format(tau))
        for grid in grids:
            print("N:{0}".format(grid))
            for exp_method in exp_methods:
                print("EXP method:{0}".format(exp_method))
                for kyrlovSize in krylovSizes:

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

                    model, T, tauFE, u0, exact = problem(gridView)
                    op = galerkin(model, domainSpace=space, rangeSpace=space)

                    u_h = space.interpolate(u0, name='u_h')


                    tester = Tester(u_h, op, problemName, start_time, exact=exact)
                    
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
        plt.savefig(f"FEMethodPlots/Tau V Operator Calls for {problemName} with grid size {grid_size}.png")
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
        plt.savefig(f"FEMethodPlots/Operator Calls V Error for {problemName} with grid size {grid_size}.png")
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
        plt.savefig(f"FEMethodPlots/Tau V L2 Error for {problemName} with grid size {grid_size}.png")
        plt.close()

    # Tau vs error split my method
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
        plt.savefig(f"FEMethodPlots/Tau V L2 Error for {exp_method} for {problemName}.png")
        plt.close()


