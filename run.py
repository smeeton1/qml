import sys
sys.path.insert(1, './src')
import warnings
warnings.filterwarnings("ignore")
import argparse

import main as mn
import plotting as pl


vqc = False
tol= 0.02

parser = argparse.ArgumentParser(description="Getting values for .")
parser.add_argument("--vqc", type=bool, default=False, help="Flage to make vqc run.")
parser.add_argument("--tol", type=float, default=0.02, help="Tolerance for optimizers.")

args = parser.parse_args()
if args.vqc:
    vqc = args.vqc
if args.tol:
    tol = args.tol


print('---------------------------------')
# Generate Results and process for classical AI
print("Classical AI training")
resultsClassical = mn.generate_classical_results(50, 10, tol)
pl.print_results(resultsClassical, "resultsClassical.csv")
print('---------------------------------')
print("Classical AI plotting")
plotResultsClassical = pl.sort_results_for_plot(resultsClassical)
pl.one_method_plot(plotResultsClassical, "Classical AI")

print('---------------------------------')
# Generate Results and process for QNN
print("QNN training")
resultsQNN = mn.generate_qnn_results(50, 10, tol)
pl.print_results(resultsQNN, "resultsQNN.csv")
print('---------------------------------')
print("QNN plotting")
plotResultsQNN = pl.sort_results_for_plot(resultsQNN)
pl.one_method_plot(plotResultsQNN, "Quantum Hybrid")

print('---------------------------------')
# Plotting QNN and classical comparison.
print("Classical AI and QNN comparison plotting")
pl.one_measure_plot(plotResultsClassical, plotResultsQNN, 1, "Classical AI", "Quantum Hybrid")
pl.one_measure_plot(plotResultsClassical, plotResultsQNN, 2, "Classical AI", "Quantum Hybrid")
pl.one_measure_plot(plotResultsClassical, plotResultsQNN, 3, "Classical AI", "Quantum Hybrid")
pl.one_measure_plot(plotResultsClassical, plotResultsQNN, 4, "Classical AI", "Quantum Hybrid")
pl.one_measure_plot(plotResultsClassical, plotResultsQNN, 5, "Classical AI", "Quantum Hybrid")
pl.one_measure_plot(plotResultsClassical, plotResultsQNN, 6, "Classical AI", "Quantum Hybrid")

if vqc:
    print('---------------------------------')
    # Generate Results and process for VQC
    print("VQC training")
    resultsVQC = mn.generate_vqc_results()
    pl.print_results(resultsVQC, "resultsVQC.csv")
    print('---------------------------------')
    print("VQC plotting")
    plotResultsVQC = pl.sort_results_for_plot(resultsVQC)
    pl.one_method_plot(plotResultsVQC, "VQC")