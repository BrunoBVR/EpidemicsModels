# SIR folder

## SIR.py:

Simple implementation of the SIR model with and object-oriented approach. Just running the code will generate curves for time evolution of Susceptibles, Infected and Removed individuals using an Euler integration scheme and a second order Runge-Kutta scheme.

Default parameters can be changed in lines 87-93.

## SIR-GP.py:

Implementation of a Gillespie algorithm for the SIR model. Uncommenting all lines of code bellow line 84 will allow for multiple runs of the algorithm and plot histograms with the distribution of final values of S and R. The parameter `sMax` controls the number of samples in the multiple run.

## SIR_GP_MP.py:

Implementation of a Gillespie algorithm for the SIR model. Using `multiprocessing` module to optimize the multiple runs of the algorithm. The number of processes used can be changed in line 65.

# SEIR folder

## SEIR.py

Implementation of the SEIR model with a fourth order Runge-Kutta scheme. This implementation allows for the simulation of a quarantine period. This period will start once the number of infected individuals reach `max_inf` and will last for a time equal to `t_max_Q`. Initial conditions can be found on lines 21-42.

Simply running the code will generate 8 figures:

1. Time evolution of all individuals (S,E,I,R)
2. Time evolution of Infected individuals with a vertical line showing the time it took to reach `max_inf` individuals.
3. Cumulative number of infected individuals.
4. Comparison between the previous curve and that of recovered individuals.
5. Evolution of the infected population in different scenarios based on the reduction of the basic reproduction number. The number on the legend box corresponds to the reduction.
6. Time evolution of all individuals given a quarantine period. This period is shown as a horizontal green bar.
7. Time evolution of infected individuals in a quarantine regime.
8. Comparison between the evolution of infected individuals when there is no quarantine and when a quarantine with a 70% reduction in the basic reproduction number is done.

The file `funcs.py` contains all functions use to simulate the model. This includes:

* function `f`: the dynamics of the model (vector field).
* function `RK4`: fourth order RUnge-Kutta step.
* function `time-loop`: time evolution of the dynamics.

## SEIR-GP.py

Implementation of a Gillespie algorithm for the SEIR model. Initial conditions can be modified in lines 86-100. Variable `ts` is a choice of the printing interval for a sanity check on terminal (printing the time and total population - that must be a constant). Simply running the code produces a figure with the time evolution of all individuals.

## SEIR.ipynb

Notebook with all the code in SEIR.py **plus** a section on fitting the model to actual data from Brasília (Capital city of Brazil).

Data was gathered from [here](https://covid.saude.gov.br/). We perform a simple least squares method with a Nelder-Mead method for minimization. The last figure shows a comparison between the fitted model and data. The model performs well for the very beginning of the pandemic and we can get an estimate for the basic reproduction number for this period.

Functions for fitting data are inside the `funcs_fit.py` file. This includes:
* Same `f` and `RK4` as before.
* a `time_loop` function that evolves the dynamics.
* a `seir_model` function that accounts for the evolution of infected and recovered individuals cumulatively.

## DataCleaning.ipynb

Simple notebook to test data cleaning of file gathered [here](https://covid.saude.gov.br/).
