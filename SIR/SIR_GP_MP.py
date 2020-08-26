import multiprocessing as mp
import numpy as np
import math
import matplotlib.pyplot as plt

def histPlot(x, bins, xl, yl, title='', grid=True):
	plt.xlabel(xl)
	plt.ylabel(yl)
	plt.hist(x, bins = bins)
	
	if title != '':
		plt.title(title)

	if grid:
		plt.grid(linestyle='--')	

def rates(beta, gamma, x):
	r = np.zeros(2)
	
	r[0] = beta*x[0]*x[1]/np.sum(x)
	r[1] = gamma*x[1]

	return r, np.sum(r)

def gpRun(x0, ti, tf, beta, gamma, seed):

	np.random.seed(seed)

	t = ti
	time = np.array(t)

	x = np.asarray(x0)

	while t <= tf:
		if x[1] == 0: break

		r, Rn = rates(beta, gamma, x)

		if np.random.rand() < r[0]/Rn:
			x[0] -= 1
			x[1] += 1

		else:
			x[1] -= 1
			x[2] += 1

		dt = -math.log(np.random.rand())/Rn

		t += dt

	return x
################# End of functions definition #################

# Main body of algorithm

# Setting initial population, time and parameters
x0 = (10000, 10, 0)		# Using tuple to keep constant for each new seed

ti = 0
tf = 100
beta = 0.4
gamma = 0.2

# Initializing multiple processes
pool = mp.Pool(processes=4)

# Number of seeds to run
sMax=100

# Running multiple seeds and storing results in sMax-dimensional list 'results'
results = [pool.apply(gpRun, args=(x0, ti, tf, beta, gamma, s)) for s in range(sMax)]

# Saving number of individuals after end of gpRun for each different seed
finalS = np.array([])
finalR = np.array([])

for res in results:
	finalS = np.append(finalS, res[0])
	finalR = np.append(finalR, res[2])

# print(finalR)
# print(finalS)

histPlot(finalS, 50, 'Final number of S', 'Occurrences', 
		 title=rf'$\beta$ ={beta}, $\gamma$ = {gamma}, ($S_0, I_0, R_0$) = {x0}, {sMax} runs',
		 grid=True)
plt.savefig(f'finalS-{sMax}runs-{x0[0]}-S0.png')
plt.close()
# plt.show()

histPlot(finalR, 50, 'Final number of R', 'Occurrences', 
		 title=rf'$\beta$ ={beta}, $\gamma$ = {gamma}, ($S_0, I_0, R_0$) = {x0}, {sMax} runs',
		 grid=True)
plt.savefig(f'finalR-{sMax}runs.png')
plt.close()
# plt.show()
