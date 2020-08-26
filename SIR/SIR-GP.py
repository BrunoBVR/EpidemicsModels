import numpy as np
import math
import matplotlib.pyplot as plt

def simplePlot(x,y,legend,xl,yl,title='',grid=False):
	plt.xlabel(xl)
	plt.ylabel(yl)
	plt.plot(x,y,label=legend)
	if legend != '':
		plt.legend(loc='best',prop={'size':10})
	
	if title != '':
		plt.title(title)

	if grid:
		plt.grid(linestyle='--')

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
	S = np.array(x0[0])
	I = np.array(x0[1])
	R = np.array(x0[2])

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

		time = np.append(time, t)
		S = np.append(S, x[0])
		I = np.append(I, x[1])
		R = np.append(R, x[2])

	return time, S, I, R
################# End of functions definition #################

# Main body of algorithm

# Setting initial population, time and parameters
x0 = (10000,10,0)		# Using tuple to keep constant for each new seed

ti = 0
tf = 100
beta = 0.4
gamma = 0.2

# # Saving number of individuals after end of gpRun for each different seed
# finalS = np.array([])
# finalR = np.array([])

# Number of different seed runs
sMax = 1
for i in range (sMax):
	seed = i*13
	time, S, I, R = gpRun(x0, ti, tf, beta, gamma, seed)

	# finalS = np.append(finalS, S[-1])
	# finalR = np.append(finalR, R[-1])

	simplePlot(time,S,'','time','',title='SIR - Gillespie',grid=True)
	simplePlot(time,I,'','','')
	simplePlot(time,R,'','','')

plt.show()

# print(finalS)
# print(finalR)

# histPlot(finalS, 20, 'Final number of S', 'Occurrences', 
# 		 title=rf'$\beta$ ={beta}, $\gamma$ = {gamma}, ($S_0, I_0, R_0$) = {x0}, {sMax} runs',
# 		 grid=True)
# plt.show()

# histPlot(finalR, 20, 'Final number of R', 'Occurrences', 
# 		 title=rf'$\beta$ ={beta}, $\gamma$ = {gamma}, ($S_0, I_0, R_0$) = {x0}, {sMax} runs',
# 		 grid=True)
# plt.show()
