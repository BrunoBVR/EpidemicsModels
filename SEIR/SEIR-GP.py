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

def rates(beta, gamma, sigma, x):
	r = np.zeros(3)
	
	r[0] = beta*x[0]*x[1]/np.sum(x)
	r[1] = sigma*x[1]
	r[2] = gamma*x[2]

	return r, np.sum(r)

def gpRun(x0, ti, tf, ts, beta, gamma, sigma, seed):

	np.random.seed(seed)

	tc = 0
	t = ti
	time = np.array(t)

	x = np.asarray(x0)
	S = np.array(x0[0])
	E = np.array(x0[1])
	I = np.array(x0[2])
	R = np.array(x0[3])

	while t <= tf:
		# Fixed point condition
		if x[2] == 0 and x[1] == 0:
			print('Fixed point met!')
			break

		r, Rn = rates(beta, gamma, sigma, x)

		choice = np.random.rand()

		if choice < r[0]/Rn:
			x[0] -= 1
			x[1] += 1

		elif r[0]/Rn <= choice < r[1]/Rn:
			x[1] -= 1
			x[2] += 1

		else:
			# Not allowinf negative number of infected
			if x[2] == 0:
				continue
			x[2] -= 1
			x[3] += 1

		dt = -math.log(np.random.rand())/Rn

		t += dt

		time = np.append(time, t)
		S = np.append(S, x[0])
		E = np.append(E, x[1])
		I = np.append(I, x[2])
		R = np.append(R, x[3])

		tc += dt
		if tc > ts:
			print(f"time: {t},  N = {np.sum(x)}")
			tc = 0

	return time, S, E, I, R
################# End of functions definition #################

# Main body of algorithm

# Setting initial population, time and parameters
x0 = (10000,3,1,0)		# Using tuple to keep constant for each new seed

# Setting disease parameters for COVID-19
t_inc = 5
t_inf = 1.61
R0 = 2.74

beta = R0/t_inf
gamma = 1/t_inf
sigma = 1/t_inc


ti = 0
tf = 1000
ts = 10

seed = 15

time, S, E, I, R = gpRun(x0, ti, tf, ts, beta, gamma, sigma, seed)


simplePlot(time,S,'S','time','',title='SIR - Gillespie',grid=True)
simplePlot(time,E,'E','','')
simplePlot(time,I,'I','','')
simplePlot(time,R,'R','','')

plt.show()