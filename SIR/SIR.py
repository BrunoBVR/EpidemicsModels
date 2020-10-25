# SIR model - OO approach
# Bruno Vieira Ribeiro

import numpy as np
import matplotlib.pyplot as plt

class SIR:
	''' This class defines the SIR model system.
	There are 2 parameters in this class defining the dynamics:

	Attributes:
		beta		transmission rate
		gamma		probability per unit time to become removed
	'''
	def __init__(self, beta, gamma):
		self.beta = beta
		self.gamma = gamma

	def __call__(self, x, t):
		y = np.zeros(len(x))
		N = np.sum(x)
		y[0] = -self.beta*x[0]*x[1]/N
		y[1] = self.beta*x[0]*x[1]/N - self.gamma*x[1]
		y[2] = self.gamma*x[1]

		return y

class ExEuler:
	'''This class defines the Explicit Euler scheme
	for numerical resolution of system of differential
	equations.
	'''
	def __init__(self, f):
		self.f = f

	def iterate(self, x0, t, dt):
		return x0+dt*self.f(x0,t)

class RK2:
	'''This class defines the 2nd order Runge-Kutta scheme
	for numerical resolution of system of differential
	equations.
	'''
	def __init__(self, f):
		self.f = f

	def iterate(self, x0, t, dt):
		return x0+dt*self.f(x0+dt/2*self.f(x0,t),t+dt/2)


class Integrator:
	'''This class defines the Integration  
	of a differential equation between ti and tf
	with N discretization steps and x0 as an initial condition

	'''
	def __init__(self, method, x0, ti, tf, N):
		self.x0 = x0
		self.ti = ti
		self.tf = tf
		self.dt = (tf - ti)/(N)

		self.F = method

	def getIntegrationTime(self):
		return np.arange(self.ti, self.tf+self.dt, self.dt)

	def integrate(self):
		x = np.array([self.x0])
		for t in np.arange(self.ti, self.tf, self.dt):
			x = np.append(x, [self.F.iterate(x[-1,:], t, self.dt)], axis=0)
		return x

def simplePlot(x,y,legend,xl,yl,title='',grid=False):
	plt.xlabel(xl)
	plt.ylabel(yl)
	plt.plot(x,y,label=legend)
	plt.legend(loc=2,prop={'size':20})
	
	if title != '':
		plt.title(title)

	if grid:
		plt.grid(linestyle='--')


x0 = np.array([100,1,0])
ti = 0
tf = 100
beta = 0.4
gamma = 0.2

N = 1000

runE = Integrator(ExEuler(SIR(beta, gamma)), x0, ti, tf, N)

time = runE.getIntegrationTime()
S = runE.integrate()[:,0]
I = runE.integrate()[:,1]
R = runE.integrate()[:,2]

runRK = Integrator(RK2(SIR(beta, gamma)), x0, ti, tf, N)

time = runRK.getIntegrationTime()
S_RK = runRK.integrate()[:,0]
I_RK = runRK.integrate()[:,1]
R_RK = runRK.integrate()[:,2]

simplePlot(time,S,'S','time','')
simplePlot(time,I,'I','','')
simplePlot(time,R,'R','','','SIR - Euler',grid=True)


plt.show()

simplePlot(time,S_RK,'S','time','')
simplePlot(time,I_RK,'I','','')
simplePlot(time,R_RK,'R','','','SIR - RK2',grid=True)

plt.show()
