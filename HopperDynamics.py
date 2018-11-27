import numpy as np
import matplotlib.pyplot as plt
from MuscleTendonUnit import MTU
from scipy import signal
import copy
from scipy.integrate import odeint
#from muscles import *

def muscle(x,t,act):
	dx = np.zeros(4,)
	
	L_m = x[2]#lm_init
	V_m = x[3]
	a = act[int(t/0.0005)]
	F_a,fv,fl = MTU_unit.MuscleDynamics(a,L_m,V_m)
	F_p = MTU_unit.PassiveMuscleForce(L_m)
	F_m = F_a + F_p
	F_lever = 0.33*F_m

	dl_mt = -1*0.33*x[0]
	F = F_lever
	acc_mass = (F - 35.*9.81)/35.
	
	dx[0] = x[1]
	dx[1] = acc_mass
	L_mtu = dl_mt + 0.292


	L_t = MTU_unit.TendonDynamics(F_m)
	L_mnew = MTU_unit.MuscleLength(L_mtu,L_t)
	

	dx[2] = (L_mnew - x[2])/0.0005
	dx[3] = (dx[2] - x[3])/0.0005
	
	
	return dx
	





if __name__ == '__main__':

	# with open("Data/dataMuscle.txt","rb") as fp:
	# 	muscledata = np.genfromtxt(fp,delimiter=',')

	with open("Data/Simulink.txt","rb") as fp:
		SimulinkData = np.genfromtxt(fp)
	MTU_unit = MTU()
	t = np.linspace(0, 10,20000, endpoint=False)
	excitation = 0.5+0.5*signal.square(2.5*2*np.pi* (t-0.1),duty=0.1)

	a = 0#np.zeros(excitation.shape[0])
	act = []
	for i in range(excitation.shape[0]):
		a = MTU_unit.dActivation(a,excitation[i])
		act.append(a)

	
	
	state = np.array([[0.05,0,0.052,0]])
	pos = []
	t0 = 0.
	dt = 0.0005
		
	t = np.arange(t0, 5.0, dt)
	y = odeint(muscle, state[0], t,args=(act,))
	pos.append(y[0])



	plt.plot(t,y[:,0],"r")
	plt.plot(t,SimulinkData[:10000,16],'b')
	plt.title("Virtual Hopper Dynamics")
	plt.legend(['Python simulation','Simulink Model'])

	plt.xlabel(['time (s)'])
	plt.ylabel(['Hop Height (m)'])

	plt.figure()
	plt.plot(t,y[:,1],"r")
	plt.plot(t,SimulinkData[:10000,17],'b')
	plt.title("Virtual Hopper Dynamics")
	plt.legend(['Python simulation','Simulink Model'])

	plt.xlabel(['time (s)'])

	plt.figure()
	print(y[:,2])
	plt.plot(t,y[:,2],'r')
	plt.plot(t,SimulinkData[:10000,5],'b')

	plt.figure()
	print(y[:,3])
	plt.plot(t,y[:,3],'r')
	plt.plot(t,SimulinkData[:10000,8],'b')
	plt.show()







		








