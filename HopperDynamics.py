import numpy as np
import matplotlib.pyplot as plt
from MuscleTendonUnit import MTU
from scipy import signal
import copy
from scipy.integrate import odeint
from scipy.integrate import RK45

# Kinematics function for integration with RK45
def muscle(x,t,act):

	# Initialization for the derivative
	dx = np.zeros(6,)
	# Activation signal at time instant t
	a = act[int(t/MTU_unit.dt)-1]
	# Muscle dynamics
	Fmtu, Lmuscle_new, Ltendon_new=MTU_unit.MTU(a,x[2],x[4],x[5])
	# Environment feedback
	F_lever = 0.33*Fmtu
	acc_mass = (F_lever - 35.*9.81)/35.

	# Giving values to the derivatives
	dx[0] = x[1]
	dx[1] = acc_mass
	dx[2] = x[3]
	dx[3] = acc_mass*-0.33
	dx[4] = (Lmuscle_new - x[4])/MTU_unit.dt
	dx[5] = (Ltendon_new - x[5])/MTU_unit.dt
	# Sequencially, conponents of x are: length and velocity of body mass, length and velocity of MTU
	#	and length of muscle and length of tendon
	# There is no kinematics eqaution for length of muscle and tendon, derivatives are set only to update to the 
	#	new value at the next time step 
	
	return dx
	

if __name__ == '__main__':
	
	# Open data obtained from simulink model for comparison and initial condition
	with open("Validation/Data/Simulink.txt","rb") as fp:
		SimulinkData = np.genfromtxt(fp)

	# Problem setup, timestep needs to be changed to be 1/5 of the orginal: 1e-4
	MTU_unit = MTU()
	MTU_unit.dt=0.0001
	t = np.arange(0, 5.0, MTU_unit.dt)
	excitation = 0.5+0.5*signal.square(2.5*2*np.pi* (t-0.1),duty=0.1)

	# Integrate to get the activation signal a(t)
	act = np.zeros([t.shape[0],1])
	for i in range(t.shape[0]-1):
		act[i+1] = MTU_unit.dActivation(act[i],excitation[i])

	# METHOD 1--Using 1st-Order forward finite difference integration
	# Initialization 
	state = np.zeros([t.shape[0],4])
	Lmuscle=np.zeros([t.shape[0],1])
	Ltendon=np.zeros([t.shape[0],1])

	# Initial Condition
	state[0] = np.array([[SimulinkData[0,16],SimulinkData[0,17],SimulinkData[0,7],SimulinkData[0,10]]])	
	Lmuscle[0]=SimulinkData[0,5]
	Ltendon[0]=SimulinkData[0,6]
	Lmtu=SimulinkData[0,7]

	for i in range(t.shape[0]-1):
		# environment
		Fmtu, Lmuscle[i+1], Ltendon[i+1]=MTU_unit.MTU(act[i],state[i][2],Lmuscle[i],Ltendon[i])
		F_lever = 0.33*Fmtu
		acc_mass = (F_lever - 35.*9.81)/35.
		# 1st order integration
		state[i+1][0]=state[i][0]+MTU_unit.dt*state[i][1]
		state[i+1][1]=state[i][1]+MTU_unit.dt*(acc_mass)
		state[i+1][2]=state[i][2]+MTU_unit.dt*state[i][3]
		state[i+1][3]=state[i][3]+MTU_unit.dt*(-0.33*acc_mass)

	# METHOD 2--Using RK ODE solver
	# Using ODE solver
	xinitial=np.array([SimulinkData[0,16],SimulinkData[0,17],SimulinkData[0,7],SimulinkData[0,10],SimulinkData[0,5],SimulinkData[0,6]])	
	y = odeint(muscle, xinitial, t,args=(act,))

	# METHOD 3--Using RK 45
	yy=np.zeros([t.shape[0],6])
	yy[0]=xinitial
	for i in range(t.shape[0]-1):
		# RK substeps
		dy1=MTU_unit.dt*muscle(yy[i],t[i],act)
		dy2=MTU_unit.dt*muscle(yy[i]+0.5*dy1,t[i]+0.5*MTU_unit.dt,act)
		dy3=MTU_unit.dt*muscle(yy[i]+0.5*dy2,t[i]+0.5*MTU_unit.dt,act)
		dy4=MTU_unit.dt*muscle(yy[i]+dy3,t[i]+MTU_unit.dt,act)
		# RK45 Integration
		yy[i+1]=yy[i]+1.0/6.0*(dy1+2*dy2+2*dy3+dy4)



	# Accuracy Method 2 > Method 3 > Method 1
	# Plotting Results Only Method 2

	# For plotting time series from Simulink data
	tt = np.arange(0, 5.0, 0.0005)
	index = np.arange(0, 10000, 1)

	plt.figure(1)
	# plt.plot(t,state[:,0],"k")
	# plt.plot(t,yy[:,0],"r")
	plt.plot(t,y[:,0],"y")
	plt.plot(tt,SimulinkData[index,16],'b:')
	plt.legend(['Python simulation','Simulink Model'])
	plt.xlabel('time (s)')
	plt.ylabel('Hop Height (m)')

	plt.figure(2)
	# plt.plot(t,state[:,1],"k")
	# plt.plot(t,yy[:,1],"r")
	plt.plot(t,y[:,1],"y")
	plt.plot(tt,SimulinkData[index,17],'b:')
	plt.legend(['Python simulation','Simulink Model'])
	plt.xlabel('time (s)')
	plt.ylabel('Hop Velocity (m/s)')

	plt.figure(3)
	# plt.plot(t,state[:,2],'k')
	# plt.plot(t,yy[:,2],'r')
	plt.plot(t,y[:,2],'y')
	plt.plot(tt,SimulinkData[index,7],'b:')
	plt.legend(['Python simulation','Simulink Model'])
	plt.xlabel('time (s)')
	plt.ylabel('MTU Length (m)')

	plt.figure(4)
	# plt.plot(t,state[:,3],'k')
	# plt.plot(t,yy[:,3],'r')
	plt.plot(t,y[:,3],'y')
	plt.plot(tt,SimulinkData[index,10],'b:')
	plt.legend(['Python simulation','Simulink Model'])
	plt.xlabel('time (s)')
	plt.ylabel('MTU Velocity (m/s)')
	

	plt.figure(5)
	# plt.plot(t,Lmuscle,'k')
	# plt.plot(t,yy[:,4],'r')
	plt.plot(t,y[:,4],'y')
	plt.plot(tt,SimulinkData[index,5],'b:')
	plt.legend(['Python simulation','Simulink Model'])
	plt.xlabel('time (s)')
	plt.ylabel('Muscle Length (m)')

	plt.figure(6)
	# plt.plot(t,Ltendon,'k')
	# plt.plot(t,yy[:,5],'r')
	plt.plot(t,y[:,5],'y')
	plt.plot(tt,SimulinkData[index,6],'b:')
	plt.legend(['Python simulation','Simulink Model'])
	plt.xlabel('time (s)')
	plt.ylabel('Tendon Length (m)')

	plt.show()







		








