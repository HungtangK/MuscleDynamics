import numpy as np
import matplotlib.pyplot as plt
from Parameters import *
from MuscleTendonUnit import *
from scipy.integrate import odeint
import matplotlib.animation as animation
import gym
from gym import error, spaces
import time

class TwoDofArmEnv(gym.Env):
	def __init__(self,sim_length=0.1,traj_track=False,exo=False,exo_gain=0.,delay=0.):
		
		self.sim_length = sim_length

		# Init state of the Arm 
		self.InitState = np.array([0.,0.,0.,0.])
		# Timestep
		self.dt = 0.0005
		self.t = 0.
		self.exo = exo
		self.exo_gain = exo_gain
		self.time_delay = delay
		self.traj_track = traj_track
		if self.traj_track:
			with open("Circle_new.txt","rb") as fp:
				self.Circle_points = np.loadtxt(fp)
			with open("JAngles_new.txt","rb") as fp:
				self.target_angles = np.loadtxt(fp)
	
		self.MTU_ad = MTU(L0=lm0_ad,F_max=Fmax_ad,Vm_max=Vmax_ad,Lt_slack=lt0_ad)
		self.MTU_pd = MTU(L0=lm0_pd,F_max=Fmax_pd,Vm_max=Vmax_pd,Lt_slack=lt0_pd)
		self.MTU_bb = MTU(L0=lm0_bb,F_max=Fmax_bb,Vm_max=Vmax_bb,Lt_slack=lt0_bb)
		self.MTU_tb = MTU(L0=lm0_tb,F_max=Fmax_tb,Vm_max=Vmax_tb,Lt_slack=lt0_tb)
		self.obs_dim = 12
		self.act_dim = 4

		nvec = [10]*4
		control_bounds = np.array([[0,0,0,0],[1,1,1,1]])
		self.action_space = spaces.MultiDiscrete(nvec)
		#self.action_space = spaces.Box(control_bounds[0],control_bounds[1])
		high = np.inf*np.ones(self.obs_dim)
		low = -high
		self.observation_space = spaces.Box(low, high)


	def activation_bb(self,t):
		index = int(t/self.dt)
		return self.act_bb[index]

	def activation_ad(self,t):
		index = int(t/self.dt)
		return self.act_ad[index]

	def activation_pd(self,t):
		index = int(t/self.dt)
		return self.act_pd[index]
		

	def activation_tb(self,t):
		index = int(t/self.dt)
		return self.act_tb[index]

	def TargetAngles(self,):
		if self.t-self.time_delay < 0:
			index = 0
		else:
			index = int(self.t/0.005)
		
		return self.target_angles[index,:].reshape(2,1)



	def MuscleArmDynamics(self,x,t):

		# 12 dimensional vector
		#x[0] - shoulder angle
		#x[1] - shoulder velocity
		#x[2] - elbow angle
		#x[3] - elbow velocity
		#x[4] - lm_ad
		#x[5] - lm_pd
		#x[6] - lm_bb
		#x[7] - lm_tb
		#x[8] - lt_ad
		#x[9] - lt_pd
		#x[10] - lt_bb
		#x[11] - lt_tb

		# Activation levels
		a_bb = self.activation_bb(t)
		a_tb = self.activation_tb(t)
		a_ad = self.activation_ad(t)
		a_pd = self.activation_pd(t)

		# MTU length from the joint angles
		Lmtu_ad=ADeltoid_MuscleLength(x[0])
		Lmtu_pd=PDeltoid_MuscleLength(x[0])
		Lmtu_bb=Bicep_MuscleLength(x[2])
		Lmtu_tb=Tricep_MuscleLength(x[2])

		# MTU vel from the joint angles
		Vmtu_ad=(ADeltoid_MuscleLength(x[0]+x[1]*self.dt)-Lmtu_ad)/self.dt
		Vmtu_pd=(PDeltoid_MuscleLength(x[0]+x[1]*self.dt)-Lmtu_pd)/self.dt
		Vmtu_bb=(Bicep_MuscleLength(x[2]+x[3]*self.dt)-Lmtu_bb)/self.dt
		Vmtu_tb=(Tricep_MuscleLength(x[2]+x[3]*self.dt)-Lmtu_tb)/self.dt

		# Muscle Dynamics Block
		F_ad, Lmuscle_new_ad, Ltendon_new_ad=self.MTU_ad.MTU2(a_ad,Lmtu_ad,Vmtu_ad,x[8])
		F_pd, Lmuscle_new_pd, Ltendon_new_pd=self.MTU_pd.MTU2(a_pd,Lmtu_pd,Vmtu_pd,x[9])
		F_bb, Lmuscle_new_bb, Ltendon_new_bb=self.MTU_bb.MTU2(a_bb,Lmtu_bb,Vmtu_bb,x[10])
		F_tb, Lmuscle_new_tb, Ltendon_new_tb=self.MTU_tb.MTU2(a_tb,Lmtu_tb,Vmtu_tb,x[11])

		# Exo
		K_exo = np.array([[0,0.],[0.,self.exo_gain]])
		TargetAngles = self.TargetAngles()-np.array([[x[0]],[x[2]]])
		T_exo = np.dot(K_exo,TargetAngles)

		# Environment Feedback Block
		Torques = np.array([[ADeltoid_MomentArm(x[0])*F_ad + PDeltoid_MomentArm(x[0])*F_pd],\
							[Bicep_MomentArm(x[2])*F_bb + Tricep_MomentArm(x[2])*F_tb]])\
							+T_exo
		acc=TwoLinkDynamics(x[0],x[2],x[1],x[3],Torques)

		# Returning Derivatives 
		dx = np.zeros(12,)
		dx[0] = x[1]
		dx[1] = acc[0]
		dx[2] = x[3]
		dx[3] = acc[1]
		dx[4] = (Lmuscle_new_ad - x[4])/self.dt
		dx[5] = (Lmuscle_new_pd - x[5])/self.dt
		dx[6] = (Lmuscle_new_bb - x[6])/self.dt
		dx[7] = (Lmuscle_new_tb - x[7])/self.dt
		dx[8] = (Ltendon_new_pd - x[8])/self.dt
		dx[9] = (Ltendon_new_ad - x[9])/self.dt
		dx[10] = (Ltendon_new_bb - x[10])/self.dt
		dx[11] = (Ltendon_new_tb - x[11])/self.dt

		return dx


	def InverseKinematics(self,x,y):
		a1 = 0.3
		a2 = 0.5
		q2 = np.arccos((x**2 + y**2 - a1**2 - a2**2)/(2*a1*a2))
		q1 = np.arctan(y/x) - np.arctan(a2*np.sin(q2)/(a1 + a2*np.cos(q2)))
		return q1,q2


	def step(self,a):
		# this is where we take a simulation step - baselines calls this repeatedly
		# a - set up the square wave for 0.2 seconds for the activation
		# use this to simulate 0.2 seconds with odeint
		# get  s,a,r,s' after computing reward for the next sim-step
		# return those values

		sim_length = self.sim_length
		sim_nsteps = int(sim_length/self.dt)+1000
		#if self.t < 0.5:
		#	a[1] = 0.
		#elif self.t >= 0.5:
		#	a[3] = 0
		'''
		if a[0] > a[2]:#biceps
			a[2] = 0.
			a[0]-=a[2]
		elif a[0] < a[2]:
			a[0] = 0.
			a[2]-=a[0]

		if a[1] > a[3]:
			a[3] = 0.
			a[1]-=a[3]
		elif a[1] < a[3]:
			a[1] = 0.
			a[3]-=a[1]
		'''
		self.act_bb = abs(a[0])*np.ones(sim_nsteps)*0.1#1#5#1#05
		self.act_ad = abs(a[1])*np.ones(sim_nsteps)*0.1#1#5#1#05#25
		self.act_tb = abs(a[2])*np.ones(sim_nsteps)*0.1#1#5#1#05#25
		self.act_pd = abs(a[3])*np.ones(sim_nsteps)*0.1#1#5#1#05#25

		t = np.arange(0,sim_length,self.dt)
		
		state = np.concatenate((self.ArmState,self.Cur_lm,self.Cur_vm))
		begin = time.time()
		data = odeint(self.MuscleArmDynamics,state,t)
		end = time.time()

		self.ArmState = data[-1,:4]
		self.Cur_lm = data[-1,4:8]
		self.Cur_vm = data[-1,8:]

		pos = data
		x = np.cumsum([0,0.3*np.sin(pos[-1,0]),0.51*np.sin(pos[-1,0]+pos[-1,2])])
		y = np.cumsum([0,-0.3*np.cos(pos[-1,0]),-0.51*np.cos(pos[-1,0]+pos[-1,2])])

		done = False

		# Just testing the dynamics
		angle1 = pos[-1,0]
		angle2 = pos[-1,2]

		point_on_circle = int(self.t/0.005)
		angle_space = int(self.t/0.005)

		joint_error = (angle1 - self.target_angles[angle_space,0])**2 + (angle2 - self.target_angles[angle_space,1])**2
		joint_reward = np.exp(-10*joint_error)

		diff = ((x[2] - self.Circle_points[point_on_circle,0])**2 + (y[2] - self.Circle_points[point_on_circle,1])**2)
		ee_reward = np.exp(-50*diff)

		if self.traj_track:
			reward = joint_reward + ee_reward - 1e-4*(sum(a**2))
		else:
			reward = 1.0/(0.001 +  (x[2] - 0.6)**2 
				+ (y[2] - 0.6)**2)
		
		angle1 = pos[-1,0]
		angle2 = pos[-1,0] + pos[-1,2]

		if pos[-1,0] < -np.pi/3 or pos[-1,0] > np.pi/2 or pos[-1,2] < -np.pi/100 or angle2 > 3*np.pi/2:
			done = True
			reward = 0.

		if reward < 1.6:
			done = True
			reward = 0.
		
		self.t+=sim_length
		if self.t >= 0.5:
			done = True

		return np.concatenate((self.ArmState,self.Cur_lm,self.Cur_vm)),reward,done,{'data':data,'a':a}



	def Calculate_Data(self,data,actions):
		# This function reconstructs the muscle forces, etc given a,Lm and Vm
		# needed to compute Metabolic Cost etc..

		# inpute data - 2Darray of states
		torques = []
		for i in range(len(data)):
			# compute the muscle force for the given lengths etc..
			theta2 = data[i,2]
			theta1 = data[i,0]
			a_bb = actions[i,0]
			a_ad = actions[i,1]
			a_tb = actions[i,2]
			a_pd = actions[i,3]

			lm_bb = data[i,4]
			lm_tb = data[i,5]
			lm_ad = data[i,6]
			lm_pd = data[i,7]

			vm_bb = data[i,8]
			vm_tb = data[i,9]
			vm_ad = data[i,10]
			vm_pd = data[i,11]

			fl_bb = np.exp(-((lm_bb/lm0_bb) - 1)**2/0.45)
			fl_tb = np.exp(-((lm_tb/lm0_tb) - 1)**2/0.45)
			fl_ad = np.exp(-((lm_ad/lm0_ad) - 1)**2/0.45)
			fl_pd = np.exp(-((lm_pd/lm0_pd) - 1)**2/0.45)
			
			F_a_bb= self.MTU_unit_bb.MuscleDynamics(a_bb,lm_bb,vm_bb)
			F_p_bb = self.MTU_unit_bb.PassiveMuscleForce(lm_bb)
			F_m_bb = F_a_bb + F_p_bb
			ema_bb = Bicep_MomentArm(theta2)
			Torque_bb = ema_bb*F_m_bb

			#Tricep Muscle Dynamics
			
			F_a_tb = self.MTU_unit_tb.MuscleDynamics(a_tb,lm_tb,vm_tb)
			F_p_tb = self.MTU_unit_tb.PassiveMuscleForce(lm_tb)
			F_m_tb = F_a_tb + F_p_tb
			ema_tb = Tricep_MomentArm(theta2)
			Torque_tb = ema_tb*F_m_tb
			

			# Anterior Deltoid Muscle Dynamics	
			F_a_ad = self.MTU_unit_ad.MuscleDynamics(a_ad,lm_ad,vm_ad)
			F_p_ad = self.MTU_unit_ad.PassiveMuscleForce(lm_ad)
			F_m_ad = F_a_ad + F_p_ad
			ema_ad = ADeltoid_MomentArm(theta1)
			Torque_ad = ema_ad*F_m_ad
			
			# Posterios Deltoid Muscle Dynamics
			lmtu_old_pd = PDeltoid_MuscleLength(theta1)
			F_a_pd = self.MTU_unit_pd.MuscleDynamics(a_pd,lm_pd,vm_pd)
			F_p_pd = self.MTU_unit_pd.PassiveMuscleForce(lm_pd)
			F_m_pd = F_a_pd + F_p_pd
			ema_pd = PDeltoid_MomentArm(theta1)
			Torque_pd = ema_pd*F_m_pd
			
			Torques = np.array([Torque_bb,Torque_ad,Torque_tb,Torque_pd])
			torques.append(Torques)


		return torques

	def reset(self,):
		# Uniform Initial state distribution
		x = 0.6
		y = -0.5
		
		q1,q2 = self.InverseKinematics(x,y)

		self.InitState += np.random.uniform(low=-0.005,high=0.005,size=4)
		self.InitState[0] =  self.target_angles[0,0] #np.pi/2 + q1
		self.InitState[2] = self.target_angles[0,1]# q2
		#print(self.Circle_points[0,0])
		#print(self.Circle_points[0,1])
		
		#print("init state",q1+np.pi/2)
		#print("inid",q2)
		self.ArmState = np.copy(self.InitState)
		#print("armstate",self.ArmState)
		self.lm0 = np.array([lm0_bb,lm0_tb,lm0_ad,lm0_pd])
		self.Cur_lm = np.array([lm0_bb,lm0_tb,lm0_ad,lm0_pd])
		self.vm0 = np.zeros(4,)
		self.Cur_vm = np.zeros(4,)

		#print(self.Cur_lm)
		self.t = 0.
		state = np.concatenate((self.InitState,self.lm0,self.vm0))
		return state


















