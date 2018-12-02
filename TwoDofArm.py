import numpy as np
import matplotlib.pyplot as plt
from Parameters import *
from MuscleTendonUnit import *
from scipy.integrate import odeint
import matplotlib.animation as animation
import gym


class TwoDofArmEnv(gym.Env):
	def __init__(self,ActiveMuscles='antagonistic',actionParameterization = True,sim_length=0.1):
		
		self.sim_length = sim_length
		print(m1)
		## ARM PHYSICAL DIMENSIONS - taken from Lan et al. 2008
		self.m1 = 2.1 # Upper Arm mass
		self.m2 = 1.0 # Forearm Mass
		# Lengths
		self.l1 = 0.3
		self.l2 = 0.5
		#lengths to COM
		self.lc1 = self.l1/2.0
		self.lc2 = self.l2/2.0
		# Moment of Inertias
		self.I1 = self.m1*self.l1**2
		self.I2 = self.m2*self.l2**2

		# Init state of the Arm 
		self.InitState = np.array([0.,0.,0.,0.])
		# Gravity
		self.g = -9.81
		# Timestep
		self.dt = 0.0005
		# MTU - need to define 4 of these
		if ActiveMuscles == 'antagonistic':
			self.antagonistic = True
		else:
			self.antagonistic = False

		self.Fmax_bicep = 1200
		self.Fmax_tricep = 450
		self.Fmax_ad = 1500
		self.Fmax_pd = 750

		# Initial muscle lengths
		self.lm0_bb = lmtu0_bb
		self.lm0_tb = lmtu0_tb
		self.lm0_ad = lmtu0_ad
		self.lm0_pd = lmtu0_pd
		#print("lm0_ad",lm0_ad)
		self.vm0_bb = 0.
		self.vm0_tb = 0.
		self.vm0_ad = 0.
		self.vm0_pd = 0.


		self.MTU_unit_bb = MTU(L0=self.lm0_bb,F_max=self.Fmax_bicep,Vm_max=-1.0,Lt_slack=lmtu0_bb)
		self.MTU_unit_ad = MTU(L0=self.lm0_ad,F_max=self.Fmax_ad,Vm_max=-1.0,Lt_slack=lmtu0_ad)
		if self.antagonistic:
			self.MTU_unit_pd = MTU(L0=self.lm0_pd,F_max=self.Fmax_pd,Vm_max=-1.0,Lt_slack=lmtu0_tb)
			self.MTU_unit_tb = MTU(L0=self.lm0_tb,F_max=self.Fmax_tb,Vm_max=-1.0,Lt_slack=lmtu0_pd)




	

	def activation_bb(self,t):
		#print("time",t)
		index = int(t/self.dt)

		return self.act_bb[index]

	def activation_ad(self,t):
		index = int(t/self.dt)
		return self.act_ad[index]

	def activation_pd(self,t):
		return None

	def activation_tb(self,t):
		return None


	def MuscleArmDynamics(self,x,t):

		# 12 dimensional vector
		#x[0] - shoulder angle
		#x[1] - shoulder velocity
		#x[2] - elbow angle
		#x[3] - elbow velocity
		#x[4] - lm_ad
		#x[5] - Vm_ad
		#x[6] - lm_pd
		#x[7] - Vm_pd
		#x[8] - lm_bb
		#x[9] - Vm_bb
		#x[10] - lm_tb
		#x[11] - Vm-tb
		# if not using antagonisitc muscles then we use only Biceps and Anterior Deltoids
		################################# TO Do #################################
		# IMPLEMENT CONSTRAINTS ON THE JOINTS
		# BETTER APPROXIMATION OF SHOULDER MUSCLE LENGTHS
		#########################################################################
		theta1 = x[0]
		dtheta1 = x[1]
		theta2 = x[2]
		dtheta2 = x[3]

		if self.antagonistic:
			lm_bb = x[4]
			vm_bb = x[8]

			lm_tb = x[5]
			vm_tb = x[9]

			lm_ad=x[6]
			vm_ad = x[10]

			lm_pd=x[7]
			vm_pd =x[11]

			a_bb = self.activation_bb(t)
			a_tb = self.activation_tb(t)
			a_ad = self.activation_ad(t)
			a_pd = self.activation_pd(t)

			fl_bb = np.exp(-((lm_bb/lmtu0_bb) - 1)**2/0.45)
			fl_tb = np.exp(-((lm_tb/lmtu0_tb) - 1)**2/0.45)
			fl_ad = np.exp(-((lm_ad/lmtu0_ad) - 1)**2/0.45)
			fl_pd = np.exp(-((lm_pd/lmtu0_pd) - 1)**2/0.45)	

			# Bicep Muscle Dynamics
			F_a_bb,_,_ = self.MTU_unit_bb.MuscleDynamics(a_bb,lm_bb,vm_bb,fl_bb)
			F_p_bb = self.MTU_unit_bb.PassiveMuscleForce(lm_bb)
			F_m_bb = F_a_bb + F_p_bb
			ema_bb = self.Bicep_MomentArm(theta2)
			Torque_bb = ema_bb*F_m_bb
			lt_bb = self.MTU_unit_bb.TendonDynamics(F_m_bb)
			theta2_new = x[2] + x[3]*self.dt
			new_Lmtu_bb = self.Bicep_MuscleLength(theta2_new)
			lm_new_bb = new_Lmtu_bb - lt_bb
			dlm_bb = (lm_new_bb - lm_bb)/self.dt
			dvm_bb = (dlm_bb - vm_bb)/self.dt

			#Tricep Muscle Dynamics
			F_a_tb,_,_ = self.MTU_unit_tb.MuscleDynamics(a_tb,lm_tb,vm_tb,fl_tb)
			F_p_tb = self.MTU_unit_tb.PassiveMuscleForce(lm_tb)
			F_m_tb = F_a_tb + F_p_tb
			ema_tb = self.Tricep_MomentArm(theta2)
			Torque_tb = ema_tb*F_m_tb
			lt_tb = self.MTU_unit_tb.TendonDynamics(F_m_tb)
			#theta2_new = x[2] + x[3]*self.dt
			new_Lmtu_tb = self.Tricep_MuscleLength(theta2_new)
			lm_new_tb = new_Lmtu_tb - lt_tb
			dlm_tb = (lm_new_tb - lm_tb)/self.dt
			dvm_tb = (dlm_tb - vm_tb)/self.dt

			# Anterior Deltoid Muscle Dynamics
			F_a_ad,_,_ = self.MTU_unit_ad.MuscleDynamics(a_ad,lm_ad,vm_ad,fl_ad)
			F_p_ad = self.MTU_unit_ad.PassiveMuscleForce(lm_ad)
			F_m_ad = F_a_ad + F_p_ad
			ema_ad = self.Adeltoid_MomentArm(theta1)
			Torque_ad = ema_ad*F_m_ad
			lt_ad = self.MTU_unit_ad.TendonDynamics(F_m_ad)
			theta1_new = x[0] + x[1]*self.dt
			new_Lmtu_ad = self.ADeltoid_MuscleLength(theta1_new)
			lm_new_ad = new_Lmtu_ad - lt_ad
			dlm_ad = (lm_new_ad - lm_ad)/self.dt
			dvm_ad = (dlm_ad - vm_ad)/self.dt

			
			# Posterios Deltoid Muscle Dynamics
			F_a_pd,_,_ = self.MTU_unit_pd.MuscleDynamics(a_pd,lm_pd,vm_pd,fl_pd)
			F_p_pd = self.MTU_unit_pd.PassiveMuscleForce(lm_pd)
			F_m_pd = F_a_pd + F_p_pd
			ema_pd = self.PDeltoid_MomentArm(theta1)
			Torque_pd = ema_pd*F_m_pd
			lt_pd = self.MTU_unit_pd.TendonDynamics(F_m_pd)
			#theta2_new = x[2] + x[3]*self.dt
			new_Lmtu_pd = self.PDeltoid_MuscleLength(theta1_new)
			lm_new_pd = new_Lmtu_pd - lt_pd
			dlm_pd = (lm_new_pd - lm_pd)/self.dt
			dvm_pd = (dlm_pd - vm_pd)/self.dt

			Torques = np.array([[Torque_ad - Torque_pd],[Torque_bb-Torque_tb]])


		else:
			lm_bb = x[4]
			vm_bb = x[6]

			lm_ad=x[5]
			vm_ad = x[7]
			a_bb = self.activation_bb(t)
			a_ad = self.activation_ad(t)


			#debug print

			fl_bb = np.exp(-((lm_bb/lmtu0_bb) - 1)**2/0.45)
			#fl_tb = np.exp(-((lm_tb/lm0_tb) - 1)**2/0.45)
			fl_ad = np.exp(-((lm_ad/lmtu0_ad) - 1)**2/0.45)
			#fl_pd = np.exp(-((lm_pd/lm0_pd) - 1)**2/0.45)
			# Bicep Muscle Dynamics
			lmtu_old_bb = self.Bicep_MuscleLength(theta2)

			F_a_bb,_,_ = self.MTU_unit_bb.MuscleDynamics(a_bb,lm_bb,vm_bb,fl_bb)
			F_p_bb = self.MTU_unit_bb.PassiveMuscleForce(lm_bb)
			F_m_bb = F_a_bb + F_p_bb
			ema_bb = self.Bicep_MomentArm(theta2)
			Torque_bb = ema_bb*F_m_bb
			lt_bb = self.MTU_unit_bb.TendonDynamics(F_m_bb)
			theta2_new = x[2] + x[3]*self.dt
			new_Lmtu_bb = self.Bicep_MuscleLength(theta2_new)
			lm_new_bb = new_Lmtu_bb# - lt_bb
			dlm_bb = (lm_new_bb - lmtu_old_bb)/self.dt #+0.245
			dvm_bb = (dlm_bb - vm_bb)/self.dt
			#debug print
			#print("new_Lmtu_bb",new_Lmtu_bb)
			#print("lt_bb",lt_bb)
			#print("dlm_bb",dlm_bb)
			#print("a_bb",a_bb)
			#print("F_m",F_m_bb)
			#print("ema_bb",ema_bb)
			#print("torqu",Torque_bb)

			# Anterior Deltoid Muscle Dynamics
			lmtu_old_ad = self.ADeltoid_MuscleLength(theta1)
			F_a_ad,_,_ = self.MTU_unit_ad.MuscleDynamics(a_ad,lm_ad,vm_ad,fl_ad)
			F_p_ad = self.MTU_unit_ad.PassiveMuscleForce(lm_ad)
			F_m_ad = F_a_ad + F_p_ad
			ema_ad = self.ADeltoid_MomentArm(theta1)
			
			Torque_ad = ema_ad*F_m_ad
			lt_ad = self.MTU_unit_ad.TendonDynamics(F_m_ad)
			theta1_new = x[0] + x[1]*self.dt
			new_Lmtu_ad = self.ADeltoid_MuscleLength(theta1_new)
			lm_new_ad = new_Lmtu_ad# - lt_ad
			dlm_ad = (lm_new_ad - lmtu_old_ad)/self.dt #+0.105
			dvm_ad = (dlm_ad - vm_ad)/self.dt
			
			Torques = np.array([[Torque_ad],[Torque_bb]])
			#debug print
			#print("***************************************")
			
			#print("new_Lmtu_ad",new_Lmtu_ad)
			##print("dlm_bb",dlm_bb)
			#print("theta1_new",theta1_new)
			#print("theta1",theta1)
			#print("lt_ad",lt_ad)
			#print("lm_ad",lm_ad)
			#print("dlm",dlm_ad)
			#print("a_ad",a_ad)
			#print("F_m_ad",F_m_ad)
			#print("ema_ad",ema_ad)
			#print("torqu",Torque_ad)
			#print("******************************************")
		#d = 2*r
		# state vectors - q and qdot
		qdot = np.array([[dtheta1],[dtheta2]])
		q = np.array([[theta1],[theta2]])

		# Matrices - H(q)q'' + C(q,q')q' + G(q) = Tau - following this dynamical equation
		Hq = np.array([[self.I1 + self.I2 + self.m2*self.l1**2 + 2*self.m2*self.l1*self.lc2*np.cos(theta2),
			self.I2+self.m2*self.l1*self.lc2*np.cos(theta2)],
			[self.I2+self.m2*self.l1*self.lc2*np.cos(theta2),
			self.I2]])
		Cq = np.array([[-2*self.m2*self.l1*self.lc2*np.sin(theta2)*dtheta2,
			-self.m2*self.l1*self.lc2*np.sin(theta2)*dtheta2],
			[self.m2*self.l1*self.lc2*np.sin(theta2)*dtheta1,
			0]])

		Gq = np.array([[(self.m1*self.lc1 + self.m2*self.l1)*self.g*np.sin(theta1) + self.m2*self.g*self.l2*np.sin(theta1+theta2)],
			[self.m2*self.g*self.l2*np.sin(theta1+theta2)]])

		Damping = np.array([[2.10,0],[0,2.10]])
		#Torques = np.zeros(2,) # For Time Being
		acc = np.dot(np.linalg.inv(Hq),(Torques+-np.dot(Cq,qdot) + Gq - np.dot(Damping,qdot)))
		#print(np.dot(Damping,qdot).shape)

		#print(acc)

		# return derivatives
		if self.antagonistic:
			dx = np.zeros(12,)

			dx[0] = x[1]
			dx[1] = acc[0]
			dx[2] = x[3]
			dx[3] = acc[1]
			dx[4] = dlm_bb
			dx[5] = dlm_tb
			dx[6] = dlm_ab
			dx[7] = dlm_pb
			dx[8] = dvm_bb
			dx[9] = dvm_tb
			dx[10] = dvm_ad
			dx[11] = dvm_pd

		else:
			dx = np.zeros(8,)

			dx[0] = x[1]
			dx[1] = acc[0]
			dx[2] = x[3]
			dx[3] = acc[1]
			dx[4] = dlm_bb
			dx[5] = dlm_ad
			dx[6] = dvm_bb
			dx[7] = dvm_ad

		return dx



	def step(self,a):
		# this is where we take a simulation step - baselines calls this repeatedly
		# a - set up the square wave for 0.2 seconds for the activation
		# use this to simulate 0.2 seconds with odeint
		# get  s,a,r,s' after computing reward for the next sim-step
		# return those values

		# create square waves as excitation with parameters in a
		# 

		print(a[0])
		sim_length = self.sim_length
		sim_nsteps = int(sim_length/self.dt)+100
		if self.antagonistic:
			self.act_bb = a[0]*np.ones(sim_nsteps)
			self.act_ad = a[1]*np.ones(sim_nsteps)
			self.act_tb = a[2]*np.ones(sim_nsteps)
			self.act_pd = a[3]*np.ones(sim_nsteps)

		else:
			self.act_bb =a[0]*np.ones(sim_nsteps,)
			self.act_ad = a[1]*np.ones(sim_nsteps,)

		t = np.arange(0,sim_length,self.dt)
		
		state = np.concatenate((self.ArmState,self.Cur_lm,self.Cur_vm))

		data = odeint(self.MuscleArmDynamics,state,t)

		if self.antagonistic:
			self.ArmState = data[-1,:4]
			self.Cur_lm = data[-1,4:8]
			self.Cur_vm = data[-1,8:]

		else:
			self.ArmState = data[-1,:4]
			self.Cur_lm = data[-1,4:6]
			self.Cur_vm = data[-1,6:]

		#print("data",data.shape)
		done = False
		# Just testing the dynamics
		reward = 0.
		# GYM STYLE - s,reward,done,{extra params}
		return np.concatenate((self.ArmState,self.Cur_lm,self.Cur_vm)),reward,done,{'data':data}





	def Calculate_Data(self,):
		# This function reconstructs the muscle forces, etc given a,Lm and Vm
		# needed to compute Metabolic Cost etc..

		return None

	def reset(self,):
		# Uniform Initial state distribution
		self.InitState += np.random.uniform(low=-0.005,high=0.005,size=4)
		self.ArmState = np.copy(self.InitState)
		if self.antagonistic:
			self.lm0 = np.array([self.lm0_bb,self.lm0_tb,self.lm0_ad,self.lm0_pd])
			self.Cur_lm = np.array([self.lm0_bb,self.lm0_tb,self.lm0_ad,self.lm0_pd])
			self.vm0 = np.zeros(4,)
			self.Cur_vm = np.zeros(4,)
		else:
			self.lm0 = np.array([self.lm0_bb,self.lm0_ad])
			self.Cur_lm = np.array([self.lm0_bb,self.lm0_ad])
			self.vm0 = np.zeros(2,)
			self.Cur_vm = np.zeros(2,)
		#print(self.Cur_lm)

		state = np.concatenate((self.InitState,self.lm0,self.vm0))
		return state



















