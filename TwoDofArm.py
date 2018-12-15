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
	def __init__(self,ActiveMuscles='antagonistic',actionParameterization = True,sim_length=0.1,traj_track=False,exo=False,exo_gain=0.,delay=0.):
		
		self.sim_length = sim_length
		## ARM PHYSICAL DIMENSIONS - taken from Lan et al. 2008
		self.m1 = 2.1 # Upper Arm mass
		self.m2 = 1.0 # Forearm Mass
		# Lengths
		self.l1 = 0.3
		self.l2 = 0.51
		#lengths to COM
		self.lc1 = self.l1/2.0
		self.lc2 = self.l2/2.0
		# Moment of Inertias
		self.I1 = self.m1*self.l1**2/3.
		self.I2 = self.m2*self.l2**2/3.

		# Init state of the Arm 
		self.InitState = np.array([0.,0.,0.,0.])
		# Gravity
		self.g = -9.81
		# Timestep
		self.dt = 0.0005
		self.t = 0.
		self.exo = exo
		self.exo_gain = exo_gain
		self.time_delay = delay
		# MTU - need to define 4 of these
		if ActiveMuscles == 'antagonistic':
			self.antagonistic = True
		else:
			self.antagonistic = False

		self.Fmax_bicep = 1063
		self.Fmax_tricep = 1098#2004.3
		self.Fmax_ad = 1700.65 #2600
		self.Fmax_pd = 1354.65 #2200

		# Initial muscle lengths
		self.lm0_bb = lm0_bb
		self.lm0_tb = lm0_tb
		self.lm0_ad = lm0_ad
		self.lm0_pd = lm0_pd


		#print("lm0_ad",lm0_ad)
		self.vm0_bb = 0.
		self.vm0_tb = 0.
		self.vm0_ad = 0.
		self.vm0_pd = 0.
		self.traj_track = traj_track
		if self.traj_track:
			with open("Circle_new.txt","rb") as fp:
				self.Circle_points = np.loadtxt(fp)

			with open("JAngles_new.txt","rb") as fp:
				self.target_angles = np.loadtxt(fp)

		vm_max_bb = -1.*9.*0.16
		vm_max_tb = -1.*9.*0.1360
		vm_max_ad = -1.*9.*0.11
		vm_max_pd = -1.*9.*0.138
	
		if self.antagonistic:
			self.MTU_unit_pd = MTU(L0=self.lm0_pd,F_max=self.Fmax_pd,Vm_max=vm_max_pd,Lt_slack=lmtu0_tb)
			self.MTU_unit_tb = MTU(L0=self.lm0_tb,F_max=self.Fmax_tricep,Vm_max=vm_max_tb,Lt_slack=lmtu0_pd)
			self.MTU_unit_bb = MTU(L0=self.lm0_bb,F_max=self.Fmax_bicep,Vm_max=vm_max_bb,Lt_slack=lmtu0_bb)
			self.MTU_unit_ad = MTU(L0=self.lm0_ad,F_max=self.Fmax_ad,Vm_max=vm_max_ad,Lt_slack=lmtu0_ad)
			self.obs_dim = 12
			self.act_dim = 4
		else:
			self.MTU_unit_bb = MTU(L0=self.lm0_bb,F_max=self.Fmax_bicep,Vm_max=-1.0,Lt_slack=lmtu0_bb)
			self.MTU_unit_ad = MTU(L0=self.lm0_ad,F_max=self.Fmax_ad,Vm_max=-1.0,Lt_slack=lmtu0_ad)
			self.obs_dim = 8
			self.act_dim = 2
		nvec = [10]*4
		control_bounds = np.array([[0,0,0,0],[1,1,1,1]])
		self.action_space = spaces.MultiDiscrete(nvec)
		#self.action_space = spaces.Box(control_bounds[0],control_bounds[1])
		high = np.inf*np.ones(self.obs_dim)
		low = -high
		self.observation_space = spaces.Box(low, high)


	def Bicep_MomentArm(self,angle):

		# convert to degrees
		if angle < 0:
			angle = 0.

		if angle > 100.:
			angle = 100.
		elbow_angle = angle*(180/np.pi)

		ma = coeff_bb_ma[0]*elbow_angle**3 + \
				coeff_bb_ma[1]*elbow_angle**2 + \
				coeff_bb_ma[2]*elbow_angle**1 + \
				coeff_bb_ma[3]*elbow_angle**0

		return ma*0.001

	def Bicep_MuscleLength(self,angle):
		if angle < 0:
			angle = 0.
		if angle > 100.:
			angle = 100.
		# convert to degrees
		elbow_angle = angle*(180/np.pi)

		ml = cst_bb + coeff_bb_ml[0]*elbow_angle**4 + \
					coeff_bb_ml[1]*elbow_angle**3 + \
					coeff_bb_ml[2]*elbow_angle**2 + \
					coeff_bb_ml[3]*elbow_angle**1 

		# need to convert to meters from mm
		return ml*0.001

	def Tricep_MomentArm(self,angle):
		if angle < 0:
			angle = 0.
		if angle > 100.:
			angle = 100.
		elbow_angle = angle*(180/np.pi)

		ma = coeff_tb_ma[0]*elbow_angle**5 + \
				coeff_tb_ma[1]*elbow_angle**4 + \
				coeff_tb_ma[2]*elbow_angle**3 + \
				coeff_tb_ma[3]*elbow_angle**2 + \
				coeff_tb_ma[4]*elbow_angle**1 + \
				coeff_tb_ma[5]*elbow_angle**0



		return ma*0.001

	def Tricep_MuscleLength(self,angle):
		if angle < 0:
			angle = 0.
		if angle > 100.:
			angle = 100.
		elbow_angle = angle*(180/np.pi)

		ml = cst_tb + coeff_tb_ml[0]*elbow_angle**6 + \
				coeff_tb_ml[1]*elbow_angle**5 + \
				coeff_tb_ml[2]*elbow_angle**4 + \
				coeff_tb_ml[3]*elbow_angle**3 + \
				coeff_tb_ml[4]*elbow_angle**2 + \
				coeff_tb_ml[5]*elbow_angle**1

		return ml*0.001


	def ADeltoid_MomentArm(self,angle):
		if angle < 0:
			angle = 0.
		if angle > 100.:
			angle = 100.
		shoudler_angle = angle*(180/np.pi)
		Poly_ad = np.poly1d(coeff_ad_ma)

		ma = Poly_ad(shoudler_angle)


		return ma*0.1

	def PDeltoid_MomentArm(self,angle):
		if angle < 0:
			angle = 0.
		if angle > 100.:
			angle = 100.
		shoudler_angle = angle*(180/np.pi)
		Poly_pd = np.poly1d(coeff_pd_ma)

		ma = Poly_pd(shoudler_angle)

		return ma*0.1

	def ADeltoid_MuscleLength(self,angle):
		if angle < 0:
			angle = 0.
		if angle > 100.:
			angle = 100.
		shoulder_angle = angle*(180/np.pi)
		ml = cst_ad + slope_ad*shoulder_angle

		return ml*0.001

	def PDeltoid_MuscleLength(self,angle):
		if angle < 0:
			angle = 0.
		if angle > 100.:
			angle = 100.
		shoulder_angle = angle*(180/np.pi)
		ml = cst_pd + slope_pd*shoulder_angle
		return ml*0.001

	def activation_bb(self,t):
		#print("time",t)
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
		theta2 = x[2]# - x[0]
		dtheta2 = x[3] #- x[1]

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

			theta2_new = x[2] + x[3]*self.dt

			# Bicep Muscle Dynamics
			lmtu_old_bb = self.Bicep_MuscleLength(theta2)
			
			F_a_bb= self.MTU_unit_bb.MuscleDynamics(a_bb,lm_bb,vm_bb)
			F_p_bb = self.MTU_unit_bb.PassiveMuscleForce(lm_bb)
			F_m_bb = F_a_bb + F_p_bb
			
			lt_bb = self.MTU_unit_bb.TendonDynamics(F_m_bb)

			new_Lmtu_bb = self.Bicep_MuscleLength(theta2_new)
			lm_new_bb = new_Lmtu_bb# - lt_bbs

			dlm_bb = (lm_new_bb - lmtu_old_bb)/self.dt #+0.245
			dvm_bb = (dlm_bb - vm_bb)/self.dt

			ema_bb = self.Bicep_MomentArm(theta2)
			Torque_bb = ema_bb*F_m_bb

			#Tricep Muscle Dynamics
			lmtu_old_tb = self.Tricep_MuscleLength(theta2)
			F_a_tb = self.MTU_unit_tb.MuscleDynamics(a_tb,lm_tb,vm_tb)
			F_p_tb = self.MTU_unit_tb.PassiveMuscleForce(lm_tb)
			#print("F_p_bb",F_p_tb)
			F_m_tb = F_a_tb + F_p_tb
			#print("F_m_tb",F_m_tb)
			ema_tb = self.Tricep_MomentArm(theta2)
			Torque_tb = ema_tb*F_m_tb
			lt_tb = self.MTU_unit_tb.TendonDynamics(F_m_tb)
			#theta2_new = x[2] + x[3]*self.dt
			new_Lmtu_tb = self.Tricep_MuscleLength(theta2_new)
			lm_new_tb = new_Lmtu_tb# - lt_tb
			dlm_tb = (lm_new_tb - lmtu_old_tb)/self.dt
			dvm_tb = (dlm_tb - vm_tb)/self.dt

			# Anterior Deltoid Muscle Dynamics
			lmtu_old_ad = self.ADeltoid_MuscleLength(theta1)
			F_a_ad = self.MTU_unit_ad.MuscleDynamics(a_ad,lm_ad,vm_ad)
			F_p_ad = self.MTU_unit_ad.PassiveMuscleForce(lm_ad)
			F_m_ad = F_a_ad + F_p_ad
			ema_ad = self.ADeltoid_MomentArm(theta1)
			#print("ema_ad",ema_ad)
			Torque_ad = ema_ad*F_m_ad
			lt_ad = self.MTU_unit_ad.TendonDynamics(F_m_ad)
			theta1_new = x[0] + x[1]*self.dt
			new_Lmtu_ad = self.ADeltoid_MuscleLength(theta1_new)
			lm_new_ad = new_Lmtu_ad# - lt_ad
			dlm_ad = (lm_new_ad - lmtu_old_ad)/self.dt #+0.105
			dvm_ad = (dlm_ad - vm_ad)/self.dt

			
			# Posterios Deltoid Muscle Dynamics
			lmtu_old_pd = self.PDeltoid_MuscleLength(theta1)
			#print("theta1",theta1)
			#print("lmtu_old_pd",lmtu_old_pd)
			F_a_pd = self.MTU_unit_pd.MuscleDynamics(a_pd,lm_pd,vm_pd)
			#print("lm_pd",lm_pd)
			#print("vm_pd",vm_pd)
			#print("fa_pd",F_a_pd)
			F_p_pd = self.MTU_unit_pd.PassiveMuscleForce(lm_pd)
			#print("fm_pd",F_p_pd)
			F_m_pd = F_a_pd + F_p_pd
			ema_pd = self.PDeltoid_MomentArm(theta1)
			#print("ema_pd",ema_pd)
			Torque_pd = ema_pd*F_m_pd
			lt_pd = self.MTU_unit_pd.TendonDynamics(F_m_pd)
			#theta2_new = x[2] + x[3]*self.dt
			new_Lmtu_pd = self.PDeltoid_MuscleLength(theta1_new)
			lm_new_pd = new_Lmtu_pd# - lt_pd
			#print("new lmtu",new_Lmtu_pd)
			dlm_pd = (lm_new_pd - lmtu_old_pd)/self.dt
			dvm_pd = (dlm_pd - vm_pd)/self.dt
			#Torque_pd = 3000.
			#print("Torque_ad",Torque_ad)
			#print("Torque_bb",Torque_bb)
			#print("Torque_pd",Torque_pd)
			#print("Torque_tb",Torque_tb)
			#Torque_pd = 3000.
			Torques = np.array([[Torque_ad + Torque_pd],[Torque_bb+Torque_tb]])

			#r = d/2
		else:
			lm_bb = x[4]
			vm_bb = x[6]

			lm_ad=x[5]
			vm_ad = x[7]

			a_bb = self.activation_bb(t)
			a_ad = self.activation_ad(t)


			#debug print

			fl_bb = np.exp(-((lm_bb/lm0_bb) - 1)**2/0.45)
			#fl_tb = np.exp(-((lm_tb/lm0_tb) - 1)**2/0.45)
			fl_ad = np.exp(-((lm_ad/lm0_ad) - 1)**2/0.45)
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

		Damping = np.array([[0.50,0],[0,0.50]])
		#Torques = np.zeros(2,) # For Time Being
		K_exo = np.array([[0,0.],[0.,self.exo_gain]])
		TargetAngles = self.TargetAngles()-q
		T_exo = np.dot(K_exo,TargetAngles)
		
		if self.exo == False:
			acc = np.dot(np.linalg.inv(Hq),(Torques+-np.dot(Cq,qdot) + Gq - np.dot(Damping,qdot)))
		elif self.exo == True:
			#print("Exo torque",T_exo)
			acc = np.dot(np.linalg.inv(Hq),(Torques+-np.dot(Cq,qdot) + T_exo + Gq - np.dot(Damping,qdot)))
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
			dx[6] = dlm_ad
			dx[7] = dlm_pd
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

		# create square waves as excitation with parameters in a
		# 

		#
		#print(self.t)

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
		#print(a)
		if self.antagonistic:
			#print(abs(a[0]))
			self.act_bb = abs(a[0])*np.ones(sim_nsteps)*0.1#1#5#1#05
			self.act_ad = abs(a[1])*np.ones(sim_nsteps)*0.1#1#5#1#05#25
			self.act_tb = abs(a[2])*np.ones(sim_nsteps)*0.1#1#5#1#05#25
			self.act_pd = abs(a[3])*np.ones(sim_nsteps)*0.1#1#5#1#05#25

		else:
			self.act_bb = a[0]*np.ones(sim_nsteps,)*0.05
			self.act_ad = a[1]*np.ones(sim_nsteps,)*0.05

		t = np.arange(0,sim_length,self.dt)
		
		state = np.concatenate((self.ArmState,self.Cur_lm,self.Cur_vm))
		begin = time.time()
		data = odeint(self.MuscleArmDynamics,state,t)
		end = time.time()

		#print("sim time",end-begin)
		if self.antagonistic:
			self.ArmState = data[-1,:4]
			self.Cur_lm = data[-1,4:8]
			self.Cur_vm = data[-1,8:]

		else:
			self.ArmState = data[-1,:4]
			self.Cur_lm = data[-1,4:6]
			self.Cur_vm = data[-1,6:]

		#print("data",data.shape)
		pos = data
		#print(pos[0,1])
		x = np.cumsum([0,0.3*np.sin(pos[-1,0]),0.51*np.sin(pos[-1,0]+pos[-1,2])])
		y = np.cumsum([0,-0.3*np.cos(pos[-1,0]),-0.51*np.cos(pos[-1,0]+pos[-1,2])])

		done = False

		# Just testing the dynamics
		angle1 = pos[-1,0]
		angle2 = pos[-1,2]

		point_on_circle = int(self.t/0.005)
		angle_space = int(self.t/0.005)

		#print("in",point_on_circle)
		#print("sd",angle_space)
		#print("t",self.t)
		joint_error = (angle1 - self.target_angles[angle_space,0])**2 + (angle2 - self.target_angles[angle_space,1])**2
		joint_reward = np.exp(-10*joint_error)

		#print("joint error",joint_error)
		#print("index",point_on_circle)
		#print("x",self.Circle_points[point_on_circle,0])
		#print("y:",self.Circle_points[point_on_circle,1])
		#print("current x",x[2])
		#print("current y",y[2])
##
		#print("theta1",self.target_angles[angle_space,0])
		#print("theta2",self.target_angles[angle_space,1])
		#print("cure_theta1",angle1)
		#print("curr_theta2",angle2)

		diff = ((x[2] - self.Circle_points[point_on_circle,0])**2 + (y[2] - self.Circle_points[point_on_circle,1])**2)
		ee_reward = np.exp(-50*diff)

		#print("Joint Error : %f, ee_reward : %f"%(joint_reward,ee_reward))
		#print("*****************************************")
		if self.traj_track:
			#print("Error :",diff)
			reward = joint_reward + ee_reward - 1e-4*(sum(a**2))
			#print(sum(a**2))
			 #1/(0.001 + 10*joint_error)  + 1.0/(0.001 +  20*((x[2] - self.Circle_points[point_on_circle,0])**2 
				#+ (y[2] - self.Circle_points[point_on_circle,1])**2))    # np.exp(-5*diff)#
			#print(reward)
		else:
			reward = 1.0/(0.001 +  (x[2] - 0.6)**2 
				+ (y[2] - 0.6)**2)

		#print("Reward :",reward)
		#print("time",self.t)
		
		angle1 = pos[-1,0]
		angle2 = pos[-1,0] + pos[-1,2]
		#print(angle2 < -np.pi/100 )
		if pos[-1,0] < -np.pi/3 or pos[-1,0] > np.pi/2 or pos[-1,2] < -np.pi/100 or angle2 > 3*np.pi/2:
			done = True
			reward = 0.

		if reward < 1.6:
			done = True
			reward = 0.
		
		self.t+=sim_length
		if self.t >= 0.5:
			done = True

		#done = False

		# GYM STYLE - s,reward,done,{extra params}
		#print("done",done)
		#print(np.concatenate((self.ArmState,self.Cur_lm,self.Cur_vm)))
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
			
			F_a_bb = self.MTU_unit_bb.MuscleDynamics(a_bb,lm_bb,vm_bb)
			F_p_bb = self.MTU_unit_bb.PassiveMuscleForce(lm_bb)
			F_m_bb = F_a_bb + F_p_bb
			ema_bb = self.Bicep_MomentArm(theta2)
			Torque_bb = ema_bb*F_m_bb

			#Tricep Muscle Dynamics
			
			F_a_tb = self.MTU_unit_tb.MuscleDynamics(a_tb,lm_tb,vm_tb)
			F_p_tb = self.MTU_unit_tb.PassiveMuscleForce(lm_tb)
			F_m_tb = F_a_tb + F_p_tb
			ema_tb = self.Tricep_MomentArm(theta2)
			Torque_tb = ema_tb*F_m_tb
			

			# Anterior Deltoid Muscle Dynamics
			
			F_a_ad = self.MTU_unit_ad.MuscleDynamics(a_ad,lm_ad,vm_ad)
			F_p_ad = self.MTU_unit_ad.PassiveMuscleForce(lm_ad)
			F_m_ad = F_a_ad + F_p_ad
			ema_ad = self.ADeltoid_MomentArm(theta1)
			Torque_ad = ema_ad*F_m_ad
			

			
			# Posterios Deltoid Muscle Dynamics
			lmtu_old_pd = self.PDeltoid_MuscleLength(theta1)
			F_a_pd = self.MTU_unit_pd.MuscleDynamics(a_pd,lm_pd,vm_pd)
			F_p_pd = self.MTU_unit_pd.PassiveMuscleForce(lm_pd)
			F_m_pd = F_a_pd + F_p_pd
			ema_pd = self.PDeltoid_MomentArm(theta1)
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
		self.t = 0.
		state = np.concatenate((self.InitState,self.lm0,self.vm0))
		return state


















