import numpy as np
import scipy


class MTU:
	def __init__(self,T_act=0.0329,T_deact=0.0905,L0=0.055,F_max=6000,Vm_max=-0.45,Lt_slack=0.237,kT=180000,dt=0.0005):
		
		self.T_act = T_act
		self.T_deact = T_deact
		self.L0 = L0
		self.F_max = F_max
		self.Vm_max = Vm_max
		self.Lt_slack = Lt_slack
		self.kT = kT
		self.dt = dt
		self.L_mtu_slack = self.L0 + self.Lt_slack

	def dActivation(self,a,u):
		
		B = self.T_act/self.T_deact
		da = u/self.T_act - (1/self.T_act*(B+(1-B)*u))*a
		a += da*self.dt

		return a

	def MuscleDynamics(self,act,Lm,Vm):

		# Hill-type Muscles things
		# Force-velocity 
		if Vm < self.Vm_max:
			fv = 0.
		
		else:
			if Vm >= self.Vm_max and Vm < 0.:
				#print("here")
				fv = ((1 - (Vm/self.Vm_max))/(1 + (Vm/0.17/self.Vm_max)))

			else:
				#print("here")
				fv = (1.8 - 0.8*((1 + (Vm/self.Vm_max))/(1 - 7.56*(Vm/0.17/self.Vm_max))))
		
		# Force-length 
		a = 3.1108
		b = 0.8698
		s = 0.3914

		if Lm < 0: 
			Lm = 0.
		fl = np.exp(-(abs((((Lm/self.L0)**b)-1)/s))**a)

		# Total Active Muscle Force
		muscleForce = self.F_max*act*fv*fl
		
		return muscleForce

	def PassiveMuscleForce(self,Lm,):
		
		A = 0.0238
		b = 5.31

		if Lm>0:
			Fp = self.F_max*A*np.exp(b*((Lm/self.L0)-1))
		else:
			Fp = 0.

		return Fp

	def TendonDynamics(self,Fm):

		dL_t = (Fm + (self.F_max*np.log((-9*np.exp(-(20*Fm)/self.F_max)) + 10))/20)/self.kT
		L_t =  self.Lt_slack +dL_t# +

		return L_t

	def MuscleLength(self,L_mtu,L_t):
		
		return L_mtu - L_t

	def MTU(self,act,Lmtu,Lmuscle_old,Ltendon_old):
		Lmuscle=Lmtu-Ltendon_old
		Vmuscle=(Lmtu-Lmuscle_old)/self.dt
		Factive=MuscleDynamics(self,act,Lmuscle,Vmuscle)
		Fpassive=PassiveMuscleForce(self,Lmuscle)
		Fmuscle=Factive+Fpassive
		Ltendon=TendonDynamics(self,Fmuscle)
		return Fmtu, Lmuscle, Ltendon

		# I think the logic is this:
		# What this block does is taking in length of MTU, activation signal, and the old length of muscle and tendon
		# It calculate vm and lm as intermediate steps, bunch of forces,
		# and output Fmtu=Fmuscle as well as current muscle and tendon length

		# Then the environment block will then take in Fmtu and spit out accleration of mtu(Amtu)
		# state variables are Lmtu and Vmtu, they are integrated by Amtu
		# Lmtu after integration is then sent back to the muscle dynamic block










