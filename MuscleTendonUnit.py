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


		self.L_mtu_slack = self.L0 - self.Lt_slack



	def dActivation(self,a,u):
		B = self.T_act/self.T_deact
		#print("B",B)
		da = u/self.T_act - (1/self.T_act*(B+(1-B)*u))*a

		a += da*self.dt

		return a



	def MuscleDynamics(self,act,Lm,Vm):

		# Hill-type Muscles things
		#print("vm inside muscle",Vm)
		if Vm < self.Vm_max:
			fv = 0.

		

		else:
			if Vm >= self.Vm_max and Vm < 0.:
				#print("here")
				fv = ((1 - (Vm/self.Vm_max))/(1 + (Vm/0.17/self.Vm_max)))

			else:
				#print("here")
				fv = (1.8 - 0.8*((1 + (Vm/self.Vm_max))/(1 - 7.56*(Vm/0.17/self.Vm_max))))

		a = 3.1108
		b = 0.8698
		s = 0.3914


		if Lm < 0: # this looks like a hack
			Lm = 0.

		fl = np.exp(-(abs((((Lm/self.L0)**b)-1)/s))**a)

		muscleForce = self.F_max*act*fv*fl
		
		#print("fv",fv)
		#print("fl",fl)

		return muscleForce,fv,fl

	def Force_vm(self,vm,a,fl):
		# Where did you get this from?I can't find it in the original model -Hungtang
		lmopt = 0.055
		vmmax = 0.45
		fmlen = 1.4
		af = 0.25
		vmmax = vmmax

		if vm <= 0.:
			fvm = af*a*fl*(4*vm + vmmax*(3*a + 1))/(-4*vm + vmmax*af*(3*a + 1))
		else:
			fvm = a*fl*(af*vmmax*(3*a*fmlen - 3*a + fmlen - 1) + 8*vm*fmlen*(af + 1)) / (af*vmmax*(3*a*fmlen - 3*a + fmlen - 1) + 8*vm*(af + 1)) 

		return fvm


	def PassiveMuscleForce(self,Lm,):
		
		#kpe = 5.0
		#epsm0 = 0.6
		#lm = Lm/0.055
		
		A = 0.0238
		b = 5.31

		if Lm>0:
			Fp = self.F_max*A*np.exp(b*((Lm/self.L0)-1))

		else:
			Fp = 0.

		'''
		if lm <= 1.0:
			fpe = 0.
		else:
			fpe = (np.exp(kpe*(lm-1)/epsm0)-1)/(np.exp(kpe)-1)
		'''
		return Fp


	def TendonDynamics(self,Fm):

		dL_t = (Fm + (self.F_max*np.log((-9*np.exp(-(20*Fm)/self.F_max)) + 10))/20)/self.kT
		L_t =  self.Lt_slack +dL_t# +

		return L_t

	def MuscleLength(self,L_mtu,L_t):
		return L_mtu - L_t






