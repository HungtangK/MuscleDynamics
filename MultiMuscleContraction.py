import numpy as np
import matplotlib.pyplot as plt
from Parameters import *
from MuscleTendonUnit import *
from scipy.integrate import odeint
from scipy import signal
import matplotlib.animation as animation



def act(t,peak,duty,delay,background):
	# Function form of the square wave activation signal
	return max(background,peak*(((t-delay*T)%T)/T < duty))

def TwoLinkArm(x,t):
	# Main Integration Block

	# 12 dimensional vector
	#x[0] - shoulder angle
	#x[1] - shoulder velocity
	#x[2] - elbow angle
	#x[3] - elbow velocity
	#x[4] - lm_ad
	#x[5] - lt_ad
	#x[6] - lm_pd
	#x[7] - lt_pd
	#x[8] - lm_bb
	#x[9] - lt_bb
	#x[10] - lm_tb
	#x[11] - lt_tb

	# Activation levels
	a_ad=act(t,peak[0],duty[0],delay[0],background[0])
	a_pd=act(t,peak[1],duty[1],delay[1],background[1])
	a_bb=act(t,peak[2],duty[2],delay[2],background[2])
	a_tb=act(t,peak[3],duty[3],delay[3],background[3])

	# MTU length from the joint angles
	Lmtu_ad=ADeltoid_MuscleLength(x[0])
	Lmtu_pd=PDeltoid_MuscleLength(x[0])
	Lmtu_bb=Bicep_MuscleLength(x[2])
	Lmtu_tb=Tricep_MuscleLength(x[2])

	# MTU velocity from the joint angles
	Vmtu_ad=(ADeltoid_MuscleLength(x[0]+x[1]*dt)-Lmtu_ad)/dt
	Vmtu_pd=(PDeltoid_MuscleLength(x[0]+x[1]*dt)-Lmtu_pd)/dt
	Vmtu_bb=(Bicep_MuscleLength(x[2]+x[3]*dt)-Lmtu_bb)/dt
	Vmtu_tb=(Tricep_MuscleLength(x[2]+x[3]*dt)-Lmtu_tb)/dt

	# Muscle Dynamics Block
	F_ad, Lmuscle_new_ad, Ltendon_new_ad=MTU_ad.MTU2(a_ad,Lmtu_ad,Vmtu_ad,x[5])
	F_pd, Lmuscle_new_pd, Ltendon_new_pd=MTU_pd.MTU2(a_pd,Lmtu_pd,Vmtu_pd,x[7])
	F_bb, Lmuscle_new_bb, Ltendon_new_bb=MTU_bb.MTU2(a_bb,Lmtu_bb,Vmtu_bb,x[9])
	F_tb, Lmuscle_new_tb, Ltendon_new_tb=MTU_tb.MTU2(a_tb,Lmtu_tb,Vmtu_tb,x[11])

	# Environment Feedback Block
	Torques = np.array([[ADeltoid_MomentArm(x[0])*F_ad + PDeltoid_MomentArm(x[0])*F_pd],\
						[Bicep_MomentArm(x[2])*F_bb + Tricep_MomentArm(x[2])*F_tb]])
	acc=TwoLinkDynamics(x[0],x[2],x[1],x[3],Torques)

	# Debug Print
	# print(str(Torques[0])+';  '+str(Torques[1]))
	# print(str(a_ad)+';  '+str(a_pd)+';  '+str(a_bb)+';  '+str(a_tb))
	# print(str(F_ad)+';  '+str(F_pd)+';  '+str(F_bb)+';  '+str(F_tb))
	# print(str((ADeltoid_MuscleLength(x[0])-lmtu0_ad)/x[0])+'; '+str((Bicep_MuscleLength(x[1])-lmtu0_bb)/x[1]))
	# print(str(ADeltoid_MomentArm(x[0]))+';  '+str(Bicep_MomentArm(x[1])))
	# print('EMA='+str((ADeltoid_MuscleLength(x[0])-lmtu0_ad)/x[0])+'; Or MA: '+str(ADeltoid_MomentArm(x[0])))

	# Returning Derivatives 
	dx = np.zeros(12,)
	dx[0] = x[1]
	dx[1] = acc[0]
	dx[2] = x[3]
	dx[3] = acc[1]
	dx[4] = (Lmuscle_new_ad - x[4])/dt
	dx[5] = (Ltendon_new_ad - x[5])/dt
	dx[6] = (Lmuscle_new_pd - x[6])/dt
	dx[7] = (Ltendon_new_pd - x[7])/dt
	dx[8] = (Lmuscle_new_bb - x[8])/dt
	dx[9] = (Ltendon_new_bb - x[9])/dt
	dx[10] = (Lmuscle_new_tb - x[10])/dt
	dx[11] = (Ltendon_new_tb - x[11])/dt

	# # Make sure elboe angle > 0, very crude estimation, complicated, better not let theta2 goes to zero ever
	if x[2] + x[3]*dt < 0:
		# dx[0] = x[1] + x[3] * I2/(I1+I2)
		# dx[1] = acc[0] + acc[1] * I2/(I1+I2)
		# dx[2] = (0 - x[2])/dt
		print('Hit Elbow Joint Limit!')

	return dx


def init():
	# For animation purpose
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
	# For animation purpose
    """perform animation step"""
    # global pos
    x = np.cumsum([0,l1*np.sin(pos[i,0]),l2*np.sin(pos[i,0]+pos[i,2])])
    y = np.cumsum([0,-l1*np.cos(pos[i,0]),-l2*np.cos(pos[i,0]+pos[i,2])])
    line.set_data(*(x,y))
    time_text.set_text('time = %.2f' % t[i])
    return line, time_text


# Main Function 

if __name__ == '__main__':
	
	# Initialization of MTU class
	MTU_ad = MTU(L0=lm0_ad,F_max=Fmax_ad,Vm_max=Vmax_ad,Lt_slack=lt0_ad)
	MTU_pd = MTU(L0=lm0_pd,F_max=Fmax_pd,Vm_max=Vmax_pd,Lt_slack=lt0_pd)
	MTU_bb = MTU(L0=lm0_bb,F_max=Fmax_bb,Vm_max=Vmax_bb,Lt_slack=lt0_bb)
	MTU_tb = MTU(L0=lm0_tb,F_max=Fmax_tb,Vm_max=Vmax_tb,Lt_slack=lt0_tb)

	# Integration Time Series
	dt = 0.0005
	t = np.arange(0, 5.0, dt)

	# Activation Parameters
	f = 3
	T = 1/f
	peak = 			[1.0,1.0,1.0,0.5]		# peak value of the signal
	background = 	[0.0,0.0,0.5,0.0]		# background activation
	duty = 			[0.5,0.5,0.5,0.5]		# duty cycle dimensionless %cycle
	delay = 		[0.5,0.0,0.6,0.1]		# delay dimensionless %cycle
	
	activation=np.zeros([4,t.shape[0]])
	for i in range(t.shape[0]):
		for j in range(4):
			activation[j,i]=act(t[i],peak[j],duty[j],delay[j],background[j])

	# Initial Condition
	state = np.array([	0,0,		\
						np.pi/2,0,		\
						lm0_ad,lt0_ad,	\
						lm0_pd,lt0_pd,	\
						lm0_bb,lt0_bb,	\
						lm0_tb,lt0_tb	\
						])

	# Integration
	# pos = odeint(TwoLinkArm, state, t, mxstep=5000000)	# Relaxing iteration requirement
	pos = odeint(TwoLinkArm, state, t)

	# Natural Frequency Calculation
	end = t.shape[0] - 10
	A1=(m1*lc1 + m2*l1)*g
	A2=m2*g*lc2
	EQ1=-A2*np.sin(pos[end,1])/(A1+A2*np.cos(pos[end,1]))
	NF1=((A1*np.cos(pos[end,0]) + A2*np.cos(pos[end,0]+pos[end,1]))\
					/(alpha + 2*beta*np.cos(pos[end,1])))**0.5/2/np.pi
	NF2=((A2*np.cos(pos[end,0]+pos[end,1]))\
					/(delta))**0.5/2/np.pi
	print('Equlibrium Pose Shoulder(rad): '+str(EQ1))
	print('Natural Frequency Shoulder(Hz): '+str(NF1))	
	print('Natural Frequency Elbow(Hz): '+str(NF2))

	# Saving Control
	Flag_save = 1
	FilePath = 'Free Movement/Case8_'
	
	# Saving Log File
	if Flag_save:
		file = open(FilePath+'Log.txt','w')
		file.write('Activation Settings:\n') 
		file.write('\tPeak:\t\t') 
		for i in range(4):
			file.write('{:.1f}'.format(peak[i])+'\t')
		file.write('\n')
		file.write('\tBackground:\t') 
		for i in range(4):
			file.write('{:.1f}'.format(background[i])+'\t')
		file.write('\n')
		file.write('\tDuty:\t\t') 
		for i in range(4):
			file.write('{:.1f}'.format(duty[i])+'\t')
		file.write('\n')
		file.write('\tDelay:\t\t') 
		for i in range(4):
			file.write('{:.1f}'.format(delay[i])+'\t')
		file.write('\n')
		file.write('\nEqulibrium Pose Shoulder(rad): '+str(EQ1))
		file.write('\nNatural Frequency Shoulder(Hz): '+str(NF1))	
		file.write('\nNatural Frequency Elbow(Hz): '+str(NF2))

		file.close

	# Figures
	# Joint Angles
	plt.figure()
	plt.plot(t,pos[:,2])
	plt.plot(t,pos[:,0])
	plt.legend(['theta2','theta1'],loc='center right')
	plt.xlabel('Time(s)')
	plt.ylabel('Joint Angle(rad)')
	plt.grid()
	if Flag_save:
		plt.savefig(FilePath+'JointAngle.png')
		plt.savefig(FilePath+'JointAngle.eps')

	# Joint Velocities
	plt.figure()
	plt.plot(t,pos[:,3])
	plt.plot(t,pos[:,1])
	plt.legend(['theta2','theta1'],loc='center right')
	plt.xlabel('Time(s)')
	plt.ylabel('Joint Velocit(rad/s)')
	plt.grid()
	if Flag_save:
		plt.savefig(FilePath+'JointVelocity.png')
		plt.savefig(FilePath+'JointVelocity.eps')

	# Muscle Tendon Length
	plt.figure()

	plt.subplot(4,1,1)
	plt.plot(t, pos[:,4] - lm0_ad)
	plt.plot(t, pos[:,5] - lt0_ad)
	plt.legend(['muscle','tendon'],loc='center right')
	plt.ylabel('L_AD(m)')
	plt.xticks([])

	plt.subplot(4,1,2)
	plt.plot(t, pos[:,6] - lm0_pd)
	plt.plot(t, pos[:,7] - lt0_pd)
	plt.legend(['muscle','tendon'],loc='center right')
	plt.ylabel('L_PD(m)')
	plt.xticks([])

	plt.subplot(4,1,3)
	plt.plot(t, pos[:,8] - lm0_bb)
	plt.plot(t, pos[:,9] - lt0_bb)
	plt.legend(['muscle','tendon'],loc='center right')
	plt.ylabel('L_BB(m)')
	plt.xticks([])

	plt.subplot(4,1,4)
	plt.plot(t, pos[:,10] - lm0_tb)
	plt.plot(t, pos[:,11] - lt0_tb)
	plt.legend(['muscle','tendon'],loc='center right')
	plt.ylabel('L_TB(m)')
	plt.xlabel('Time(s)')

	
	if Flag_save:
		plt.savefig(FilePath+'Lengths.png')
		plt.savefig(FilePath+'Lengths.eps')

	# Plot Activation Level
	YLabels=['AD','PD','BB','TB']
	plt.figure()
	for i in range(4):
		plt.subplot(4, 1, i+1)
		plt.plot(t, activation[i,:])
		plt.ylim([0,1.1])
		plt.ylabel('a_'+YLabels[i])
		if i==3:
			plt.xlabel('Time(s)')
		else:
			plt.xticks([])

	if Flag_save:
		plt.savefig(FilePath+'Activation.png')
		plt.savefig(FilePath+'Activation.eps')

	# End Effector Trajectory
	plt.figure()
	plt.plot(l1*np.sin(pos[:,0])+l2*np.sin(pos[:,0]+pos[:,2]),\
			-l1*np.cos(pos[:,0])-l2*np.cos(pos[:,0]+pos[:,2]))
	plt.plot(l1*np.sin(pos[:,0]),\
			-l1*np.cos(pos[:,0]))
	plt.plot(0,0,'ko')
	plt.xlabel('x(m)')
	plt.ylabel('y(m)')
	plt.legend(['hand','elbow','shoulder'],loc='upper right')
	if Flag_save:
		plt.savefig(FilePath+'Trajectory.png')
		plt.savefig(FilePath+'Trajectory.eps')
	# Animation
	# Cant get the animation interval to be right, give up for now.
	# Ouput file fps is correct

	dframe = 10
	speedx = 1
	fig = plt.figure(figsize=(4,4))
	ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-1, 1), ylim=(-1, 1))
	ax.grid()
	line, = ax.plot([], [], 'o-', lw=4, mew=5)
	time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
	interval = 1000 * dt *dframe / speedx
	ani = animation.FuncAnimation(fig, animate, frames=range(0, t.shape[0], dframe),
                              interval=interval, blit=True, 
                              init_func=init)
	
	# Save Animation
	if Flag_save:
		ani.save(FilePath+'Animation.mp4', fps=int(1/dt/dframe*speedx), extra_args=['-vcodec', 'libx264'])

	plt.show()



	