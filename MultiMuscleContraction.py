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

def TwoLinkDynamics(theta1,theta2,dtheta1,dtheta2,Torques):
	# Take in kinematics data and torque
	# Return the accelerations at the joint level

	qdot = np.array([[dtheta1],[dtheta2]])
	q = np.array([[theta1],[theta2]])
	
	Hq = np.array([[	alpha + 2*beta*np.cos(theta2)	,	delta + beta*np.cos(theta2)]\
				,[		delta + beta*np.cos(theta2)		,  	delta					]])
	Cq = np.array([[	-beta*np.sin(theta2)*dtheta2	,	-beta*np.sin(theta2)*(dtheta1+dtheta2)]\
				,[		beta*np.sin(theta2)*dtheta1		,	0						]])	
	Gq = np.array([[	(m1*lc1 + m2*l1)*g*np.sin(theta1) + m2*g*lc2*np.sin(theta1+theta2)]\
				,[		m2*g*l2*np.sin(theta1+theta2)								]])
	
	# damping = np.array([[2.10,0],[0,2.10]])
	damping = np.array([[0,0],[0,0]])
	
	acc = np.dot(np.linalg.inv(Hq),(Torques+-np.dot(Cq,qdot) + Gq - np.dot(damping,qdot)))

	return acc

def TwoLinkArm(x,t):
	# Main Integration Block

	# 6 dimensional vector
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

	# if (x[2] <= 0) & (x[3] <= 0):
	# 	x[1] +=  x[3] * I2/(I1+I2)
	# 	x[3] = 0
	# 	print('t='+str(t)+'; x[2]='+str(x[2])+'; x[3]='+str(x[3]))
	# 	print('Hit Elbow Joint Limit!')

	# Activation levels
	a_ad=act(t,peak[0],duty[0],delay[0],background[0])
	a_pd=act(t,peak[1],duty[1],delay[1],background[1])
	a_bb=act(t,peak[2],duty[2],delay[2],background[2])
	a_tb=act(t,peak[3],duty[3],delay[3],background[3])

	# MTU length from the joint angles
	Lmtu_ad=ADeltoid_MuscleLength(x[2])
	Lmtu_pd=PDeltoid_MuscleLength(x[2])
	Lmtu_bb=Bicep_MuscleLength(x[2])
	Lmtu_tb=Tricep_MuscleLength(x[2])

	# Muscle Dynamics Block
	F_ad, Lmuscle_new_ad, Ltendon_new_ad=MTU_unit.MTU(a_ad,Lmtu_ad,x[4],x[5])
	F_pd, Lmuscle_new_pd, Ltendon_new_pd=MTU_unit.MTU(a_pd,Lmtu_pd,x[6],x[7])
	F_bb, Lmuscle_new_bb, Ltendon_new_bb=MTU_unit.MTU(a_bb,Lmtu_bb,x[8],x[9])
	F_tb, Lmuscle_new_tb, Ltendon_new_tb=MTU_unit.MTU(a_tb,Lmtu_tb,x[10],x[11])

	# Environment Feedback Block
	Torques = np.array([[ADeltoid_MomentArm(x[1])*F_ad + PDeltoid_MomentArm(x[1])*F_pd],\
						[Bicep_MomentArm(x[2])*F_bb + Tricep_MomentArm(x[2])*F_tb]])
	acc=TwoLinkDynamics(x[0],x[2],x[1],x[3],Torques)

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

	# Make sure elboe angle > 0, very crude estimation, complicated, better not let theta2 goes to zero ever
	if x[2] + x[3]*dt < 0:
		dx[0] = x[1] + x[3] * I2/(I1+I2)
		dx[1] = acc[0] + acc[1] * I2/(I1+I2)
		dx[2] = (0 - x[2])/dt
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
    x = np.cumsum([0,0.3*np.sin(pos[i,0]),0.5*np.sin(pos[i,0]+pos[i,2])])
    y = np.cumsum([0,-0.3*np.cos(pos[i,0]),-0.5*np.cos(pos[i,0]+pos[i,2])])
    line.set_data(*(x,y))
    time_text.set_text('time = %.2f' % t[i])
    return line, time_text


# Main Function 

if __name__ == '__main__':
	
	MTU_unit = MTU(L0=0.160,F_max=1000,)

	# Integration Time Series
	dt = 0.0005
	t = np.arange(0, 5.0, dt)

	# Activation Parameters
	f = 3.3
	T = 1/f
	peak = 			[1.0,1.0,1.0,1.0]		# peak value of the signal
	background = 	[0.0,0.0,0.5,0.0]		# background activation
	duty = 			[0.5,0.5,0.5,0.5]		# duty cycle dimensionless %cycle
	delay = 		[0.0,0.5,0.0,0.5]		# delay dimensionless %cycle
	activation=np.zeros([4,t.shape[0]])
	for i in range(t.shape[0]):
		for j in range(4):
			activation[j,i]=act(t[i],peak[j],duty[j],delay[j],background[j])

	# Initial Condition
	state = np.array([	-np.pi/8,0,		\
						np.pi/3*2,0,		\
						lm0_ad,lt0_ad,	\
						lm0_pd,lt0_pd,	\
						lm0_bb,lt0_bb,	\
						lm0_tb,lt0_tb	\
						])

	# Integration
	# pos = odeint(TwoLinkArm, state, t, mxstep=5000000)
	pos = odeint(TwoLinkArm, state, t)

	# Plotting Figures
	# Joint Angles
	plt.figure()
	plt.plot(t,pos[:,2])
	plt.plot(t,pos[:,0])
	plt.legend(['theta2','theta1'],loc='center right')
	plt.xlabel('Time')
	plt.ylabel('Joint Angle(rad)')
	
	# Joint Velocities
	plt.figure()
	plt.plot(t,pos[:,3])
	plt.plot(t,pos[:,1])
	plt.legend(['theta2','theta1'],loc='center right')
	plt.xlabel('Time')
	plt.ylabel('Joint Velocit(rad/s)')

	# Muscle Tendon Length
	plt.figure()
	plt.plot(t, pos[:,8])
	plt.plot(t, pos[:,9])
	plt.legend(['muscle','tendon'],loc='center right')
	plt.xlabel('Time')
	plt.ylabel('Length(m)')

	plt.figure()
	for i in range(4):
		plt.subplot(4, 1, i+1)
		plt.plot(t, activation[i,:])
		plt.ylim([0,1])
		plt.ylabel('a_'+str(i))
		if i==3:
			plt.xlabel('Time(s)')
		else:
			plt.xticks([])

	plt.show()

	# Animation
	fig = plt.figure(figsize=(4,4))
	ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-1, 1), ylim=(-1, 1))
	ax.grid()
	line, = ax.plot([], [], 'o-', lw=4, mew=5)
	time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
	ani = animation.FuncAnimation(fig, animate, frames=t.shape[0]-1,
                              interval=1, blit=True, 
                              init_func=init)

	# Save Animation
	# ani.save('2linkarm_withMuscleDynamics.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

	plt.show()



	