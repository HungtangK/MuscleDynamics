import numpy as np
import matplotlib.pyplot as plt
from Parameters import *
from MuscleTendonUnit import *
from scipy.integrate import odeint
import matplotlib.animation as animation




def TwoLinkDynamics(theta1,theta2,dtheta1,dtheta2,Torques):

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

	# 6 dimensional vector
	#x[0] - shoulder angle
	#x[1] - shoulder velocity
	#x[2] - elbow angle
	#x[3] - elbow velocity
	#x[4] - lm_bb
	# if old method x[5] - Vm_bb
	# if new method x[5] - lt_bb
	
	# Passing in variables
	theta1 = x[0]
	theta2 = x[2]
	dtheta1 = x[1]
	dtheta2 = x[3]
	L_m = x[4]
	V_m = x[5]

	# Constant activation
	# a = 0.2

	# Square wave activation
	if (t%0.33)/0.33 > 0.1:
		a=1
	else:
		a=0

	# Method flag, if 1 then mtu based new method if 0 then lm based old method
	Flag_Method = 0

	# Muscle Dynamics Block
	if Flag_Method == 0:
		F_a = MTU_unit.MuscleDynamics(a,L_m,V_m)
		F_p = MTU_unit.PassiveMuscleForce(L_m)
		F_m = F_a + F_p
	else:
		Lmtu=Bicep_MuscleLength(theta2)
		F_m, Lmuscle_new, Ltendon_new=MTU_unit.MTU(a,Lmtu,x[4],x[5])
	
	# Environment Feedback Block
	ema = Bicep_MomentArm(x[2])
	Torques = np.array([[0.0],[ema*F_m]])
	acc=TwoLinkDynamics(theta1,theta2,dtheta1,dtheta2,Torques)

	# Postprocessing needed for old method
	if Flag_Method == 0:
		L_t = MTU_unit.TendonDynamics(F_m)
		x_new = x[2] + x[3]*0.0005
		New_Lmtu = Bicep_MuscleLength(x_new)
		Lm_new = New_Lmtu - L_t

	# Returning Derivatives 
	dx = np.zeros(6,)
	dx[0] = x[1]
	dx[1] = acc[0]
	dx[2] = x[3]
	dx[3] = acc[1]

	if Flag_Method ==0:
		# x[4] is muscle length x[5] is muscle velocity
		dx[4] = (Lm_new - x[4])/0.0005
		dx[5] = (dx[4] - x[5])/0.0005
	else:
		# x[4] is muscle length x[5] is tendon length
		dx[4] = (Lmuscle_new - x[4])/dt
		dx[5] = (Ltendon_new - x[5])/dt

	# Make sure elboe angle > 0
	if x[2] + x[3]*dt < 0:
		dx[2] = (0 - x[2])/dt

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

	# Initial Condition
	state = np.array([np.pi/4,0,0,0,lm0_bb,lt0_bb])

	# Integration
	pos = odeint(TwoLinkArm, state, t)

	# Plotting Figures
	plt.figure()
	plt.plot(t,pos[:,2])
	plt.plot(t,pos[:,0])
	plt.legend(['theta2','theta1'])
	plt.xlabel('Time')
	plt.ylabel('Joint Angle/rad')
	
	plt.figure()
	plt.plot(t, pos[:,4])
	plt.xlabel('Time')
	plt.ylabel('Bicep Muscle Length/m')

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



	