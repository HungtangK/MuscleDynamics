import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
sys.path.insert(0,"/home/visak/Documents/MuscleDynamics")
from TwoDofArm import TwoDofArmEnv
import gym 
from gym import error, spaces
#from mlp_pol import MlpPol
import baselines.common.tf_util as U
import tensorflow as tf
from baselines.ppo1.mlp_policy import MlpPolicy
from gym import wrappers
import pyformulas as pf
import time


def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    """perform animation step"""
    #global arm, dt
    #arm.step(dt)
    #print(i)
    global x
    global pos
    #print(pos[i,:])
    index = int(i/50)
    angle1 = pos[i,0]#np.pi/2+-0.97#
    angle2 = pos[i,2]#angle1 + 0.45#
    if i < 50:
    	index = int(i)
    else:
    	index = 50-1

    #lines = plt.plot(x[i,0],x[i,1],'ro',linewidth=10)
    #plt.plot(0.5,0.4,'ro')
    x1 = np.cumsum([0,0.3*np.sin(angle1),0.51*np.sin(angle1+angle2)])
    y1 = np.cumsum([0,-0.3*np.cos(angle1),-0.51*np.cos(angle1+angle2)])
    #print(x)
    #print(y)

    line.set_data(*(x1,y1))
    time = float(i)*0.0005
    time_text.set_text('time = %.2f' % time)
    #lines[0].remove()
    return line, time_text

U.make_session(num_cpu=1).__enter__()

env = TwoDofArmEnv(ActiveMuscles='antagonistic',actionParameterization=True,sim_length=0.005,traj_track=True,exo=True,exo_gain=70,delay=0.020)
pol = MlpPolicy("pi",env.observation_space,env.action_space,hid_size=128,num_hid_layers=2)

U.initialize()
U.load_state('Exo_elbow_70_delay020')#reacherNoExo_11 _exo_k50   act_pen_discrete_new_exo_k200
#Trajectory3_continuos
n = 100
r = 0.075
number = [0,1,2,3,0]
global x
x = [[np.cos(2*np.pi/n*x)*r + 0.5,np.sin(2*np.pi/n*x)*r - 0.5] for x in range(0,n)]

x = np.asarray(x)

o = env.reset()

with open("JAngles_new.txt","rb") as fp:
	angles = np.loadtxt(fp)


time = 0.
data = np.empty((1,12))
activations = []
rew = 0
i = 0

while time <= 0.5:
	print(i)
	ac,vpred = pol.act(False,o)
	
	o,r,d,look = env.step(ac)
	
	activations.append(ac*0.1)
	rew+=r#*(0.99**time)
	#print(look['data'][-1,:])
	data = np.append(data,np.array([look['data'][-1,:]]),axis=0)
	time+=0.005
	i+=1
	if d:
		break



print("data",data.shape)
print("activations",np.asarray(activations).shape)

#with open("Activations_elbow_20_delay2.txt","wb") as fp:
#	np.savetxt(fp,np.asarray(activations),fmt="%1.5f")

#with open("State_data_elbow_20_delay2.txt","wb") as fp:
#	np.savetxt(fp,data,fmt="%1.5f")
data = data[1:,:]
x_end = []
y_end = []
for i in range(data.shape[0]):
	angle1 = data[i,0]
	angle2 = data[i,2]
	x1 = np.cumsum([0,0.3*np.sin(angle1),0.51*np.sin(angle1+angle2)])
	y1 = np.cumsum([0,-0.3*np.cos(angle1),-0.51*np.cos(angle1+angle2)])
	x_end.append(x1[2])
	y_end.append(y1[2])


Torques = env.Calculate_Data(data[1:,:],np.asarray(activations))

plt.figure()
plt.plot(Torques)
plt.legend(['1','2','3','4'])
plt.title('Torques')
#plt.show()

global pos

indices = np.arange(0,data.shape[0],1)

pos = data[indices,:]

print(pos.shape)
frames = int(data.shape[0]/1)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,xlim=(-1, 1), ylim=(-1, 1))
ax.grid()
plt.plot(x_end,y_end,'g',linewidth=1)
plt.plot(x[:,0],x[:,1],'b',linewidth=1)
line, = ax.plot([], [], 'o-', lw=4, mew=5)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ani = animation.FuncAnimation(fig, animate, frames=frames,interval=0, blit=True,init_func=init)
ani.save('Target_reac0.mp4', fps=60,extra_args=['-vcodec', 'libx264'])

plt.figure()
plt.plot(activations)
plt.legend(['1','2','3','4'])



plt.figure()
plt.plot(pos[:,[0,2]],'r')
plt.plot(angles,'b')
plt.legend(['1','2'])

plt.show()