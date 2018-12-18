# Parameters
# Both the Muscle and the Bone parameters are from Song et el at 2008(DOI: 10.1007/s10439-008-9461-8)
# The effective moment arm is got from Holzbaur et el 2005(DOI: 10.1007/s10439-005-3320-7)

# Muscle Parameters
# Units following SI: Force(N), Length(m), Velocity(m/s), Mass(kg)

# Gravitational constant
g = 9.81

# Vmax got from scaling factor with respect to muscle length:
factor_Vmax = -0.45/0.055


# Biceps: Fmax as addition of Long and Short; lengths as the Long muscle
Fmax_bb = 1063.99
lm0_bb = 16.00/100
lt0_bb = 23.33/100
lmtu0_bb = lm0_bb+lt0_bb
Vmax_bb = factor_Vmax*lm0_bb
EMA_bb = 3.6/100

# Triceps: Fmax as addition of Lateral, Long and Medial; lengths as the average of Lateral, Long and Medial
Fmax_tb = 2004.65
lm0_tb = 13.60/100
lt0_tb = 12.38/100
lmtu0_tb = lm0_tb+lt0_tb
Vmax_tb = factor_Vmax*lm0_tb
EMA_tb = -2.1/100

# Anterior Deltoid
Fmax_ad = 1147.99
lm0_ad = 11.00/100
lt0_ad = 10.00/100
lmtu0_ad = lm0_ad+lt0_ad
Vmax_ad = factor_Vmax*lm0_ad
EMA_ad= 1.9/100

# Posterior Deltoid: 
Fmax_pd = 265.99
lm0_pd = 13.80/100
lt0_pd = 4.00/100
lmtu0_pd = lm0_pd+lt0_pd
Vmax_pd = factor_Vmax*lm0_pd
EMA_pd = -0.8/100


# Bone Parameters, the moment of inertia is with respect to the center of mass
# Upper arm: Humerus
m1 = 1.79
l1 = 0.30
lc1 = 0.1308
I1 = 132.080/10000

# Lower arm: average of Ulna and Radius
m2 = 0.545*2
l2 = 0.2525
lc2 = (0.1036+0.0972)/2
I2 = 28.17*2/10000

# For constructing dynamic matrices
alpha = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2)
beta = m2 * l1 * lc2
delta = I2 + m2 * lc2**2

# Moment arm/ Muscle length as a function of angle 
# Shoulder data obtained from Feldman et el 1996
# original length scales in mm, angles in degrees

cst_bb = 378.06
cst_tb = 260.5
cst_ad = 200
cst_pd = 180
coeff_tb_ma = [-3.5171e-9,13.277e-7,-19.092e-5,12.886e-3,-3.0284e-1,-23.287] 
coeff_bb_ma = [-2.9883e-5,1.8047e-3,4.5322e-1,14.660]
coeff_ad_ma = [ -3.20000000e-07,   6.00000000e-05,   1.10000000e-03,   1.03000000e-01]
coeff_pd_ma = [ -2.66666667e-07,   1.13142857e-04,  -6.34761905e-03,  -2.25571429e-01]
slope_ad = (120 - 200)/120
slope_pd = (260 - 180)/120
coeff_tb_ml = [6.1385e-11,-2.3174e-8,33.321e-7,-22.491e-5,5.2856e-3,40.644e-2]
coeff_bb_ml = [5.21e-7,-3.1498e-5,-7.9101e-3,-25.587e-2]

# Muscle length relationshp in the function form
# Length scale in m and angle in rad

import numpy as np

def Bicep_MomentArm(angle):
	elbow_angle = angle*(180/np.pi)
	ma = coeff_bb_ma[0]*elbow_angle**3 + \
			coeff_bb_ma[1]*elbow_angle**2 + \
			coeff_bb_ma[2]*elbow_angle**1 + \
			coeff_bb_ma[3]*elbow_angle**0
	return ma*0.001

def Bicep_MuscleLength(angle):
	elbow_angle = angle*(180/np.pi)
	ml = cst_bb + coeff_bb_ml[0]*elbow_angle**4 + \
				coeff_bb_ml[1]*elbow_angle**3 + \
				coeff_bb_ml[2]*elbow_angle**2 + \
				coeff_bb_ml[3]*elbow_angle**1 
	return ml*0.001

def Tricep_MomentArm(angle):
	elbow_angle = angle*(180/np.pi)
	ma = coeff_tb_ma[0]*elbow_angle**5 + \
			coeff_tb_ma[1]*elbow_angle**4 + \
			coeff_tb_ma[2]*elbow_angle**3 + \
			coeff_tb_ma[3]*elbow_angle**2 + \
			coeff_tb_ma[4]*elbow_angle**1 + \
			coeff_tb_ma[5]*elbow_angle**0
	return ma*0.001

def Tricep_MuscleLength(angle):
	elbow_angle = angle*(180/np.pi)
	ml = cst_tb + coeff_tb_ml[0]*elbow_angle**6 + \
			coeff_tb_ml[1]*elbow_angle**5 + \
			coeff_tb_ml[2]*elbow_angle**4 + \
			coeff_tb_ml[3]*elbow_angle**3 + \
			coeff_tb_ml[4]*elbow_angle**2 + \
			coeff_tb_ml[5]*elbow_angle**1
	return ml*0.001


def ADeltoid_MomentArm(angle):
	shoudler_angle = angle*(180/np.pi)
	Poly_ad = np.poly1d(coeff_ad_ma)
	ma = Poly_ad(shoudler_angle)
	return ma*0.1

def PDeltoid_MomentArm(angle):
	shoudler_angle = angle*(180/np.pi)
	Poly_pd = np.poly1d(coeff_pd_ma)
	ma = Poly_pd(shoudler_angle)
	return ma*0.1

def ADeltoid_MuscleLength(angle):
	shoulder_angle = angle*(180/np.pi)
	ml = cst_ad + slope_ad*shoulder_angle
	return ml*0.001

def PDeltoid_MuscleLength(angle):
	shoulder_angle = angle*(180/np.pi)
	ml = cst_pd + slope_pd*shoulder_angle
	return ml*0.001

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
				,[		m2*g*lc2*np.sin(theta1+theta2)								]])
	
	# damping = np.array([[2.10,0],[0,2.10]])
	damping = np.array([[0,0],[0,0]])
	
	acc = np.dot(np.linalg.inv(Hq),(Torques+-np.dot(Cq,qdot) - Gq - np.dot(damping,qdot)))

	return acc

def HandleForce(t):
	# Handle Force in N
	# In the video, the wok is on the left and the chef is on the right
	# Therefore a negative sign is put in front of Fx and Px

	Fx = 27.8013 * np.cos(	3.0395*t  + -0.6416 )+\
    	4.1793   * np.cos(	0     *t  +  0		)+\
    	0.4661   * np.cos( 	9.1185*t  + -2.8723 )

    Fy = 7.9358  * np.cos(  3.0395*t  + -0.5657	)+\
    	3.4581   * np.cos(      0 *t  +  0		)+\
    	2.9076   * np.cos( 6.0790 *t  + -0.6469 )

	return np.array([[-Fx],[Fy]])

def HandlePosition(t):
	# Handle Position in m
	# Coordinate originates at the stove rim. 
	
	Px = 149.7777 * np.cos( 0      *t  +  0		)+\
   		35.7263   * np.cos( 3.0395 *t  +  2.8876)+\
    	2.3334    * np.cos( 6.0790 *t  + -0.9191)

    Py = 145.8037 * np.cos( 0      *t  +  0		)+\
   		35.7356   * np.cos( 3.0395 *t  +  1.2410)+\
    	2.3208    * np.cos(	6.0790 *t  +  2.9916)

    Px = Px/1000
    Py = Py/1000

	return np.array([[-Px],[Py]])

def ExternalTorque(theta1,theta2,t)
	# Return External Torque as a result of the handle force
	# Handle position is to be treated as one of the learning objective therefore not included in the function

	# Receive the Handle Force
	HandleForce = HandleForce(t)

	# Vector from the shoulder to the hand
	Vs = [[l1*np.sin(theta1)+l2*np.sin(theta1+theta2)],[-l1*np.cos(theta1)-l2*np.cos(theta1+theta2)]]
	# Vector prependicular to Vs in the positive torque direction
	Vsn = [-Vs[1],Vs[0]]
	# External Torque on the shoulder
	ExternalTorqueShoulder = np.dot(HandleForce,Vsn)

	# Vector from the elbow to the hand
	Ve = [[l2*np.sin(theta1+theta2)],[-l2*np.cos(theta1+theta2)]]
	# Vector prependicular to Vs in the positive torque direction
	Ven = [-Vs[1],Vs[0]]
	# External Torque on the shoulder
	ExternalTorqueElbow = np.dot(HandleForce,Ven)

	return np.array([[ExternalTorqueShoulder],[ExternalTorqueElbow]])