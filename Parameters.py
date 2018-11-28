# Parameters
# Both the Muscle and the Bone parameters are from Song et el at 2008(DOI: 10.1007/s10439-008-9461-8)
# The effective moment arm is got from Holzbaur et el 2005(DOI: 10.1007/s10439-005-3320-7)

# Muscle Parameters
# Units following SI: Force(N), Length(m), Velocity(m/s), Mass(kg)

# Gravitational constant
g = -9.81

# Vmax got from scaling factor with respect to muscle length:
factor_Vmax = 0.45/5.5


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
Fmax_ad = 2004.65
lm0_ad = 13.60/100
lt0_ad = 12.38/100
lmtu0_ad = lm0_ad+lt0_ad
Vmax_ad = factor_Vmax*lm0_ad
EMA_ad=1.9/100

# Posterior Deltoid: 
Fmax_pd = 2004.65
lm0_pd = 13.60/100
lt0_pd = 12.38/100
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

# To be used:
# Moment arm Feldman et el 1996
coeff_tb_ma = [-3.5171e-9,13.277e-7,-19.092e-5,12.886e-3,-3.0284e-1,-23.287] # 5th order polynomial - input: angles in radians,output :moment arms in mm
coeff_bb_ma = [-2.9883e-5,1.8047e-3,4.5322e-1,14.660]
coeff_ad_ma = [ -3.20000000e-07,   6.00000000e-05,   1.10000000e-03,   1.03000000e-01]
coeff_pd_ma = [ -2.66666667e-07,   1.13142857e-04,  -6.34761905e-03,  -2.25571429e-01]
slope_ad = (120 - 200)/120
slope_pd = (260 - 180)/120
coeff_tb_ml = [6.1385e-11,-2.3174e-8,33.321e-7,-22.491e-5,5.2856e-3,40.644e-2]
coeff_bb_ml = [5.21e-7,-3.1498e-5,-7.9101e-3,-25.587e-2]




