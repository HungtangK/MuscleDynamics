# We will need to put references here -Hungtang
#Parameters


lmtu0_bb = 0.4050
lmtu0_tb = 0.2660

lm0_bb = 0.160
lm0_tb = 0.130


# Moment arm 

coeff_tb_ma = [-3.5171e-9,13.277e-7,-19.092e-5,12.886e-3,-3.0284e-1,-23.287] # 5th order polynomial - input: angles in radians,output :moment arms in mm
coeff_bb_ma = [-2.9883e-5,1.8047e-3,4.5322e-1,14.660]

coeff_ad_ma = [ -3.20000000e-07,   6.00000000e-05,   1.10000000e-03,   1.03000000e-01]
coeff_pd_ma = [ -2.66666667e-07,   1.13142857e-04,  -6.34761905e-03,  -2.25571429e-01]

# Muscle lengths 
cst_bb = 378.06
cst_tb = 260.5
cst_ad = 200
cst_pd = 180

slope_ad = (120 - 200)/120
slope_pd = (260-180)/120

coeff_tb_ml = [6.1385e-11,-2.3174e-8,33.321e-7,-22.491e-5,5.2856e-3,40.644e-2]
coeff_bb_ml = [5.21e-7,-3.1498e-5,-7.9101e-3,-25.587e-2]


lmtu0_ad = 0.215
lmtu0_pd = 0.180

lm0_ad = 0.11
lm0_pd = 0.138


Fmax_bb = 1100 
Fmax_tb = 1000

Fmax_ad = 1200
Fmax_pd = 300


