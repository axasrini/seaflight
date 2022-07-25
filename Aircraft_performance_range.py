from cmath import sqrt
from configparser import Interpolation
from email.headerregistry import ContentDispositionHeader
from turtle import shape
from unittest.util import _MIN_COMMON_LEN
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate
from scipy.interpolate import griddata
import seaborn as sns
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

# Ctrl K + Ctrl C adds the # in VS for selected lines
# Ctrl K + Ctrl U removes it 


"""
Calculate range using Breguet eqn for electric and hybrid aircraft 
Inputs:
Battery kWh vs range for 21700 NCM cell 
Varying Cl/Cd for different ground effects
WTO based on half a ton aircraft with varying Wpay
Average efficiencies for propulsion
"""
CL_CD_ratio = np.array(
    [8, 15, 20, 25, 30, 35, 40], dtype=float
)

MTOW = 500;  #kg max taekoff weight init  
Wbatt_ratio = np.array(        #nasa electric x57 is 0.28 https://www.nasa.gov/centers/armstrong/news/FactSheets/FS-109.html
	[0.2, 0.25, 0.3, 0.35], dtype=float
)

few = 0.5    # fraction of empty weight baseline from https://link.springer.com/article/10.1007/s13272-021-00530-w

## use this if you want to sweep through an empty weight
# np.array(
# 	[0.35,0.4,0.5], dtype=float
# )

empty_weight= few*MTOW 
cb=540000  #J/kg for good density LFP battery pack 150Wh*3600/kg
range_electric = np.empty(shape=(len(CL_CD_ratio),len(Wbatt_ratio))) #initializing output variable

# Other constants
rho_sl = 1.1671
g= 9.81
n_i = 0.93
n_m = 0.9
n_p = 0.9

class nf(float):
    def __repr__(self):
        str = '%.1f' % (self.__float__(),)
        if str[-1] == '0':
            return '%.0f' % self.__float__()
        else:
            return '%.1f' % self.__float__()

fmt = '%r'

def calc_range_breguet():
    """
    Function calculates Range from breguet equation
    :param Wbatt_ratio: ratio of battery pack weight to take off weight of plane
    :return:numpy array with electric range
    """
    for i in range(len(CL_CD_ratio)):
        CL_CD = CL_CD_ratio[i]
        for j in range(len(Wbatt_ratio)):
            Wbatt_Wto = Wbatt_ratio[j]
            range_electric[i, j] = cb / g * CL_CD * Wbatt_Wto * n_i * n_m * n_p / 1000 #in kms 

    return range_electric


# range_e = np.array(
#     [50,70,90,110,130,150,170,190,220,250]
# )
# Wbattrat = np.empty(
#     shape= (len(range_e),len(CL_CD_ratio))
# )

# def calc_batt_range_breguet():
#     for i in range(np.size(range_e)):
#         for j in range(np.size(CL_CD_ratio)):
#             CL_CD = CL_CD_ratio[j]
#             Wbattrat[i,j] = range_e[i]/ (cb / g * CL_CD[j] * n_i * n_m * n_p / 1000)  

#     return Wbattrat



if __name__ == '__main__':
    calc_range_breguet()
    print(range_electric)
    
    #assuming a half ton aircraft calculate the payload and battery estimates based on Wbatt_Wto ratio
    Wbatt = Wbatt_ratio * MTOW
    Wpay = MTOW - few*MTOW - Wbatt
    batt_cap = cb * Wbatt / 3600 /1000   #kWh
    print("Payloads are ", Wpay)
    print("Battery weights are ", Wbatt)
    print("Empty weight is ", few*MTOW)
    print("batt caps are in kWh:",batt_cap)
    # calc_batt_range_breguet()
    # print(Wbattrat)   
    # Wbatt_2 = Wbattrat * MTOW
    # Wpay_2 = MTOW - few*MTOW - Wbatt_2
    # batt_cap_2 = cb * Wbatt_2 / 3600 /1000   #kWh
    # print("Payloads are ", Wpay_2)
    # print("Battery weights are ", Wbatt_2)


#plot some stuff

#z3 = griddata( CL_CD_ratio, Wbatt, (xi, yi), method='linear')
plt.figure()
levels_range = np.linspace(25,500,30)
c3 = plt.contour(Wbatt, CL_CD_ratio, range_electric, levels=levels_range, linewidths=1.2,colors='k',label='Cl/Cd')
c3.levels = [nf(val) for val in c3.levels]
plt.clabel(c3,c3.levels,inline=True,fmt=fmt,fontsize=10)
plt.xlabel(r'Battery weight (kg)')
plt.ylabel(r'Cl_Cd ratio')
plt.annotate('Range in km ',(1000,100),color='k')
plt.title('Battery Range curve')

plt.figure()
levels_range = np.linspace(25,500,30)
c4 = plt.contour(batt_cap, CL_CD_ratio, range_electric, levels=levels_range, linewidths=1.2,colors='k',label='Cl/Cd')
c4.levels = [nf(val) for val in c4.levels]
plt.clabel(c4,c4.levels,inline=True,fmt=fmt,fontsize=10)
plt.xlabel(r'Battery capacity (kWh)')
plt.ylabel(r'Cl_Cd ratio')
plt.annotate('Range in km ',(1000,100),color='k')
plt.title('Battery Range curve')

plt.figure()
levels_range = np.linspace(25,500,30)
c4 = plt.contour(Wpay, CL_CD_ratio, range_electric, levels=levels_range, linewidths=1.2,colors='k',label='Cl/Cd')
c4.levels = [nf(val) for val in c4.levels]
plt.clabel(c4,c4.levels,inline=True,fmt=fmt,fontsize=10)
plt.xlabel(r'Payload (kg)')
plt.ylabel(r'Cl_Cd ratio')
plt.annotate('Range in km ',(1000,100),color='k')
plt.title('Battery Range curve')
# plt.figure()
# levels_range = np.linspace(5,400,10)
# c4 = plt.contour(CL_CD_ratio, range_e, Wbatt_2, levels=levels_range, linewidths=1.2,colors='k',label='Cl/Cd')
# c4.levels = [nf(val) for val in c4.levels]
# plt.clabel(c4,c4.levels,inline=True,fmt=fmt,fontsize=10)
# plt.xlabel(r'Cl_Cd ratio')
# plt.ylabel(r'Range kms')
# #plt.annotate('Range in km ',(1000,100),color='k')
# plt.title('Battery weights req (kg)')

# plt.figure()
# levels_range = np.linspace(5,400,10)
# c4 = plt.contour(CL_CD_ratio, range_e, batt_cap_2, levels=levels_range, linewidths=1.2,colors='k',label='Cl/Cd')
# c4.levels = [nf(val) for val in c4.levels]
# plt.clabel(c4,c4.levels,inline=True,fmt=fmt,fontsize=10)
# plt.xlabel(r'Cl_Cd ratio')
# plt.ylabel(r'Range kms')
# #plt.annotate('Range in km ',(1000,100),color='k')
# plt.title('Battery capacities req (kWh)')

# plt.figure()
# levels_range = np.linspace(5,400,10)
# c4 = plt.contour(CL_CD_ratio, range_e, Wpay_2, levels=levels_range, linewidths=1.2,colors='k',label='Cl/Cd')
# c4.levels = [nf(val) for val in c4.levels]
# plt.clabel(c4,c4.levels,inline=True,fmt=fmt,fontsize=10)
# plt.xlabel(r'Cl_Cd ratio')
# plt.ylabel(r'Range kms')
# #plt.annotate('Range in km ',(1000,100),color='k')
# plt.title('Payload possible (kg)')


## consider hybrid levels and calculate total range 
c_p = 0.408 / 1000 / 60 / 60 # PT6A SFC (kgfuel/W*s)
ng = 0.9
LoD = 20 #choose Cl/Cd from previous def 

def R_f(w_pay,eps):
	log_interior = (eps+(1-eps)*cb*c_p)*MTOW/(eps*MTOW+(1-eps)*cb*c_p*(w_pay+few*MTOW))
	r_f = LoD*n_i*n_m*n_p*ng/c_p/g* np.log(log_interior)
	return r_f

def R_e(w_pay,eps):
	r_e = eps*cb*LoD*n_i*n_m*n_p/g*((1-few)*MTOW-w_pay)/(eps*MTOW+(1-eps)*cb*c_p*(w_pay+few*MTOW))
	return r_e

def R_tot(w_pay,eps):
	r_tot = R_f(w_pay,eps)+R_e(w_pay,eps)
	return r_tot
	
def W_f(w_pay,eps):
	w_f = (1-eps)*cb*c_p*((1-few)*MTOW-w_pay)/(eps+(1-eps)*cb*c_p)
	return w_f

fmt = '%r'

epses = np.linspace(0,1.1,111)
payloads = np.linspace(50,300,10)

R_tots = []
eps_flat = []
pay_flat = []
w_fs = []

for i, eps in enumerate(epses):
	for j, payload in enumerate(payloads):
			R_tots.append(R_tot(payload,eps)/1000)
			eps_flat.append(eps)
			pay_flat.append(payload)
			w_fs.append(W_f(payload,eps))
			
			
xi = np.linspace(50,2000,100) #range
yi = np.linspace(50,300,100) #payload
# #print('R total is:',R_tots)
# #points = [R_tots,pay_flat]
# z3 = griddata((R_tots,pay_flat), eps_flat, (xi,yi), method='linear',fill_value=0)
# levels_eps = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
# #eps_z = np.reshape(eps_flat, (payloads.size,R_tots.size))
# plt.figure()
# #xvals = xi.unique()
# #yvals = yi.unique()
# #zvals = z3.values.reshape(len(xvals), len(yvals)).T
# #c5= plt.contourf(R_tots,pay_flat,levels=levels_eps,linewidths=1.2,colors='k',label='eps')
# c5 = plt.tricontourf(xi,yi,z3,levels=levels_eps,linewidths=1.2,colors='k',label='eps')
# c5.levels = [nf(val) for val in c5.levels]
# plt.clabel(c5,c5.levels,inline=True,fmt=fmt,fontsize=10)

#z4 = griddata(R_tots,pay_flat,w_fs,xi,yi,method="linear")
levels_wf = [0,50,100,150]
fuel_wt = np.reshape(w_fs, (payloads.size,epses.size))
plt.figure()
c4 = plt.contour(epses,payloads,fuel_wt,levels=levels_wf,linewidths=1.2,colors='r',label='fuel')
c4.levels = [nf(val) for val in c4.levels]
plt.clabel(c4,c4.levels,inline=True,fmt=fmt,fontsize=10)

plt.xlim(0,1)
plt.ylim(0,500)
plt.xlabel(r'Level of hybridization')
plt.ylabel(r'Payload (kg)')
plt.annotate('$\epsilon$ - degree of hybridization',(2000,300),color='k')
plt.annotate('Fuel volume limit',(2000,250),color='r')

print(max(w_fs))
print(min(w_fs))

plt.title('Payload-Range Curve for Hybrid \n Seaflight')


"""
Section2: Use for aircraft design when lift and coefficients of lift & drag are unknown
"""
## Calculate range based on performance 
#Use this for aitcraft design from scratch

rho_sl = 1.225 #kg/m3
Cdo = 0.028
AR = 4
e=0.9
pi=3.14
K = 1 / (pi * AR * e)
S = 5*1.875   #m2 assuming a 1/5th scale for die Uberlander wing area
#alt = 400
#rho_alt = 
V = np.arange(20,60,10)  #mps at sea level
print("Velocity is ",V*3.6)
Cl = MTOW * 9.81 / (0.5 * rho_sl * np.square(V) *S) 
drag = np.empty(shape=(len(V)))
P = np.empty(shape=(len(V)))
energy_flight = np.empty(shape=(len(V)))
flight_time = 160934 / V  #hawaiian trip in meters/mps for flight time in seconds

def calc_drag_force():
    #this defines Cl required
    for i in range(len(V)):
        drag[i] = (0.5 * rho_sl * np.square(V[i]) * S)* (Cdo + K * np.square((MTOW*9.81)/ (0.5*rho_sl* np.square(V[i]) * S))) #N
        P[i] = drag[i] * V[i] /1000 #kW at sea level
        energy_flight[i] = P[i] * 1000 * flight_time[i] / 3600 #Wh
        Cl[i] = ((MTOW*9.81)/ (0.5*rho_sl* np.square(V[i]) * S))
    return drag,P,energy_flight

if __name__ == '__main__':
    calc_drag_force()
    print("Power kW is ",P)
    print("Drag force is ", drag)
    print("Energy consumed for flight is", energy_flight)
    print("Cl across velocities", Cl)

# V_ALT = V * sqrt(rho_alt/ rho_sl)
# drag_ALT = Cdo * 0.5 * rho_sl * V_ALT^2 * S + (2* K * (MTOW*2.204)^2)/ rho_sl* V_ALT^2 * S
# P_ALT = drag_ALT * V_ALT

# calculate design velocity
design_vel = V[np.argmin(P)]*3.6 #kph
design_Cl = Cl[np.argmin(P)]
design_Cl_Cd_ratio = design_Cl/ Cdo
print("Velocity for minimum power consumption is ", design_vel)  
design_energy = energy_flight[np.argmin(P)]/1000  #kWh
print("Energy at design velocity is ", design_energy)
design_flight_time = flight_time[np.argmin(P)]
min_P = np.min(P)
print("Minimum power drawn in kW is: ", min_P)

#calculate V to achieve L/D max correct way to find min power for prop aircraft
Cle = np.sqrt(Cdo/K)
V_E = np.sqrt((2*MTOW*9.81)/(rho_sl*S*Cle))*3.6  #kph
f = interpolate.interp1d(V*3.6,energy_flight)
# print(f)
f2 = interpolate.interp1d(V*3.6,P) #kW and kph
design_energy_E =f(V_E)/1000  #kWh
design_power = f2(V_E) #kW
print("Speed for max range and L/Dmax in kph is: ", V_E, \
    "and corr energy is: ",design_energy_E, \
    "corr power in kW:", design_power)


plt.figure(700)
plt.plot(V* 3.6, drag, label="Drag newtons")
plt.legend(loc="upper right")
plt.xlabel(r'Velocity kph')
plt.ylabel(r'Drag/Energy consumed')
plt.title('drag for straight and level flight 100 mi')

plt.figure(800)
plt.plot(V* 3.6, P, label="Power kW")
plt.legend(loc="upper right")
plt.xlabel(r'Velocity kph')
plt.ylabel(r'Power kW')
plt.title('Power for straight and level flight 100 mi')

plt.figure(900)
plt.plot(V* 3.6, energy_flight, label="Energy Wh")
plt.xlabel(r'Velocity kph')
plt.ylabel(r'energy Wh')
plt.title('energy consumed for straight and level flight 100 mi')
plt.plot(design_vel, design_energy*1000, marker="o", markersize=10, label="Vel for min power") 
plt.plot(V_E, design_energy_E*1000, marker="x", markersize=10, label="Vel for min L.Dmax") 
plt.text(0.65, 500000, "Vel in kph using min power method: %s" % "{:.2f}".format(design_vel))
plt.text(0.65, 450000, "Energy in kWh: %s" % "{:.2f}".format(design_energy))
plt.text(0.65, 400000, "Flight time in s: %s" % "{:.2f}".format(design_flight_time))
plt.text(0.65, 350000, "Vel in kph using L/Dmax method: %s" % "{:.2f}".format(V_E))
plt.text(0.65, 300000, "Corr Cl/Cd using min power method: %s" % "{:.2f}".format(design_Cl_Cd_ratio))
plt.legend(loc="upper right")

plt.figure(999)
plt.plot(V* 3.6, Cl/Cdo, label ="CL/CD ratio")
plt.legend(loc="upper right")
plt.xlabel(r'Velocity kph')
plt.ylabel(r'Cl')

"""
Find new VMD = velocity at min drag with constant headwind/tailwind
"""
HW = 20/3.6 #in units of mps set this to a negative value if it is a tailwind 
coeff = [rho_sl*S*Cdo, -HW*3/2* rho_sl*S*Cdo, 0,0, \
     (-2*K*np.square(9.81*MTOW)/(0.5*rho_sl*S)), (K*np.square(9.81*MTOW)*HW)/(0.5*rho_sl*S)]
# coeff = [rho_sl*S*Cdo, -HW*3/2* rho_sl*S*Cdo, 0, -HW*9.81*np.square(MTOW)*K]
sols = np.roots(coeff)
print("roots of new velocity eqn with headwind is:", sols)
V_HW = sols[0].real*3.6
P_HW = f2(V_HW) #in kW
E_HW = f(V_HW) #in Wh
print("Power required with HW is:", P_HW, "Energy req is:",E_HW,"at new velocity",sols[0]*3.6)


"""
Section3: Use when Cl and Cd values are already provided from an initial aircraft design
Output: Calculate range and power requirements using Cl and Cd from Will
"""

V=[]
V = np.arange(20,60,10)  #mps at sea level
#Cdo = 0.028
CL_CD_ratio = np.array(
    [49.402, 35.995, 29.5, 25.557, 15, 8], dtype=float
)
Cd = np.array(
    [0.017, 0.02, 0.023, 0.025, 0.028], dtype = float
)

flight_time = 160934 / V  #hawaiian trip in meters/mps for flight time in seconds

#initializing arrays
drag = np.empty(shape=(len(V),len(Cd)))
Power = np.empty(shape=(len(V),len(Cd)))
Cl = np.empty(shape=len(Cd))
energy = np.empty(shape=(len(V),len(Cd)))

#function to calculate drag force given Cl and Cl/Cd ratio
def calc_drag_force_using_Cl():
    for i in range(len(V)):
        for j in range(len(Cd)):
            drag[i, j] = (0.5 * rho_sl * np.square(V[i]) * S) * \
                (Cd[j])  # Newtons
            Power[i,j] = drag[i,j] * V[i] / 1000  # kW at sea level
            energy[i,j] = Power[i,j] * 1000 * flight_time[i] / 3600  # Wh
    return drag, Power, energy, Cl

#run the function and output contour plots for drag, power and energy for different Cl/Cd ratios
if __name__ == '__main__':
    calc_drag_force_using_Cl()
    print("Using ClCd ratios drag is:",drag, "power is:", Power, "energy is", energy)

plt.figure()
levels_drag = np.linspace(10,500,10)
c3 = plt.contour(Cd, V*3.6, drag, levels=levels_drag, linewidths=1.2,colors='k',label='Drag')
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
c3.levels = [nf(val) for val in c3.levels]
plt.clabel(c3,c3.levels,inline=True,fmt=fmt,fontsize=10)
plt.xlabel(r'Drag coeff Cd')
plt.ylabel(r'Velocity kph')
#plt.annotate('Drag',(1000,100),color='k')
plt.title('Drag variation with Cl/Cd and velocity')

plt.figure()
levels_pow = np.linspace(0,30,10)
c3 = plt.contour(Cd,V*3.6, Power, levels=levels_pow, linewidths=1.2,colors='k',label='Power')
c3.levels = [nf(val) for val in c3.levels]
plt.clabel(c3,c3.levels,inline=True,fmt=fmt,fontsize=10)
plt.xlabel(r'Drag coeff Cd')
plt.ylabel(r'Velocity kph')
#plt.annotate('Power',(1000,100),color='k')
plt.title('Power variation with Cl/Cd and velocity')

plt.figure()
levels_ene = np.linspace(500,30000,10)
c3 = plt.contour(Cd,V*3.6, energy, levels=levels_ene, linewidths=1.2,colors='k',label='Energy')
c3.levels = [nf(val) for val in c3.levels]
plt.clabel(c3,c3.levels,inline=True,fmt=fmt,fontsize=10)
plt.xlabel(r'Drag coeff Cd')
plt.ylabel(r'Velocity kph')
#plt.annotate('Power',(1000,100),color='k')
plt.title('Energy variation with Cl/Cd and velocity')
#plt.plot(CL_CD_ratio,energy_flight[3,:],label = "energy_flight vs Cl/Cd at %s" % "{:.2f}".format(V[3]))

plt.show()


"""
Section to check cals against other planes
"""

"""
MTOW_arr = np.array(
    [500, 600, 700], dtype=float
)

velocity_range = [80 100 120 150 200]
for m in range(MTOW_arr):
    calc_drag_force_using_Cl
    plot.figure(100)
    plt.plot(V*3.6, )

# check against eviation alice plane

#9 minutes
#150 ktas 
"""

""""
V = np.arange(20,150,20)
Cdo = 0.029
e = 0.83
S = 28.9
MTOW = 7491
W = MTOW*9.81 #newtons
Rho_10000ft = 0.904 
rho_sl = Rho_10000ft
AR = 12.7
calc_drag_force()

design_vel = V[np.argmin(P)]*3.6 #kph
print("Velocity for minimum power consumption in kph is ", design_vel)  
design_energy = energy_flight[np.argmin(P)]/1000  #kWh
print("Energy at design velocity is ", design_energy)
design_flight_time = flight_time[np.argmin(P)]

plt.figure()
plt.plot(V* 3.6, drag, label="Drag newtons")
plt.legend(loc="upper right")
plt.xlabel(r'Velocity kph')
plt.ylabel(r'Drag/Energy consumed')
plt.title('drag for straight and level flight in Hawaii - 100 mi')

plt.figure()
plt.plot(V* 3.6, P, label="Power kW")
plt.legend(loc="upper right")
plt.xlabel(r'Velocity kph')
plt.ylabel(r'Power kW')
plt.title('Power for straight and level flight in Hawaii - 100 mi')

plt.figure()
plt.plot(V* 3.6, energy_flight, label="Energy Wh")
plt.legend(loc="upper right")
plt.xlabel(r'Velocity kph')
plt.ylabel(r'energy Wh')
plt.title('energy consumed for straight and level flight in Hawaii - 100 mi')
plt.plot(design_vel, design_energy*1000, marker="o", markersize=10) 
plt.text(0.65, 50000, "Vel in kph: %s" % design_vel)
plt.text(0.65, 47500, "Energy in kWh: %s" % design_energy)
plt.text(0.65, 45000, "Flight time in s: %s" % design_flight_time)

plt.show()

K = 1 / (pi * AR * e)

Cle = np.sqrt(Cdo/ K)
Ve = np.sqrt((2*MTOW*9.81)/(rho_sl*S*Cle))*3.6
print("Best range speed in kph is: ",Ve)

"""