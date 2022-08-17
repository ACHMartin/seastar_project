# -*- coding: utf-8 -*-
"""
CDOP GMF based on Mouche et al., 2012, 
IEEE TGRS "On the Use of Doppler Shift for Sea Surface 
Wind Retrieval From SAR"

Calculates the Doppler frequency shift of sea surface winds based on an empirical
Geophysical Model Function. 

 Inputs
   - inc : incidence angle in degree
   - u10 : wind speed in m/s
   - wdir : wind direction in degree (0=upwind,90=crosswind,180=downwind)
   (wdir not in [0,180] are converted by the function, e.g. -45 becomes 45)
   - pol : polarization name, 'vv' or 'hh' (case insensitive)
 Output
   - dop : geophysical Doppler anomaly in Hz
 Neural network training limits :
   - inc --> [17,42]
   - u10 --> [1,17]
   - wdir --> [0,180]


Created on Wed Aug 10 09:22:12 2022

@author: davidmccann
"""

import numpy as np

# Get coefficients (W=weights and B=biases)
# (coefficient names in mouche2012 are given)

def cdop(inc,u10,wdir,pol):
    
    Ns=np.array([inc.size,u10.size,wdir.size])
    N=max(Ns)
    
    
    if pol=='VV':
        #lambda[0:2,1]
        B1 = np.array([-0.343935744939, 0.108823529412, 0.15])
        #lambda[0:2,0]
        W1 = np.array([0.028213254683, 0.0411764705882, .00388888888889])
        #omega[i,0]
        B2 = np.array([14.5077150927, -11.4312028555, 1.28692747109,
              -1.19498666071, 1.778908726, 11.8880215573,
              1.70176062351, 24.7941267067, -8.18756617111,
              1.32555779345, -9.06560116738])
        B2 = np.transpose(B2)
        #omega[i,[3,2,1]]
        W2 = np.array([[19.7873046673, 22.2237414308, 1.27887019276],
              [2.910815875, -3.63395681095, 16.4242081101],
              [1.03269004609, 0.403986575614, 0.325018607578],
              [3.17100261168, 4.47461213024, 0.969975702316],
              [-3.80611082432, -6.91334859293, -0.0162650756459],
              [4.09854466913, -1.64290475596, -13.4031862615],
              [0.484338480824, -1.30503436654, -6.04613303002],
              [-11.1000239122, 15.993470129, 23.2186869807],
              [-0.577883159569, 0.801977535733, 6.13874672206],
              [0.61008842868, -0.5009830671, -4.42736737765],
              [-1.94654022702, 1.31351068862, 8.94943709074]])
        #gamma[0]
        B3 = 4.07777876994
        #gamma[1:11]
        W3 = np.array([7.34881153553, 0.487879873912, -22.167664703,
              7.01176085914, 3.57021820094, -7.05653415486,
              -8.82147148713, 5.35079872715, 93.627037987,
              13.9420969201, -34.4032326496])
        #beta
        B4 = -52.2644487109
        #alpha
        W4 = 111.528184073
    elif pol=='HH':
        #lambda[0:2,1]
        B1 = np.array([-0.342097701547, 0.118181818182, 0.15])
        #lambda[0:2,0]
        W1 = np.array([0.0281843837385, 0.0318181818182, 0.00388888888889])
        #omega[i,0]
        B2 = np.array([1.30653883096, -2.77086154074, 10.6792861882,
              -4.0429666906, -0.172201666743, 20.4895916824,
              28.2856865516, -3.60143441597, -3.53935574111,
              -2.11695768022, -2.57805898849])
        B2 = np.transpose(B2)
        #omega[i,[3,2,1]]
        W2 = np.array([[-2.61087309812, -0.973599180956, -9.07176856257],
              [-0.246776181361, 0.586523978839, -0.594867645776],
              [17.9261562541, 12.9439063319, 16.9815377306],
              [0.595882115891, 6.20098098757, -9.20238868219],
              [-0.993509213443, 0.301856868548, -4.12397246171],
              [15.0224985357, 17.643307099, 8.57886720397],
              [13.1833641617, 20.6983195925, -15.1439734434],
              [0.656338134446, 5.79854593024, -9.9811757434],
              [0.122736690257, -5.67640781126, 11.9861607453],
              [0.691577162612, 5.95289490539, -16.0530462],
              [1.2664066483, 0.151056851685, 7.93435940581]]);
        #gamma[0]
        B3 = 2.68352095337
        #gamma[1:11]
        W3 = np.array([-8.21498722494, -94.9645431048, -17.7727420108,
              -63.3536337981, 39.2450482271, -6.15275352542,
              16.5337543167, 90.1967379935, -1.11346786284,
              -17.57689699, 8.20219395141])
        #beta
        B4 = -66.9554922921
        #alpha
        W4 = 136.216953823
    
    #Make inputs as matrix (and clip wdir in [0,180])
    inputs=np.zeros((3,N))
    if Ns[0]==1:
        inputs[0,0:N] = (W1[0] * inc) + B1[0]
    else:
        inputs[0,0:N] = (W1[0] * np.reshape(inc,(1,N))) + B1[0]
    #------------------------------
    if Ns[1]==1:
        inputs[1,0:N] = (W1[1] * u10) + B1[1]
    else:
        inputs[1,0:N] = (W1[1] * np.reshape(u10,(1,N))) + B1[1]
    #------------------------------
    if Ns[2]==1:
        # In the original matlab code this read:
        # W1(3).*acosd(cosd(wdir))
        # Which was deemed identical to abs(wdir)
        inputs[2,0:N] = (W1[2] * np.abs(wdir)) + B1[2]
    else:
        inputs[2,0:N] = (W1[2] * np.abs(np.reshape(wdir, (1,N)))) + B1[2]
    #------------------------------
    #Compute CDOP
    dop = W4 * cdop_func(W3 * cdop_func(W2 * inputs + np.tile(B2,(N,1))) + B3) + B4
    
    if (Ns[0]>Ns[1]) and (Ns[0]>Ns[2]):
        dop = np.reshape(dop, inc.shape)
    elif (Ns[1]>Ns[0]) and (Ns[1].Ns[2]):
        dop = np.reshape(dop, u10.shape)
    elif (Ns[2]>Ns[0]) and (Ns[2]>Ns[1]):
        dop = np.reshape(dop, wdir.shape)
    
            
    return dop
    

def cdop_func(x):
    """ Function to assist in CDOP calculation
    """
    cdop_f = 1 / (1+np.exp(-x))
    return cdop_f
    
      
    
    