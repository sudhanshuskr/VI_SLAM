from odom_mapping import *
import numpy as np
from pr2_utils import *
from tqdm import tqdm
from tkinter import *



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

N_PARTICLES =10
P = np.zeros((3,N_PARTICLES))

MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -10  #meters
MAP['ymin']  = -10
MAP['xmax']  =  30
MAP['ymax']  =  30 
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8

vel,v_timestep = pos2velocity()
vel,v_timestep = downsample_vel(vel,v_timestep)



trajectory = np.zeros((N_PARTICLES,len(v_timestep),3))


wz = imu_angular_velocity[2]
wz_timestep = imu_stamps
wz_freq = 1/np.average((wz_timestep[1:] - wz_timestep[0:-1]))
wz_fs = 1/wz_freq
wz_new = butter_lowpass_filter(wz, 8, wz_freq, 6)

plt.figure(1)
ax = plt.subplot(411)
#ax.plot(wz_timestep,wz_new)

print(trajectory[:,0])

wz = []
for i in range(len(v_timestep)):
    time = v_timestep[i]
    index = find_nearest(wz_timestep,time)
    wz.append(wz_new[index])
    
wz = np.array(wz) 

ax.plot(v_timestep,wz)
for i in range(len(v_timestep)-1):
    trajectory[:,i+1] = nextState(trajectory[:,i].T,vel[i],wz[i],
                                    v_timestep[i+1] - v_timestep[i],
                                    trajectory[:,i,2].T).T
    trajectory[:,i+1] = trajectory[:,i+1] + np.append(np.random.normal(0.0, 0.01, (N_PARTICLES*2)),
                                                      np.random.normal(0.0
                                            , 0.001, (N_PARTICLES*1))).reshape(3,N_PARTICLES).T

ay = plt.subplot(412)
ay.plot(trajectory[:,:,0].T,trajectory[:,:,1].T)
    
az = plt.subplot(413)
az.plot(trajectory[:,:,2].T)

plt.figure(2)
plt.plot(trajectory[:,:,0].T,trajectory[:,:,1].T)
plt.show()
   
plt.figure(1) 
for i in tqdm(range(len(v_timestep))):
    
    lidar_time_index = find_nearest(lidar_stamsp,v_timestep[i])
    ranges = lidar_ranges[:,lidar_time_index][::LID_SUBSAM]
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    angles = angles[::LID_SUBSAM]
    
  # take valid indices
    indValid = np.logical_and((ranges < lidar_range_max),(ranges> lidar_range_min))
    ranges = ranges[indValid]
    angles = angles[indValid] + trajectory[0,i,2]
    xp = trajectory[0,i,0]
    yp = trajectory[0,i,1]
    x0 = ranges*np.cos(angles) + xp
    y0 = ranges*np.sin(angles) + yp
    
    
    xi0 = np.ceil((x0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yi0 = np.ceil((y0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    
    
    xip = np.ceil((xp - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yip = np.ceil((yp - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    
    for j in range(len(xi0)):
        epx = xi0[j]
        epy = yi0[j]
        if(epx>=MAP['sizex']-1): epx = MAP['sizex']-1
        if(epx<=0): epx = 0
        if(epy>=MAP['sizey']-1): epy = MAP['sizey']-1
        if(epy<=0): epy = 0
        spx = xip
        spy = yip
        #print(i,angles[j])
        #print(round(spx),round(spy),round(epx),round(epy))
        XY = bresenham2D(spx, spy, epx, epy)
        #print(XY[0],XY[1])
        MAP['map'][XY[0].astype(int),XY[1].astype(int)] = -1
        MAP['map'][epx,epy] = 1

axx = plt.subplot(414)
axx.imshow(MAP['map'],cmap="hot")

plt.figure(2)
plt.imshow(MAP['map'],cmap="gist_yarg")


plt.show() 

        
        
        
    
#print(1/np.average((wz_timestep[1:] - wz_timestep[0:-1])))



#y = butter_lowpass_filter(data, cutoff, fs, order)

 

