



from odom_mapping import *
import numpy as np
from pr2_utils import *
from tqdm import tqdm

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def normalize_prob(prob):
    prob = prob/np.sum(prob)
    return prob

def strat_resample(prob,trajectory):
    Np = len(prob)
    new_states = np.zeros((Np,3))
    j = 0
    c = prob[j]
    for ind in range(Np):
        u = np.random.uniform(0,1/Np)
        beta = u + (j/Np)
        while (beta>c):
            j = j+1
            c = c + prob[j]
        new_states[ind,:] = trajectory[j,:]
    
    return (np.zeros(Np) + (1/Np)), new_states


N_PARTICLES = 800
CONF = 4
P = np.zeros((3,N_PARTICLES))

MAP = {}
MAP['res']   = 0.02 #meters
MAP['xmin']  = -16  #meters
MAP['ymin']  = -16
MAP['xmax']  =  30
MAP['ymax']  =  30 
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey'])) + (10*np.log(CONF)) #DATA TYPE: char or int8

vel,v_timestep = pos2velocity()
vel,v_timestep = downsample_vel(vel,v_timestep)



trajectory = np.zeros((N_PARTICLES,len(v_timestep),3))
prob = np.zeros(N_PARTICLES) + (1/N_PARTICLES)


wz = imu_angular_velocity[2]
wz_timestep = imu_stamps
wz_freq = 1/np.average((wz_timestep[1:] - wz_timestep[0:-1]))
wz_fs = 1/wz_freq
wz_new = butter_lowpass_filter(wz, 8, wz_freq, 6)


#ax.plot(wz_timestep,wz_new)


wz = []
for i in range(len(v_timestep)):
    time = v_timestep[i]
    index = find_nearest(wz_timestep,time)
    wz.append(wz_new[index])
    
wz = np.array(wz) 



lidar_time_index = find_nearest(lidar_stamsp,v_timestep[0])
ranges = lidar_ranges[:,lidar_time_index][::LID_SUBSAM]
angles = np.arange(-135,135.25,0.25)*np.pi/180.0
angles = angles[::LID_SUBSAM]

# take valid indices
indValid = np.logical_and((ranges < 5),(ranges> 2))
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
    MAP['map'][XY[0].astype(int),XY[1].astype(int)] = -np.log(CONF)
    MAP['map'][epx,epy] = np.log(CONF)


plt.ion()
figure, ax = plt.subplots(figsize=(10, 8))
img = ax.imshow(MAP['map'],cmap="hot")  

for times_exec in range(5):  
    for i in tqdm(range(len(v_timestep)-1)):
        
        trajectory[:,i+1] = nextState(trajectory[:,i].T,vel[i],wz[i],
                                        v_timestep[i+1] - v_timestep[i],
                                        trajectory[:,i,2].T).T
        trajectory[:,i+1] = trajectory[:,i+1] + np.append(np.random.normal(0.0, 0.01, (N_PARTICLES*2)),
                                                        np.random.normal(0.0
                                                , 0.001, (N_PARTICLES*1))).reshape(3,N_PARTICLES).T
        
        lidar_time_index = find_nearest(lidar_stamsp,v_timestep[i+1])
        ranges = lidar_ranges[:,lidar_time_index][::LID_SUBSAM]
        angles = np.arange(-135,135.25,0.25)*np.pi/180.0
        angles = angles[::LID_SUBSAM]
        
    # take valid indices
        indValid = np.logical_and((ranges < 5),(ranges> 2))
        ranges = ranges[indValid]
        angles = angles[indValid]
        
        
        x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
        y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
        corr = np.zeros(N_PARTICLES)
        
        MAP_COPY = np.copy(MAP['map'])
        '''
        for x_index in range(MAP['sizex']):
            for y_index in range(MAP['sizey']):
                if(MAP['map'][x_index,y_index]>0):
                    MAP_COPY[x_index,y_index] = 1
                if(MAP['map'][x_index,y_index]<0):
                    MAP_COPY[x_index,y_index] = -1
                else:
                    MAP_COPY[x_index,y_index] = 0
    
        '''
        for particle_index in range(N_PARTICLES):
            xp = trajectory[particle_index,i+1,0]
            yp = trajectory[particle_index,i+1,1]
            angles_local = angles + trajectory[particle_index,i+1,2]
            x0 = ranges*np.cos(angles_local) + xp
            y0 = ranges*np.sin(angles_local) + yp
            YY = np.stack((x0,y0))
            corr[particle_index] = mapCorrelation(MAP['map'],x_im,y_im,YY,np.array([xp]),np.array([yp]))[0,0]
            if (corr[particle_index]<1):
                corr[particle_index] = 1
            prob[particle_index] = prob[particle_index]*corr[particle_index]
        
        '''
        if (i>=380):
            print(prob, corr)
        '''
        prob = normalize_prob(prob)
        '''
        if (i>=380):
            print(prob,corr)
        '''
        max_index = np.argmax(prob)
        
        
        angles_local = angles + trajectory[max_index,i+1,2]
        xi0 = ranges*np.cos(angles_local)  + trajectory[max_index,i+1,0]
        yi0 = ranges*np.sin(angles_local) + trajectory[max_index,i+1,1]
        
        
        xi0 = np.ceil((xi0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        yi0 = np.ceil((yi0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        
        
        
        xip = np.ceil((trajectory[max_index,i+1,0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        yip = np.ceil((trajectory[max_index,i+1,1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        
        
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
            MAP['map'][XY[0].astype(int),XY[1].astype(int)] += -np.log(CONF)
            MAP['map'][epx,epy] += np.log(CONF)
            MAP['map'][np.less(MAP['map'],(-1*np.log(CONF)))] = -1*np.log(CONF)
            
            if (MAP['map'][epx,epy]>20*np.log(CONF)):
                MAP['map'][epx,epy]=20*np.log(CONF)
        
        img.set_data(MAP['map'])
        figure.canvas.draw()
        figure.canvas.flush_events()
        Neff = 1/np.dot(prob,prob)
        if (Neff <= (N_PARTICLES/10)):
            prob, trajectory[:,i+1,:] = strat_resample(prob,trajectory[:,i+1,:])

    times_exec +=1
    
    
for x_index in range(MAP['sizex']):
    for y_index in range(MAP['sizey']):
        lambda0 = MAP['map'][x_index,y_index]
        p = np.exp(lambda0)/(1+np.exp(lambda0))
        MAP['map'][x_index,y_index] = p


plt.figure(2)
plt.plot(trajectory[:,:,0].T,trajectory[:,:,1].T)
plt.show() 

        
        
        
    
#print(1/np.average((wz_timestep[1:] - wz_timestep[0:-1])))



#y = butter_lowpass_filter(data, cutoff, fs, order)

 

