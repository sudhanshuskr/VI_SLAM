import matplotlib.pyplot as plt
import numpy as np
import time



b = np.load('MAP1800.npy', allow_pickle=True)
b = np.array(b)
b = b.astype(float)
plt.figure(1)
plt.imshow(b,cmap="gist_yarg")
plt.title('Occupancy grid map')
plt.xlabel('y (cells)')
plt.ylabel('x (cells)')
plt.show()
