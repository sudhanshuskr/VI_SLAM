import matplotlib.pyplot as plt
import numpy as np
import time
plt.ion()


b = np.load('MAP100.npy', allow_pickle=True)
b = np.array(b)
b = b.astype(float)
figure, ax = plt.subplots(figsize=(10, 8))
line1 = ax.imshow(b,cmap="gist_yarg")


for i in range(24):
    b = np.load('MAP' + str((i+1)*100) + '.npy', allow_pickle=True)
    b = np.array(b)
    b = b.astype(float)
    line1.set_data(b)
    # drawing updated values
    figure.canvas.draw()
 
    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    figure.canvas.flush_events()
 
    