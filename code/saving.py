import numpy as np


TIME_MAP = [1,1,1,1,1,1,1]
np.save('a.npy', np.array(TIME_MAP, dtype=object), allow_pickle=True)