import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

clean = np.load(r"D:\桌面\面波压制\方法对比\clean.npy")[0:1200,:]

noise = np.load(r"D:\桌面\面波压制\方法对比\本文方法\noise_50.npy")

noised = clean + noise*400
# np.save(r"D:\桌面\面波压制\方法对比\本文方法npy\noised.npy", noised)
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.title("Clean Data", fontsize=16)    
plt.imshow(clean, cmap='seismic', aspect='auto')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.title("Noised Data", fontsize=16)
plt.imshow(noised, cmap='seismic', aspect='auto')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.title("Noise Data", fontsize=16)
plt.imshow(noise, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.show()