import segyio
import numpy as np
import matplotlib.pyplot as plt

def read_segy(data_dir,shotnum=0):
    with segyio.open(data_dir,'r',ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX) #number of all trace
        if shotnum:
            shot_num = shotnum 
        else:
            shot_num = len(set(sourceX)) #shot number 
        len_shot = trace_num//shot_num   #The length of the data in each shot data
        time = f.trace[0].shape[0]
        print('start read segy data')
        data = np.zeros((shot_num,time,len_shot))
        for j in range(0,shot_num):
            data[j,:,:] = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
        return data
                                                                                                          
def gabor_transform(signal, dt, alpha=100, step=0.01):
    """
    对单道地震记录进行 Gabor 变换

    参数：
    signal : 1D numpy array，地震道信号
    dt : float，采样时间间隔（s）
    alpha : float，高斯窗参数 (1/alpha 控制窗宽度)
    step : float，滑动步长（s）

    返回：
    G : 2D numpy array，复数形式的 Gabor 时频谱
    t : numpy array，时间轴
    f : numpy array，频率轴
    """
    N = len(signal)
    t = np.arange(N) * dt
    f = np.fft.fftfreq(N, d=dt)
    
    step_samples = int(step / dt)  # 步长换算为采样点数
    num_windows = (N - step_samples) // step_samples

    G = np.zeros((len(f), num_windows), dtype=complex)
    tau_list = []

    for k in range(num_windows):
        tau = k * step_samples
        tau_time = tau * dt
        tau_list.append(tau_time)
        
        # 构造高斯窗函数 g(t)
        g = np.exp(-alpha * (t - tau_time) ** 2)
        g /= np.sqrt(np.pi / alpha)  # 对应公式(4)归一化
        
        # 计算时窗信号
        windowed = signal * g
        
        # 对时窗信号进行傅里叶变换 (对应式(1))
        G[:, k] = np.fft.fft(windowed)
    
    return G, np.array(tau_list), f

data = read_segy(r"D:\桌面\面波压制\data\f20_z_test+GathEP.sgy", shotnum=700)[0][:, 32]
dt = 0.00025  # 采样间隔（秒）
G, tau, f = gabor_transform(data, dt, alpha=400, step=0.01)

plt.figure(figsize=(10, 6))
plt.pcolormesh(tau, f[:len(f)//2], np.abs(G[:len(f)//2, :]), shading='auto')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Gabor Time-Frequency Spectrum')
plt.colorbar(label='Amplitude')
plt.show()

plt.plot(data)
plt.show()