import segyio
import numpy as np
import matplotlib.pyplot as plt

def normalize_traces_per_trace(shot_gather):
    """逐道归一化"""
    normalized_gather = np.copy(shot_gather)
    n_samples, n_traces = normalized_gather.shape
    
    for i in range(n_traces):
        trace = normalized_gather[:, i]
        max_amp = np.max(np.abs(trace))
        if max_amp > 0:
            normalized_gather[:, i] = trace / max_amp
    
    return normalized_gather

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
    
n = 0
for i in range(0, 16):
    for j in range(0, 12):
        data = read_segy(
            r"D:\桌面\深层数据面波压制\data\001_rawdata.sgy",
            shotnum=16
        )[i][:, 120*j:120*(j+1)]
        
        filename = fr"D:\桌面\深层数据面波压制\npy\real_data_gather{n:03d}.npy"
        np.save(filename, data)
        
        n += 1
    