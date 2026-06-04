#局部相似性
from matplotlib.colors import LinearSegmentedColormap
if __name__ == '__main__':    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter

    def local_similarity(data1, data2, window_size=11):
        assert data1.shape == data2.shape, "输入数据维度不匹配"
        assert window_size % 2 == 1, "窗口大小需为奇数"
        prod = data1 * data2
        sum_prod = uniform_filter(prod, size=window_size, mode='reflect')
        sum11 = uniform_filter(data1**2, size=window_size, mode='reflect')
        sum22 = uniform_filter(data2**2, size=window_size, mode='reflect')
        epsilon = 1e-10
        denominator = np.sqrt(sum11 * sum22) + epsilon
        sim_map = (sum_prod) / denominator
        return sim_map
    
    # if i == 1:
    #     result = np.load(r"D:\桌面\项目\Stratton\数据汇总\Black Bridge\Black_clean.npy")
    #     pure = np.load(r"D:\桌面\项目\Stratton\数据汇总\Black Bridge\BLACK_CONVENTIONAL_NOISE.npy")
    # elif i == 2:
    #     result = np.load(r"D:\桌面\项目\Stratton\数据汇总\Black Bridge\Black_clean.npy")
    #     pure = np.load(r"D:\桌面\项目\Stratton\数据汇总\Black Bridge\Black_UNet_noise.npy")  
    # else :
    #     result = np.load(r"D:\桌面\项目\Stratton\数据汇总\Black Bridge\Black_clean.npy")
    #     pure = np.load(r"D:\桌面\项目\Stratton\数据汇总\Black Bridge\Black_UNet3+_noise.npy")
    i = 1
    if i == 1:
        pure = np.load(r"D:\桌面\面波压制\Synthetic data test 2\合成含噪记录制作\noise.npy")[100:500,5:43]
        result = np.load(r"D:\桌面\面波压制\Synthetic data test 2\DMSSL\noise_0.npy")[100:500,5:43]
    elif i == 2:
        pure = np.load(r"D:\桌面\面波压制\Synthetic data test 2\合成含噪记录制作\clean.npy")[100:500,5:43]
        result = np.load(r"D:\桌面\面波压制\Synthetic data test 2\U-Net\noise_0.npy")[100:500,5:43]
    else :
        pure = np.load(r"D:\桌面\面波压制\Synthetic data test 2\合成含噪记录制作\clean.npy")[100:500,5:43]
        result = np.load(r"D:\桌面\面波压制\Synthetic data test 2\DMSSL\denoised_shot_0.npy")[100:500,5:43]

    colors = [
        (0.0, '#7fff7f'),  # 亮绿色
        (0.2, '#b3ff66'),  # 浅绿色
        (0.4, '#ffff00'),  # 黄色
        (0.6, '#ff9900'),  # 橙色
        (0.8, '#ff4d00'),  # 红橙色
        (1.0, '#990000')   # 深红色
    ]
    cust_cmap = LinearSegmentedColormap.from_list('seismic_green', colors)

    sim_cross = local_similarity(result, pure, window_size=9)
    # np.save(r"D:\桌面\项目\Stratton\数据汇总\MAMI\MAMI_Conventional_SSIM", sim_cross)
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_cross, cmap=cust_cmap, aspect='auto')  # viridis 
    plt.colorbar(label='相似性')
    plt.clim(0, 1)
    plt.show()