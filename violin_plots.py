import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    nd = np.load("Phantoms/type_d_phantoms.npy").shape[0]
    no = np.load("Phantoms/other_phantoms.npy").shape[0]

    train_ratio = 20/35

    fwi_low_noise_rrmse = np.load("Errors/fwi_low_noise_rrmse.npy")
    fwi_low_noise_rrmse_train = fwi_low_noise_rrmse[:int(train_ratio*nd)]
    fwi_low_noise_rrmse_train = np.concatenate((fwi_low_noise_rrmse_train, fwi_low_noise_rrmse[nd:nd + int(train_ratio*no)]))
    fwi_low_noise_rrmse_test = fwi_low_noise_rrmse[int(train_ratio*nd):nd]
    fwi_low_noise_rrmse_test = np.concatenate((fwi_low_noise_rrmse_test, fwi_low_noise_rrmse[nd + int(train_ratio*no):]))

    fwi_low_noise_ssim = np.load("Errors/fwi_low_noise_ssim.npy")
    fwi_low_noise_ssim_train = fwi_low_noise_ssim[:int(train_ratio*nd)]
    fwi_low_noise_ssim_train = np.concatenate((fwi_low_noise_ssim_train, fwi_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    fwi_low_noise_ssim_test = fwi_low_noise_ssim[int(train_ratio*nd):nd]
    fwi_low_noise_ssim_test = np.concatenate((fwi_low_noise_ssim_test, fwi_low_noise_ssim[nd + int(train_ratio*no):]))


    born_low_noise_rrmse = np.load("Errors/born_low_noise_rrmse.npy")
    born_low_noise_rrmse_train = born_low_noise_rrmse[:int(train_ratio*nd)]
    born_low_noise_rrmse_train = np.concatenate((born_low_noise_rrmse_train, born_low_noise_rrmse[nd:nd + int(train_ratio*no)]))
    born_low_noise_rrmse_test = born_low_noise_rrmse[int(train_ratio*nd):nd]
    born_low_noise_rrmse_test = np.concatenate((born_low_noise_rrmse_test, born_low_noise_rrmse[nd + int(train_ratio*no):]))

    born_low_noise_ssim = np.load("Errors/born_low_noise_ssim.npy")
    born_low_noise_ssim_train = born_low_noise_ssim[:int(train_ratio*nd)]
    born_low_noise_ssim_train = np.concatenate((born_low_noise_ssim_train, born_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    born_low_noise_ssim_test = born_low_noise_ssim[int(train_ratio*nd):nd]
    born_low_noise_ssim_test = np.concatenate((born_low_noise_ssim_test, born_low_noise_ssim[nd + int(train_ratio*no):]))

    art_low_noise_rrmse = np.load("Errors/art_low_noise_rrmse.npy")
    art_low_noise_rrmse_train = art_low_noise_rrmse[:int(train_ratio*nd)]
    art_low_noise_rrmse_train = np.concatenate((art_low_noise_rrmse_train, art_low_noise_rrmse[nd:nd + int(train_ratio*no)]))
    art_low_noise_rrmse_test = art_low_noise_rrmse[int(train_ratio*nd):nd]
    art_low_noise_rrmse_test = np.concatenate((art_low_noise_rrmse_test, art_low_noise_rrmse[nd + int(train_ratio*no):]))

    art_low_noise_ssim = np.load("Errors/art_low_noise_ssim.npy")
    art_low_noise_ssim_train = art_low_noise_ssim[:int(train_ratio*nd)]
    art_low_noise_ssim_train = np.concatenate((art_low_noise_ssim_train, art_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    art_low_noise_ssim_test = art_low_noise_ssim[int(train_ratio*nd):nd]
    art_low_noise_ssim_test = np.concatenate((art_low_noise_ssim_test, art_low_noise_ssim[nd + int(train_ratio*no):]))

    dc_low_noise_rrmse = np.load("Errors/dc_low_noise_rrmse.npy")
    dc_low_noise_rrmse_train = dc_low_noise_rrmse[:int(train_ratio*nd)]
    dc_low_noise_rrmse_train = np.concatenate((dc_low_noise_rrmse_train, dc_low_noise_rrmse[nd:nd + int(train_ratio*no)]))
    dc_low_noise_rrmse_test = dc_low_noise_rrmse[int(train_ratio*nd):nd]
    dc_low_noise_rrmse_test = np.concatenate((dc_low_noise_rrmse_test, dc_low_noise_rrmse[nd + int(train_ratio*no):]))

    dc_low_noise_ssim = np.load("Errors/dc_low_noise_ssim.npy")
    dc_low_noise_ssim_train = dc_low_noise_ssim[:int(train_ratio*nd)]
    dc_low_noise_ssim_train = np.concatenate((dc_low_noise_ssim_train, dc_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    dc_low_noise_ssim_test = dc_low_noise_ssim[int(train_ratio*nd):nd]
    dc_low_noise_ssim_test = np.concatenate((dc_low_noise_ssim_test, dc_low_noise_ssim[nd + int(train_ratio*no):]))

    dual_low_noise_rrmse = np.load("Errors/dual_low_noise_rrmse.npy")
    dual_low_noise_rrmse_train = dual_low_noise_rrmse[:int(train_ratio*nd)]
    dual_low_noise_rrmse_train = np.concatenate((dual_low_noise_rrmse_train, dual_low_noise_rrmse[nd:nd + int(train_ratio*no)]))
    dual_low_noise_rrmse_test = dual_low_noise_rrmse[int(train_ratio*nd):nd]
    dual_low_noise_rrmse_test = np.concatenate((dual_low_noise_rrmse_test, dual_low_noise_rrmse[nd + int(train_ratio*no):]))

    dual_low_noise_ssim = np.load("Errors/dual_low_noise_ssim.npy")
    dual_low_noise_ssim_train = dual_low_noise_ssim[:int(train_ratio*nd)]
    dual_low_noise_ssim_train = np.concatenate((dual_low_noise_ssim_train, dual_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    dual_low_noise_ssim_test = dual_low_noise_ssim[int(train_ratio*nd):nd]
    dual_low_noise_ssim_test = np.concatenate((dual_low_noise_ssim_test, dual_low_noise_ssim[nd + int(train_ratio*no):]))

    inet_low_noise_rrmse = np.load("Errors/inet_low_noise_rrmse.npy")
    inet_low_noise_rrmse_train = inet_low_noise_rrmse[:int(train_ratio*nd)]
    inet_low_noise_rrmse_train = np.concatenate((inet_low_noise_rrmse_train, inet_low_noise_rrmse[nd:nd + int(train_ratio*no)]))
    inet_low_noise_rrmse_test = inet_low_noise_rrmse[int(train_ratio*nd):nd]
    inet_low_noise_rrmse_test = np.concatenate((inet_low_noise_rrmse_test, inet_low_noise_rrmse[nd + int(train_ratio*no):]))

    inet_low_noise_ssim = np.load("Errors/inet_low_noise_ssim.npy")
    inet_low_noise_ssim_train = inet_low_noise_ssim[:int(train_ratio*nd)]
    inet_low_noise_ssim_train = np.concatenate((inet_low_noise_ssim_train, inet_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    inet_low_noise_ssim_test = inet_low_noise_ssim[int(train_ratio*nd):nd]
    inet_low_noise_ssim_test = np.concatenate((inet_low_noise_ssim_test, inet_low_noise_ssim[nd + int(train_ratio*no):]))


    rrmse_train = [fwi_low_noise_rrmse_train, 
                   inet_low_noise_rrmse_train,
                   born_low_noise_rrmse_train,
                   art_low_noise_rrmse_train,
                   dc_low_noise_rrmse_train,
                   dual_low_noise_rrmse_train]
    
    ssim_train = [fwi_low_noise_ssim_train, 
                   inet_low_noise_ssim_train,
                   born_low_noise_ssim_train,
                   art_low_noise_ssim_train,
                   dc_low_noise_ssim_train,
                   dual_low_noise_ssim_train]
    
    rrmse_test = [fwi_low_noise_rrmse_test, 
                   inet_low_noise_rrmse_test,
                   born_low_noise_rrmse_test,
                   art_low_noise_rrmse_test,
                   dc_low_noise_rrmse_test,
                   dual_low_noise_rrmse_test]
    
    ssim_test = [fwi_low_noise_ssim_test, 
                   inet_low_noise_ssim_test,
                   born_low_noise_ssim_test,
                   art_low_noise_ssim_test,
                   dc_low_noise_ssim_test,
                   dual_low_noise_ssim_test]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey= True)
    ax1.violinplot(rrmse_train, showmedians=True)
    ax2.violinplot(rrmse_test, showmedians=True)
    plt.savefig("Figures/rrmse_low_noise.png")
    

    



    fwi_medium_noise_rrmse = np.load("Errors/fwi_medium_noise_rrmse.npy")
    fwi_medium_noise_rrmse_train = fwi_medium_noise_rrmse[:int(train_ratio*nd)]
    fwi_medium_noise_rrmse_train = np.concatenate((fwi_medium_noise_rrmse_train, fwi_medium_noise_rrmse[nd:nd + int(train_ratio*no)]))
    fwi_medium_noise_rrmse_test = fwi_medium_noise_rrmse[int(train_ratio*nd):nd]
    fwi_medium_noise_rrmse_test = np.concatenate((fwi_medium_noise_rrmse_test, fwi_medium_noise_rrmse[nd + int(train_ratio*no):]))

    fwi_medium_noise_ssim = np.load("Errors/fwi_medium_noise_ssim.npy")
    fwi_medium_noise_ssim_train = fwi_medium_noise_ssim[:int(train_ratio*nd)]
    fwi_medium_noise_ssim_train = np.concatenate((fwi_medium_noise_ssim_train, fwi_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    fwi_medium_noise_ssim_test = fwi_medium_noise_ssim[int(train_ratio*nd):nd]
    fwi_medium_noise_ssim_test = np.concatenate((fwi_medium_noise_ssim_test, fwi_medium_noise_ssim[nd + int(train_ratio*no):]))


    born_medium_noise_rrmse = np.load("Errors/born_medium_noise_rrmse.npy")
    born_medium_noise_rrmse_train = born_medium_noise_rrmse[:int(train_ratio*nd)]
    born_medium_noise_rrmse_train = np.concatenate((born_medium_noise_rrmse_train, born_medium_noise_rrmse[nd:nd + int(train_ratio*no)]))
    born_medium_noise_rrmse_test = born_medium_noise_rrmse[int(train_ratio*nd):nd]
    born_medium_noise_rrmse_test = np.concatenate((born_medium_noise_rrmse_test, born_medium_noise_rrmse[nd + int(train_ratio*no):]))

    born_medium_noise_ssim = np.load("Errors/born_medium_noise_ssim.npy")
    born_medium_noise_ssim_train = born_medium_noise_ssim[:int(train_ratio*nd)]
    born_medium_noise_ssim_train = np.concatenate((born_medium_noise_ssim_train, born_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    born_medium_noise_ssim_test = born_medium_noise_ssim[int(train_ratio*nd):nd]
    born_medium_noise_ssim_test = np.concatenate((born_medium_noise_ssim_test, born_medium_noise_ssim[nd + int(train_ratio*no):]))

    art_medium_noise_rrmse = np.load("Errors/art_medium_noise_rrmse.npy")
    art_medium_noise_rrmse_train = art_medium_noise_rrmse[:int(train_ratio*nd)]
    art_medium_noise_rrmse_train = np.concatenate((art_medium_noise_rrmse_train, art_medium_noise_rrmse[nd:nd + int(train_ratio*no)]))
    art_medium_noise_rrmse_test = art_medium_noise_rrmse[int(train_ratio*nd):nd]
    art_medium_noise_rrmse_test = np.concatenate((art_medium_noise_rrmse_test, art_medium_noise_rrmse[nd + int(train_ratio*no):]))

    art_medium_noise_ssim = np.load("Errors/art_medium_noise_ssim.npy")
    art_medium_noise_ssim_train = art_medium_noise_ssim[:int(train_ratio*nd)]
    art_medium_noise_ssim_train = np.concatenate((art_medium_noise_ssim_train, art_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    art_medium_noise_ssim_test = art_medium_noise_ssim[int(train_ratio*nd):nd]
    art_medium_noise_ssim_test = np.concatenate((art_medium_noise_ssim_test, art_medium_noise_ssim[nd + int(train_ratio*no):]))

    dc_medium_noise_rrmse = np.load("Errors/dc_medium_noise_rrmse.npy")
    dc_medium_noise_rrmse_train = dc_medium_noise_rrmse[:int(train_ratio*nd)]
    dc_medium_noise_rrmse_train = np.concatenate((dc_medium_noise_rrmse_train, dc_medium_noise_rrmse[nd:nd + int(train_ratio*no)]))
    dc_medium_noise_rrmse_test = dc_medium_noise_rrmse[int(train_ratio*nd):nd]
    dc_medium_noise_rrmse_test = np.concatenate((dc_medium_noise_rrmse_test, dc_medium_noise_rrmse[nd + int(train_ratio*no):]))

    dc_medium_noise_ssim = np.load("Errors/dc_medium_noise_ssim.npy")
    dc_medium_noise_ssim_train = dc_medium_noise_ssim[:int(train_ratio*nd)]
    dc_medium_noise_ssim_train = np.concatenate((dc_medium_noise_ssim_train, dc_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    dc_medium_noise_ssim_test = dc_medium_noise_ssim[int(train_ratio*nd):nd]
    dc_medium_noise_ssim_test = np.concatenate((dc_medium_noise_ssim_test, dc_medium_noise_ssim[nd + int(train_ratio*no):]))

    dual_medium_noise_rrmse = np.load("Errors/dual_medium_noise_rrmse.npy")
    dual_medium_noise_rrmse_train = dual_medium_noise_rrmse[:int(train_ratio*nd)]
    dual_medium_noise_rrmse_train = np.concatenate((dual_medium_noise_rrmse_train, dual_medium_noise_rrmse[nd:nd + int(train_ratio*no)]))
    dual_medium_noise_rrmse_test = dual_medium_noise_rrmse[int(train_ratio*nd):nd]
    dual_medium_noise_rrmse_test = np.concatenate((dual_medium_noise_rrmse_test, dual_medium_noise_rrmse[nd + int(train_ratio*no):]))

    dual_medium_noise_ssim = np.load("Errors/dual_medium_noise_ssim.npy")
    dual_medium_noise_ssim_train = dual_medium_noise_ssim[:int(train_ratio*nd)]
    dual_medium_noise_ssim_train = np.concatenate((dual_medium_noise_ssim_train, dual_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    dual_medium_noise_ssim_test = dual_medium_noise_ssim[int(train_ratio*nd):nd]
    dual_medium_noise_ssim_test = np.concatenate((dual_medium_noise_ssim_test, dual_medium_noise_ssim[nd + int(train_ratio*no):]))

    inet_medium_noise_rrmse = np.load("Errors/inet_medium_noise_rrmse.npy")
    inet_medium_noise_rrmse_train = inet_medium_noise_rrmse[:int(train_ratio*nd)]
    inet_medium_noise_rrmse_train = np.concatenate((inet_medium_noise_rrmse_train, inet_medium_noise_rrmse[nd:nd + int(train_ratio*no)]))
    inet_medium_noise_rrmse_test = inet_medium_noise_rrmse[int(train_ratio*nd):nd]
    inet_medium_noise_rrmse_test = np.concatenate((inet_medium_noise_rrmse_test, inet_medium_noise_rrmse[nd + int(train_ratio*no):]))

    inet_medium_noise_ssim = np.load("Errors/inet_medium_noise_ssim.npy")
    inet_medium_noise_ssim_train = inet_medium_noise_ssim[:int(train_ratio*nd)]
    inet_medium_noise_ssim_train = np.concatenate((inet_medium_noise_ssim_train, inet_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    inet_medium_noise_ssim_test = inet_medium_noise_ssim[int(train_ratio*nd):nd]
    inet_medium_noise_ssim_test = np.concatenate((inet_medium_noise_ssim_test, inet_medium_noise_ssim[nd + int(train_ratio*no):]))


    rrmse_train = [fwi_medium_noise_rrmse_train, 
                   inet_medium_noise_rrmse_train,
                   born_medium_noise_rrmse_train,
                   art_medium_noise_rrmse_train,
                   dc_medium_noise_rrmse_train,
                   dual_medium_noise_rrmse_train]
    
    ssim_train = [fwi_medium_noise_ssim_train, 
                   inet_medium_noise_ssim_train,
                   born_medium_noise_ssim_train,
                   art_medium_noise_ssim_train,
                   dc_medium_noise_ssim_train,
                   dual_medium_noise_ssim_train]
    
    rrmse_test = [fwi_medium_noise_rrmse_test, 
                   inet_medium_noise_rrmse_test,
                   born_medium_noise_rrmse_test,
                   art_medium_noise_rrmse_test,
                   dc_medium_noise_rrmse_test,
                   dual_medium_noise_rrmse_test]
    
    ssim_test = [fwi_medium_noise_ssim_test, 
                   inet_medium_noise_ssim_test,
                   born_medium_noise_ssim_test,
                   art_medium_noise_ssim_test,
                   dc_medium_noise_ssim_test,
                   dual_medium_noise_ssim_test]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey= True)
    ax1.violinplot(rrmse_train, showmedians=True)
    ax2.violinplot(rrmse_test, showmedians=True)
    plt.savefig("Figures/rrmse_medium_noise.png")
    

    



    fwi_high_noise_rrmse = np.load("Errors/fwi_high_noise_rrmse.npy")
    fwi_high_noise_rrmse_train = fwi_high_noise_rrmse[:int(train_ratio*nd)]
    fwi_high_noise_rrmse_train = np.concatenate((fwi_high_noise_rrmse_train, fwi_high_noise_rrmse[nd:nd + int(train_ratio*no)]))
    fwi_high_noise_rrmse_test = fwi_high_noise_rrmse[int(train_ratio*nd):nd]
    fwi_high_noise_rrmse_test = np.concatenate((fwi_high_noise_rrmse_test, fwi_high_noise_rrmse[nd + int(train_ratio*no):]))

    fwi_high_noise_ssim = np.load("Errors/fwi_high_noise_ssim.npy")
    fwi_high_noise_ssim_train = fwi_high_noise_ssim[:int(train_ratio*nd)]
    fwi_high_noise_ssim_train = np.concatenate((fwi_high_noise_ssim_train, fwi_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    fwi_high_noise_ssim_test = fwi_high_noise_ssim[int(train_ratio*nd):nd]
    fwi_high_noise_ssim_test = np.concatenate((fwi_high_noise_ssim_test, fwi_high_noise_ssim[nd + int(train_ratio*no):]))


    born_high_noise_rrmse = np.load("Errors/born_high_noise_rrmse.npy")
    born_high_noise_rrmse_train = born_high_noise_rrmse[:int(train_ratio*nd)]
    born_high_noise_rrmse_train = np.concatenate((born_high_noise_rrmse_train, born_high_noise_rrmse[nd:nd + int(train_ratio*no)]))
    born_high_noise_rrmse_test = born_high_noise_rrmse[int(train_ratio*nd):nd]
    born_high_noise_rrmse_test = np.concatenate((born_high_noise_rrmse_test, born_high_noise_rrmse[nd + int(train_ratio*no):]))

    born_high_noise_ssim = np.load("Errors/born_high_noise_ssim.npy")
    born_high_noise_ssim_train = born_high_noise_ssim[:int(train_ratio*nd)]
    born_high_noise_ssim_train = np.concatenate((born_high_noise_ssim_train, born_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    born_high_noise_ssim_test = born_high_noise_ssim[int(train_ratio*nd):nd]
    born_high_noise_ssim_test = np.concatenate((born_high_noise_ssim_test, born_high_noise_ssim[nd + int(train_ratio*no):]))

    art_high_noise_rrmse = np.load("Errors/art_high_noise_rrmse.npy")
    art_high_noise_rrmse_train = art_high_noise_rrmse[:int(train_ratio*nd)]
    art_high_noise_rrmse_train = np.concatenate((art_high_noise_rrmse_train, art_high_noise_rrmse[nd:nd + int(train_ratio*no)]))
    art_high_noise_rrmse_test = art_high_noise_rrmse[int(train_ratio*nd):nd]
    art_high_noise_rrmse_test = np.concatenate((art_high_noise_rrmse_test, art_high_noise_rrmse[nd + int(train_ratio*no):]))

    art_high_noise_ssim = np.load("Errors/art_high_noise_ssim.npy")
    art_high_noise_ssim_train = art_high_noise_ssim[:int(train_ratio*nd)]
    art_high_noise_ssim_train = np.concatenate((art_high_noise_ssim_train, art_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    art_high_noise_ssim_test = art_high_noise_ssim[int(train_ratio*nd):nd]
    art_high_noise_ssim_test = np.concatenate((art_high_noise_ssim_test, art_high_noise_ssim[nd + int(train_ratio*no):]))

    dc_high_noise_rrmse = np.load("Errors/dc_high_noise_rrmse.npy")
    dc_high_noise_rrmse_train = dc_high_noise_rrmse[:int(train_ratio*nd)]
    dc_high_noise_rrmse_train = np.concatenate((dc_high_noise_rrmse_train, dc_high_noise_rrmse[nd:nd + int(train_ratio*no)]))
    dc_high_noise_rrmse_test = dc_high_noise_rrmse[int(train_ratio*nd):nd]
    dc_high_noise_rrmse_test = np.concatenate((dc_high_noise_rrmse_test, dc_high_noise_rrmse[nd + int(train_ratio*no):]))

    dc_high_noise_ssim = np.load("Errors/dc_high_noise_ssim.npy")
    dc_high_noise_ssim_train = dc_high_noise_ssim[:int(train_ratio*nd)]
    dc_high_noise_ssim_train = np.concatenate((dc_high_noise_ssim_train, dc_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    dc_high_noise_ssim_test = dc_high_noise_ssim[int(train_ratio*nd):nd]
    dc_high_noise_ssim_test = np.concatenate((dc_high_noise_ssim_test, dc_high_noise_ssim[nd + int(train_ratio*no):]))

    dual_high_noise_rrmse = np.load("Errors/dual_high_noise_rrmse.npy")
    dual_high_noise_rrmse_train = dual_high_noise_rrmse[:int(train_ratio*nd)]
    dual_high_noise_rrmse_train = np.concatenate((dual_high_noise_rrmse_train, dual_high_noise_rrmse[nd:nd + int(train_ratio*no)]))
    dual_high_noise_rrmse_test = dual_high_noise_rrmse[int(train_ratio*nd):nd]
    dual_high_noise_rrmse_test = np.concatenate((dual_high_noise_rrmse_test, dual_high_noise_rrmse[nd + int(train_ratio*no):]))

    dual_high_noise_ssim = np.load("Errors/dual_high_noise_ssim.npy")
    dual_high_noise_ssim_train = dual_high_noise_ssim[:int(train_ratio*nd)]
    dual_high_noise_ssim_train = np.concatenate((dual_high_noise_ssim_train, dual_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    dual_high_noise_ssim_test = dual_high_noise_ssim[int(train_ratio*nd):nd]
    dual_high_noise_ssim_test = np.concatenate((dual_high_noise_ssim_test, dual_high_noise_ssim[nd + int(train_ratio*no):]))

    inet_high_noise_rrmse = np.load("Errors/inet_high_noise_rrmse.npy")
    inet_high_noise_rrmse_train = inet_high_noise_rrmse[:int(train_ratio*nd)]
    inet_high_noise_rrmse_train = np.concatenate((inet_high_noise_rrmse_train, inet_high_noise_rrmse[nd:nd + int(train_ratio*no)]))
    inet_high_noise_rrmse_test = inet_high_noise_rrmse[int(train_ratio*nd):nd]
    inet_high_noise_rrmse_test = np.concatenate((inet_high_noise_rrmse_test, inet_high_noise_rrmse[nd + int(train_ratio*no):]))

    inet_high_noise_ssim = np.load("Errors/inet_high_noise_ssim.npy")
    inet_high_noise_ssim_train = inet_high_noise_ssim[:int(train_ratio*nd)]
    inet_high_noise_ssim_train = np.concatenate((inet_high_noise_ssim_train, inet_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    inet_high_noise_ssim_test = inet_high_noise_ssim[int(train_ratio*nd):nd]
    inet_high_noise_ssim_test = np.concatenate((inet_high_noise_ssim_test, inet_high_noise_ssim[nd + int(train_ratio*no):]))


    rrmse_train = [fwi_high_noise_rrmse_train, 
                   inet_high_noise_rrmse_train,
                   born_high_noise_rrmse_train,
                   art_high_noise_rrmse_train,
                   dc_high_noise_rrmse_train,
                   dual_high_noise_rrmse_train]
    
    ssim_train = [fwi_high_noise_ssim_train, 
                   inet_high_noise_ssim_train,
                   born_high_noise_ssim_train,
                   art_high_noise_ssim_train,
                   dc_high_noise_ssim_train,
                   dual_high_noise_ssim_train]
    
    rrmse_test = [fwi_high_noise_rrmse_test, 
                   inet_high_noise_rrmse_test,
                   born_high_noise_rrmse_test,
                   art_high_noise_rrmse_test,
                   dc_high_noise_rrmse_test,
                   dual_high_noise_rrmse_test]
    
    ssim_test = [fwi_high_noise_ssim_test, 
                   inet_high_noise_ssim_test,
                   born_high_noise_ssim_test,
                   art_high_noise_ssim_test,
                   dc_high_noise_ssim_test,
                   dual_high_noise_ssim_test]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey= True)
    ax1.violinplot(rrmse_train, showmedians=True)
    ax2.violinplot(rrmse_test, showmedians=True)
    plt.savefig("Figures/rrmse_high_noise.png")

    rrmses_train = [fwi_low_noise_rrmse_train, fwi_medium_noise_rrmse_train, fwi_high_noise_rrmse_train, 
                   inet_low_noise_rrmse_train, inet_medium_noise_rrmse_train, inet_high_noise_rrmse_train,
                   born_low_noise_rrmse_train,born_medium_noise_rrmse_train,born_high_noise_rrmse_train,
                   art_low_noise_rrmse_train,art_medium_noise_rrmse_train,art_high_noise_rrmse_train,
                   dc_low_noise_rrmse_train,dc_medium_noise_rrmse_train,dc_high_noise_rrmse_train,
                   dual_low_noise_rrmse_train,dual_medium_noise_rrmse_train,dual_high_noise_rrmse_train]
    

    rrmses_test = [fwi_low_noise_rrmse_test, fwi_medium_noise_rrmse_test, fwi_high_noise_rrmse_test, 
                   inet_low_noise_rrmse_test, inet_medium_noise_rrmse_test, inet_high_noise_rrmse_test,
                   born_low_noise_rrmse_test,born_medium_noise_rrmse_test,born_high_noise_rrmse_test,
                   art_low_noise_rrmse_test,art_medium_noise_rrmse_test,art_high_noise_rrmse_test,
                   dc_low_noise_rrmse_test,dc_medium_noise_rrmse_test,dc_high_noise_rrmse_test,
                   dual_low_noise_rrmse_test,dual_medium_noise_rrmse_test,dual_high_noise_rrmse_test]
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey= True)
    f.set_size_inches(30, 10.5)
    ax1.violinplot(rrmses_train, showmedians=True)
    ax2.violinplot(rrmses_test, showmedians=True)
    plt.savefig("Figures/rrmse_combined_noises.png")

    fwi_low_noise_ssim = np.load("Errors/fwi_low_noise_ssim.npy")
    fwi_low_noise_ssim_train = fwi_low_noise_ssim[:int(train_ratio*nd)]
    fwi_low_noise_ssim_train = np.concatenate((fwi_low_noise_ssim_train, fwi_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    fwi_low_noise_ssim_test = fwi_low_noise_ssim[int(train_ratio*nd):nd]
    fwi_low_noise_ssim_test = np.concatenate((fwi_low_noise_ssim_test, fwi_low_noise_ssim[nd + int(train_ratio*no):]))

    fwi_low_noise_ssim = np.load("Errors/fwi_low_noise_ssim.npy")
    fwi_low_noise_ssim_train = fwi_low_noise_ssim[:int(train_ratio*nd)]
    fwi_low_noise_ssim_train = np.concatenate((fwi_low_noise_ssim_train, fwi_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    fwi_low_noise_ssim_test = fwi_low_noise_ssim[int(train_ratio*nd):nd]
    fwi_low_noise_ssim_test = np.concatenate((fwi_low_noise_ssim_test, fwi_low_noise_ssim[nd + int(train_ratio*no):]))


    born_low_noise_ssim = np.load("Errors/born_low_noise_ssim.npy")
    born_low_noise_ssim_train = born_low_noise_ssim[:int(train_ratio*nd)]
    born_low_noise_ssim_train = np.concatenate((born_low_noise_ssim_train, born_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    born_low_noise_ssim_test = born_low_noise_ssim[int(train_ratio*nd):nd]
    born_low_noise_ssim_test = np.concatenate((born_low_noise_ssim_test, born_low_noise_ssim[nd + int(train_ratio*no):]))

    born_low_noise_ssim = np.load("Errors/born_low_noise_ssim.npy")
    born_low_noise_ssim_train = born_low_noise_ssim[:int(train_ratio*nd)]
    born_low_noise_ssim_train = np.concatenate((born_low_noise_ssim_train, born_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    born_low_noise_ssim_test = born_low_noise_ssim[int(train_ratio*nd):nd]
    born_low_noise_ssim_test = np.concatenate((born_low_noise_ssim_test, born_low_noise_ssim[nd + int(train_ratio*no):]))

    art_low_noise_ssim = np.load("Errors/art_low_noise_ssim.npy")
    art_low_noise_ssim_train = art_low_noise_ssim[:int(train_ratio*nd)]
    art_low_noise_ssim_train = np.concatenate((art_low_noise_ssim_train, art_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    art_low_noise_ssim_test = art_low_noise_ssim[int(train_ratio*nd):nd]
    art_low_noise_ssim_test = np.concatenate((art_low_noise_ssim_test, art_low_noise_ssim[nd + int(train_ratio*no):]))

    art_low_noise_ssim = np.load("Errors/art_low_noise_ssim.npy")
    art_low_noise_ssim_train = art_low_noise_ssim[:int(train_ratio*nd)]
    art_low_noise_ssim_train = np.concatenate((art_low_noise_ssim_train, art_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    art_low_noise_ssim_test = art_low_noise_ssim[int(train_ratio*nd):nd]
    art_low_noise_ssim_test = np.concatenate((art_low_noise_ssim_test, art_low_noise_ssim[nd + int(train_ratio*no):]))

    dc_low_noise_ssim = np.load("Errors/dc_low_noise_ssim.npy")
    dc_low_noise_ssim_train = dc_low_noise_ssim[:int(train_ratio*nd)]
    dc_low_noise_ssim_train = np.concatenate((dc_low_noise_ssim_train, dc_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    dc_low_noise_ssim_test = dc_low_noise_ssim[int(train_ratio*nd):nd]
    dc_low_noise_ssim_test = np.concatenate((dc_low_noise_ssim_test, dc_low_noise_ssim[nd + int(train_ratio*no):]))

    dc_low_noise_ssim = np.load("Errors/dc_low_noise_ssim.npy")
    dc_low_noise_ssim_train = dc_low_noise_ssim[:int(train_ratio*nd)]
    dc_low_noise_ssim_train = np.concatenate((dc_low_noise_ssim_train, dc_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    dc_low_noise_ssim_test = dc_low_noise_ssim[int(train_ratio*nd):nd]
    dc_low_noise_ssim_test = np.concatenate((dc_low_noise_ssim_test, dc_low_noise_ssim[nd + int(train_ratio*no):]))

    dual_low_noise_ssim = np.load("Errors/dual_low_noise_ssim.npy")
    dual_low_noise_ssim_train = dual_low_noise_ssim[:int(train_ratio*nd)]
    dual_low_noise_ssim_train = np.concatenate((dual_low_noise_ssim_train, dual_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    dual_low_noise_ssim_test = dual_low_noise_ssim[int(train_ratio*nd):nd]
    dual_low_noise_ssim_test = np.concatenate((dual_low_noise_ssim_test, dual_low_noise_ssim[nd + int(train_ratio*no):]))

    dual_low_noise_ssim = np.load("Errors/dual_low_noise_ssim.npy")
    dual_low_noise_ssim_train = dual_low_noise_ssim[:int(train_ratio*nd)]
    dual_low_noise_ssim_train = np.concatenate((dual_low_noise_ssim_train, dual_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    dual_low_noise_ssim_test = dual_low_noise_ssim[int(train_ratio*nd):nd]
    dual_low_noise_ssim_test = np.concatenate((dual_low_noise_ssim_test, dual_low_noise_ssim[nd + int(train_ratio*no):]))

    inet_low_noise_ssim = np.load("Errors/inet_low_noise_ssim.npy")
    inet_low_noise_ssim_train = inet_low_noise_ssim[:int(train_ratio*nd)]
    inet_low_noise_ssim_train = np.concatenate((inet_low_noise_ssim_train, inet_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    inet_low_noise_ssim_test = inet_low_noise_ssim[int(train_ratio*nd):nd]
    inet_low_noise_ssim_test = np.concatenate((inet_low_noise_ssim_test, inet_low_noise_ssim[nd + int(train_ratio*no):]))

    inet_low_noise_ssim = np.load("Errors/inet_low_noise_ssim.npy")
    inet_low_noise_ssim_train = inet_low_noise_ssim[:int(train_ratio*nd)]
    inet_low_noise_ssim_train = np.concatenate((inet_low_noise_ssim_train, inet_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    inet_low_noise_ssim_test = inet_low_noise_ssim[int(train_ratio*nd):nd]
    inet_low_noise_ssim_test = np.concatenate((inet_low_noise_ssim_test, inet_low_noise_ssim[nd + int(train_ratio*no):]))


    ssim_train = [fwi_low_noise_ssim_train, 
                   inet_low_noise_ssim_train,
                   born_low_noise_ssim_train,
                   art_low_noise_ssim_train,
                   dc_low_noise_ssim_train,
                   dual_low_noise_ssim_train]
    
    ssim_train = [fwi_low_noise_ssim_train, 
                   inet_low_noise_ssim_train,
                   born_low_noise_ssim_train,
                   art_low_noise_ssim_train,
                   dc_low_noise_ssim_train,
                   dual_low_noise_ssim_train]
    
    ssim_test = [fwi_low_noise_ssim_test, 
                   inet_low_noise_ssim_test,
                   born_low_noise_ssim_test,
                   art_low_noise_ssim_test,
                   dc_low_noise_ssim_test,
                   dual_low_noise_ssim_test]
    
    ssim_test = [fwi_low_noise_ssim_test, 
                   inet_low_noise_ssim_test,
                   born_low_noise_ssim_test,
                   art_low_noise_ssim_test,
                   dc_low_noise_ssim_test,
                   dual_low_noise_ssim_test]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey= True)
    ax1.violinplot(ssim_train, showmedians=True)
    ax2.violinplot(ssim_test, showmedians=True)
    plt.savefig("Figures/ssim_low_noise.png")
    

    



    fwi_medium_noise_ssim = np.load("Errors/fwi_medium_noise_ssim.npy")
    fwi_medium_noise_ssim_train = fwi_medium_noise_ssim[:int(train_ratio*nd)]
    fwi_medium_noise_ssim_train = np.concatenate((fwi_medium_noise_ssim_train, fwi_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    fwi_medium_noise_ssim_test = fwi_medium_noise_ssim[int(train_ratio*nd):nd]
    fwi_medium_noise_ssim_test = np.concatenate((fwi_medium_noise_ssim_test, fwi_medium_noise_ssim[nd + int(train_ratio*no):]))

    fwi_medium_noise_ssim = np.load("Errors/fwi_medium_noise_ssim.npy")
    fwi_medium_noise_ssim_train = fwi_medium_noise_ssim[:int(train_ratio*nd)]
    fwi_medium_noise_ssim_train = np.concatenate((fwi_medium_noise_ssim_train, fwi_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    fwi_medium_noise_ssim_test = fwi_medium_noise_ssim[int(train_ratio*nd):nd]
    fwi_medium_noise_ssim_test = np.concatenate((fwi_medium_noise_ssim_test, fwi_medium_noise_ssim[nd + int(train_ratio*no):]))


    born_medium_noise_ssim = np.load("Errors/born_medium_noise_ssim.npy")
    born_medium_noise_ssim_train = born_medium_noise_ssim[:int(train_ratio*nd)]
    born_medium_noise_ssim_train = np.concatenate((born_medium_noise_ssim_train, born_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    born_medium_noise_ssim_test = born_medium_noise_ssim[int(train_ratio*nd):nd]
    born_medium_noise_ssim_test = np.concatenate((born_medium_noise_ssim_test, born_medium_noise_ssim[nd + int(train_ratio*no):]))

    born_medium_noise_ssim = np.load("Errors/born_medium_noise_ssim.npy")
    born_medium_noise_ssim_train = born_medium_noise_ssim[:int(train_ratio*nd)]
    born_medium_noise_ssim_train = np.concatenate((born_medium_noise_ssim_train, born_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    born_medium_noise_ssim_test = born_medium_noise_ssim[int(train_ratio*nd):nd]
    born_medium_noise_ssim_test = np.concatenate((born_medium_noise_ssim_test, born_medium_noise_ssim[nd + int(train_ratio*no):]))

    art_medium_noise_ssim = np.load("Errors/art_medium_noise_ssim.npy")
    art_medium_noise_ssim_train = art_medium_noise_ssim[:int(train_ratio*nd)]
    art_medium_noise_ssim_train = np.concatenate((art_medium_noise_ssim_train, art_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    art_medium_noise_ssim_test = art_medium_noise_ssim[int(train_ratio*nd):nd]
    art_medium_noise_ssim_test = np.concatenate((art_medium_noise_ssim_test, art_medium_noise_ssim[nd + int(train_ratio*no):]))

    art_medium_noise_ssim = np.load("Errors/art_medium_noise_ssim.npy")
    art_medium_noise_ssim_train = art_medium_noise_ssim[:int(train_ratio*nd)]
    art_medium_noise_ssim_train = np.concatenate((art_medium_noise_ssim_train, art_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    art_medium_noise_ssim_test = art_medium_noise_ssim[int(train_ratio*nd):nd]
    art_medium_noise_ssim_test = np.concatenate((art_medium_noise_ssim_test, art_medium_noise_ssim[nd + int(train_ratio*no):]))

    dc_medium_noise_ssim = np.load("Errors/dc_medium_noise_ssim.npy")
    dc_medium_noise_ssim_train = dc_medium_noise_ssim[:int(train_ratio*nd)]
    dc_medium_noise_ssim_train = np.concatenate((dc_medium_noise_ssim_train, dc_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    dc_medium_noise_ssim_test = dc_medium_noise_ssim[int(train_ratio*nd):nd]
    dc_medium_noise_ssim_test = np.concatenate((dc_medium_noise_ssim_test, dc_medium_noise_ssim[nd + int(train_ratio*no):]))

    dc_medium_noise_ssim = np.load("Errors/dc_medium_noise_ssim.npy")
    dc_medium_noise_ssim_train = dc_medium_noise_ssim[:int(train_ratio*nd)]
    dc_medium_noise_ssim_train = np.concatenate((dc_medium_noise_ssim_train, dc_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    dc_medium_noise_ssim_test = dc_medium_noise_ssim[int(train_ratio*nd):nd]
    dc_medium_noise_ssim_test = np.concatenate((dc_medium_noise_ssim_test, dc_medium_noise_ssim[nd + int(train_ratio*no):]))

    dual_medium_noise_ssim = np.load("Errors/dual_medium_noise_ssim.npy")
    dual_medium_noise_ssim_train = dual_medium_noise_ssim[:int(train_ratio*nd)]
    dual_medium_noise_ssim_train = np.concatenate((dual_medium_noise_ssim_train, dual_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    dual_medium_noise_ssim_test = dual_medium_noise_ssim[int(train_ratio*nd):nd]
    dual_medium_noise_ssim_test = np.concatenate((dual_medium_noise_ssim_test, dual_medium_noise_ssim[nd + int(train_ratio*no):]))

    dual_medium_noise_ssim = np.load("Errors/dual_medium_noise_ssim.npy")
    dual_medium_noise_ssim_train = dual_medium_noise_ssim[:int(train_ratio*nd)]
    dual_medium_noise_ssim_train = np.concatenate((dual_medium_noise_ssim_train, dual_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    dual_medium_noise_ssim_test = dual_medium_noise_ssim[int(train_ratio*nd):nd]
    dual_medium_noise_ssim_test = np.concatenate((dual_medium_noise_ssim_test, dual_medium_noise_ssim[nd + int(train_ratio*no):]))

    inet_medium_noise_ssim = np.load("Errors/inet_medium_noise_ssim.npy")
    inet_medium_noise_ssim_train = inet_medium_noise_ssim[:int(train_ratio*nd)]
    inet_medium_noise_ssim_train = np.concatenate((inet_medium_noise_ssim_train, inet_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    inet_medium_noise_ssim_test = inet_medium_noise_ssim[int(train_ratio*nd):nd]
    inet_medium_noise_ssim_test = np.concatenate((inet_medium_noise_ssim_test, inet_medium_noise_ssim[nd + int(train_ratio*no):]))

    inet_medium_noise_ssim = np.load("Errors/inet_medium_noise_ssim.npy")
    inet_medium_noise_ssim_train = inet_medium_noise_ssim[:int(train_ratio*nd)]
    inet_medium_noise_ssim_train = np.concatenate((inet_medium_noise_ssim_train, inet_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    inet_medium_noise_ssim_test = inet_medium_noise_ssim[int(train_ratio*nd):nd]
    inet_medium_noise_ssim_test = np.concatenate((inet_medium_noise_ssim_test, inet_medium_noise_ssim[nd + int(train_ratio*no):]))


    ssim_train = [fwi_medium_noise_ssim_train, 
                   inet_medium_noise_ssim_train,
                   born_medium_noise_ssim_train,
                   art_medium_noise_ssim_train,
                   dc_medium_noise_ssim_train,
                   dual_medium_noise_ssim_train]
    
    ssim_train = [fwi_medium_noise_ssim_train, 
                   inet_medium_noise_ssim_train,
                   born_medium_noise_ssim_train,
                   art_medium_noise_ssim_train,
                   dc_medium_noise_ssim_train,
                   dual_medium_noise_ssim_train]
    
    ssim_test = [fwi_medium_noise_ssim_test, 
                   inet_medium_noise_ssim_test,
                   born_medium_noise_ssim_test,
                   art_medium_noise_ssim_test,
                   dc_medium_noise_ssim_test,
                   dual_medium_noise_ssim_test]
    
    ssim_test = [fwi_medium_noise_ssim_test, 
                   inet_medium_noise_ssim_test,
                   born_medium_noise_ssim_test,
                   art_medium_noise_ssim_test,
                   dc_medium_noise_ssim_test,
                   dual_medium_noise_ssim_test]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey= True)
    ax1.violinplot(ssim_train, showmedians=True)
    ax2.violinplot(ssim_test, showmedians=True)
    plt.savefig("Figures/ssim_medium_noise.png")
    

    



    fwi_high_noise_ssim = np.load("Errors/fwi_high_noise_ssim.npy")
    fwi_high_noise_ssim_train = fwi_high_noise_ssim[:int(train_ratio*nd)]
    fwi_high_noise_ssim_train = np.concatenate((fwi_high_noise_ssim_train, fwi_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    fwi_high_noise_ssim_test = fwi_high_noise_ssim[int(train_ratio*nd):nd]
    fwi_high_noise_ssim_test = np.concatenate((fwi_high_noise_ssim_test, fwi_high_noise_ssim[nd + int(train_ratio*no):]))

    fwi_high_noise_ssim = np.load("Errors/fwi_high_noise_ssim.npy")
    fwi_high_noise_ssim_train = fwi_high_noise_ssim[:int(train_ratio*nd)]
    fwi_high_noise_ssim_train = np.concatenate((fwi_high_noise_ssim_train, fwi_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    fwi_high_noise_ssim_test = fwi_high_noise_ssim[int(train_ratio*nd):nd]
    fwi_high_noise_ssim_test = np.concatenate((fwi_high_noise_ssim_test, fwi_high_noise_ssim[nd + int(train_ratio*no):]))


    born_high_noise_ssim = np.load("Errors/born_high_noise_ssim.npy")
    born_high_noise_ssim_train = born_high_noise_ssim[:int(train_ratio*nd)]
    born_high_noise_ssim_train = np.concatenate((born_high_noise_ssim_train, born_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    born_high_noise_ssim_test = born_high_noise_ssim[int(train_ratio*nd):nd]
    born_high_noise_ssim_test = np.concatenate((born_high_noise_ssim_test, born_high_noise_ssim[nd + int(train_ratio*no):]))

    born_high_noise_ssim = np.load("Errors/born_high_noise_ssim.npy")
    born_high_noise_ssim_train = born_high_noise_ssim[:int(train_ratio*nd)]
    born_high_noise_ssim_train = np.concatenate((born_high_noise_ssim_train, born_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    born_high_noise_ssim_test = born_high_noise_ssim[int(train_ratio*nd):nd]
    born_high_noise_ssim_test = np.concatenate((born_high_noise_ssim_test, born_high_noise_ssim[nd + int(train_ratio*no):]))

    art_high_noise_ssim = np.load("Errors/art_high_noise_ssim.npy")
    art_high_noise_ssim_train = art_high_noise_ssim[:int(train_ratio*nd)]
    art_high_noise_ssim_train = np.concatenate((art_high_noise_ssim_train, art_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    art_high_noise_ssim_test = art_high_noise_ssim[int(train_ratio*nd):nd]
    art_high_noise_ssim_test = np.concatenate((art_high_noise_ssim_test, art_high_noise_ssim[nd + int(train_ratio*no):]))

    art_high_noise_ssim = np.load("Errors/art_high_noise_ssim.npy")
    art_high_noise_ssim_train = art_high_noise_ssim[:int(train_ratio*nd)]
    art_high_noise_ssim_train = np.concatenate((art_high_noise_ssim_train, art_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    art_high_noise_ssim_test = art_high_noise_ssim[int(train_ratio*nd):nd]
    art_high_noise_ssim_test = np.concatenate((art_high_noise_ssim_test, art_high_noise_ssim[nd + int(train_ratio*no):]))

    dc_high_noise_ssim = np.load("Errors/dc_high_noise_ssim.npy")
    dc_high_noise_ssim_train = dc_high_noise_ssim[:int(train_ratio*nd)]
    dc_high_noise_ssim_train = np.concatenate((dc_high_noise_ssim_train, dc_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    dc_high_noise_ssim_test = dc_high_noise_ssim[int(train_ratio*nd):nd]
    dc_high_noise_ssim_test = np.concatenate((dc_high_noise_ssim_test, dc_high_noise_ssim[nd + int(train_ratio*no):]))

    dc_high_noise_ssim = np.load("Errors/dc_high_noise_ssim.npy")
    dc_high_noise_ssim_train = dc_high_noise_ssim[:int(train_ratio*nd)]
    dc_high_noise_ssim_train = np.concatenate((dc_high_noise_ssim_train, dc_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    dc_high_noise_ssim_test = dc_high_noise_ssim[int(train_ratio*nd):nd]
    dc_high_noise_ssim_test = np.concatenate((dc_high_noise_ssim_test, dc_high_noise_ssim[nd + int(train_ratio*no):]))

    dual_high_noise_ssim = np.load("Errors/dual_high_noise_ssim.npy")
    dual_high_noise_ssim_train = dual_high_noise_ssim[:int(train_ratio*nd)]
    dual_high_noise_ssim_train = np.concatenate((dual_high_noise_ssim_train, dual_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    dual_high_noise_ssim_test = dual_high_noise_ssim[int(train_ratio*nd):nd]
    dual_high_noise_ssim_test = np.concatenate((dual_high_noise_ssim_test, dual_high_noise_ssim[nd + int(train_ratio*no):]))

    dual_high_noise_ssim = np.load("Errors/dual_high_noise_ssim.npy")
    dual_high_noise_ssim_train = dual_high_noise_ssim[:int(train_ratio*nd)]
    dual_high_noise_ssim_train = np.concatenate((dual_high_noise_ssim_train, dual_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    dual_high_noise_ssim_test = dual_high_noise_ssim[int(train_ratio*nd):nd]
    dual_high_noise_ssim_test = np.concatenate((dual_high_noise_ssim_test, dual_high_noise_ssim[nd + int(train_ratio*no):]))

    inet_high_noise_ssim = np.load("Errors/inet_high_noise_ssim.npy")
    inet_high_noise_ssim_train = inet_high_noise_ssim[:int(train_ratio*nd)]
    inet_high_noise_ssim_train = np.concatenate((inet_high_noise_ssim_train, inet_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    inet_high_noise_ssim_test = inet_high_noise_ssim[int(train_ratio*nd):nd]
    inet_high_noise_ssim_test = np.concatenate((inet_high_noise_ssim_test, inet_high_noise_ssim[nd + int(train_ratio*no):]))

    inet_high_noise_ssim = np.load("Errors/inet_high_noise_ssim.npy")
    inet_high_noise_ssim_train = inet_high_noise_ssim[:int(train_ratio*nd)]
    inet_high_noise_ssim_train = np.concatenate((inet_high_noise_ssim_train, inet_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    inet_high_noise_ssim_test = inet_high_noise_ssim[int(train_ratio*nd):nd]
    inet_high_noise_ssim_test = np.concatenate((inet_high_noise_ssim_test, inet_high_noise_ssim[nd + int(train_ratio*no):]))


    ssim_train = [fwi_high_noise_ssim_train, 
                   inet_high_noise_ssim_train,
                   born_high_noise_ssim_train,
                   art_high_noise_ssim_train,
                   dc_high_noise_ssim_train,
                   dual_high_noise_ssim_train]
    
    ssim_train = [fwi_high_noise_ssim_train, 
                   inet_high_noise_ssim_train,
                   born_high_noise_ssim_train,
                   art_high_noise_ssim_train,
                   dc_high_noise_ssim_train,
                   dual_high_noise_ssim_train]
    
    ssim_test = [fwi_high_noise_ssim_test, 
                   inet_high_noise_ssim_test,
                   born_high_noise_ssim_test,
                   art_high_noise_ssim_test,
                   dc_high_noise_ssim_test,
                   dual_high_noise_ssim_test]
    
    ssim_test = [fwi_high_noise_ssim_test, 
                   inet_high_noise_ssim_test,
                   born_high_noise_ssim_test,
                   art_high_noise_ssim_test,
                   dc_high_noise_ssim_test,
                   dual_high_noise_ssim_test]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey= True)
    ax1.violinplot(ssim_train, showmedians=True)
    ax2.violinplot(ssim_test, showmedians=True)
    plt.savefig("Figures/ssim_high_noise.png")

    ssims_train = [fwi_low_noise_ssim_train, fwi_medium_noise_ssim_train, fwi_high_noise_ssim_train, 
                   inet_low_noise_ssim_train, inet_medium_noise_ssim_train, inet_high_noise_ssim_train,
                   born_low_noise_ssim_train,born_medium_noise_ssim_train,born_high_noise_ssim_train,
                   art_low_noise_ssim_train,art_medium_noise_ssim_train,art_high_noise_ssim_train,
                   dc_low_noise_ssim_train,dc_medium_noise_ssim_train,dc_high_noise_ssim_train,
                   dual_low_noise_ssim_train,dual_medium_noise_ssim_train,dual_high_noise_ssim_train]
    

    ssims_test = [fwi_low_noise_ssim_test, fwi_medium_noise_ssim_test, fwi_high_noise_ssim_test, 
                   inet_low_noise_ssim_test, inet_medium_noise_ssim_test, inet_high_noise_ssim_test,
                   born_low_noise_ssim_test,born_medium_noise_ssim_test,born_high_noise_ssim_test,
                   art_low_noise_ssim_test,art_medium_noise_ssim_test,art_high_noise_ssim_test,
                   dc_low_noise_ssim_test,dc_medium_noise_ssim_test,dc_high_noise_ssim_test,
                   dual_low_noise_ssim_test,dual_medium_noise_ssim_test,dual_high_noise_ssim_test]
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey= True)
    f.set_size_inches(30, 10.5)
    ax1.violinplot(ssims_train, showmedians=True)
    ax2.violinplot(ssims_test, showmedians=True)
    plt.savefig("Figures/ssim_combined_noises.png")
