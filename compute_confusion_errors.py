import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

if __name__ == "__main__":
    true_c = np.load("Phantoms/type_d_phantoms.npy")
    true_c2 = np.load("Phantoms/other_phantoms.npy")
    true_c = np.concatenate((true_c, true_c2), axis = 0)
    del true_c2
    true_c_norm = np.sum((true_c - 1.5)**2, axis = (1,2,3))**(1/2)


    art_low_medium = np.load("Recons/type_d_ac_recon_low_medium.npy")
    art_low_medium2 = np.load("Recons/other_ac_recon_low_medium.npy")
    art_low_medium = np.concatenate((art_low_medium, art_low_medium2), axis = 0)
    del art_low_medium2
    art_low_medium_rrmse = np.sum((true_c - art_low_medium)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/art_low_medium_rrmse.npy", art_low_medium_rrmse)
    del art_low_medium_rrmse
    art_low_medium_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], art_low_medium[j,0,:,:], data_range= 0.2)
        art_low_medium_ssim.append(plc_ssim)
    np.save("Errors/art_low_medium_ssim.npy", np.array(art_low_medium_ssim))
    del art_low_medium_ssim
    del art_low_medium



    art_low_high = np.load("Recons/type_d_ac_recon_low_high.npy")
    art_low_high2 = np.load("Recons/other_ac_recon_low_high.npy")
    art_low_high = np.concatenate((art_low_high, art_low_high2), axis = 0)
    del art_low_high2
    art_low_high_rrmse = np.sum((true_c - art_low_high)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/art_low_high_rrmse.npy", art_low_high_rrmse)
    del art_low_high_rrmse
    art_low_high_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], art_low_high[j,0,:,:], data_range= 0.2)
        art_low_high_ssim.append(plc_ssim)
    np.save("Errors/art_low_high_ssim.npy", np.array(art_low_high_ssim))
    del art_low_high_ssim
    del art_low_high


    dc_low_medium = np.load("Recons/type_d_dc_recon_low_medium.npy")
    dc_low_medium2 = np.load("Recons/other_dc_recon_low_medium.npy")
    dc_low_medium = np.concatenate((dc_low_medium, dc_low_medium2), axis = 0)
    del dc_low_medium2
    dc_low_medium_rrmse = np.sum((true_c - dc_low_medium)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dc_low_medium_rrmse.npy", dc_low_medium_rrmse)
    del dc_low_medium_rrmse
    dc_low_medium_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dc_low_medium[j,0,:,:], data_range= 0.2)
        dc_low_medium_ssim.append(plc_ssim)
    np.save("Errors/dc_low_medium_ssim.npy", np.array(dc_low_medium_ssim))
    del dc_low_medium_ssim
    del dc_low_medium

    dc_low_high = np.load("Recons/type_d_dc_recon_low_high.npy")
    dc_low_high2 = np.load("Recons/other_dc_recon_low_high.npy")
    dc_low_high = np.concatenate((dc_low_high, dc_low_high2), axis = 0)
    del dc_low_high2
    dc_low_high_rrmse = np.sum((true_c - dc_low_high)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dc_low_high_rrmse.npy", dc_low_high_rrmse)
    del dc_low_high_rrmse
    dc_low_high_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dc_low_high[j,0,:,:], data_range= 0.2)
        dc_low_high_ssim.append(plc_ssim)
    np.save("Errors/dc_low_high_ssim.npy", np.array(dc_low_high_ssim))
    del dc_low_high_ssim
    del dc_low_high


    dual_low_medium = np.load("Recons/type_d_dual_recon_low_medium.npy")
    dual_low_medium2 = np.load("Recons/other_dual_recon_low_medium.npy")
    dual_low_medium = np.concatenate((dual_low_medium, dual_low_medium2), axis = 0)
    del dual_low_medium2
    dual_low_medium_rrmse = np.sum((true_c - dual_low_medium)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dual_low_medium_rrmse.npy", dual_low_medium_rrmse)
    del dual_low_medium_rrmse
    dual_low_medium_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dual_low_medium[j,0,:,:], data_range= 0.2)
        dual_low_medium_ssim.append(plc_ssim)
    np.save("Errors/dual_low_medium_ssim.npy", np.array(dual_low_medium_ssim))
    del dual_low_medium_ssim
    del dual_low_medium



    dual_low_high = np.load("Recons/type_d_dual_recon_low_high.npy")
    dual_low_high2 = np.load("Recons/other_dual_recon_low_high.npy")
    dual_low_high = np.concatenate((dual_low_high, dual_low_high2), axis = 0)
    del dual_low_high2
    dual_low_high_rrmse = np.sum((true_c - dual_low_high)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dual_low_high_rrmse.npy", dual_low_high_rrmse)
    del dual_low_high_rrmse
    dual_low_high_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dual_low_high[j,0,:,:], data_range= 0.2)
        dual_low_high_ssim.append(plc_ssim)
    np.save("Errors/dual_low_high_ssim.npy", np.array(dual_low_high_ssim))
    del dual_low_high_ssim
    del dual_low_high

    inet_low_medium = np.load("Recons/type_d_inet_recon_low_medium.npy")
    inet_low_medium2 = np.load("Recons/other_inet_recon_low_medium.npy")
    inet_low_medium = np.concatenate((inet_low_medium, inet_low_medium2), axis = 0)
    del inet_low_medium2
    inet_low_medium_rrmse = np.sum((true_c - inet_low_medium)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/inet_low_medium_rrmse.npy", inet_low_medium_rrmse)
    del inet_low_medium_rrmse
    inet_low_medium_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], inet_low_medium[j,0,:,:], data_range= 0.2)
        inet_low_medium_ssim.append(plc_ssim)
    np.save("Errors/inet_low_medium_ssim.npy", np.array(inet_low_medium_ssim))
    del inet_low_medium_ssim
    del inet_low_medium

    inet_low_high = np.load("Recons/type_d_inet_recon_low_high.npy")
    inet_low_high2 = np.load("Recons/other_inet_recon_low_high.npy")
    inet_low_high = np.concatenate((inet_low_high, inet_low_high2), axis = 0)
    del inet_low_high2
    inet_low_high_rrmse = np.sum((true_c - inet_low_high)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/inet_low_high_rrmse.npy", inet_low_high_rrmse)
    del inet_low_high_rrmse
    inet_low_high_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], inet_low_high[j,0,:,:], data_range= 0.2)
        inet_low_high_ssim.append(plc_ssim)
    np.save("Errors/inet_low_high_ssim.npy", np.array(inet_low_high_ssim))
    del inet_low_high_ssim
    del inet_low_high


    art_medium_low = np.load("Recons/type_d_ac_recon_medium_low.npy")
    art_medium_low2 = np.load("Recons/other_ac_recon_medium_low.npy")
    art_medium_low = np.concatenate((art_medium_low, art_medium_low2), axis = 0)
    del art_medium_low2
    art_medium_low_rrmse = np.sum((true_c - art_medium_low)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/art_medium_low_rrmse.npy", art_medium_low_rrmse)
    del art_medium_low_rrmse
    art_medium_low_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], art_medium_low[j,0,:,:], data_range= 0.2)
        art_medium_low_ssim.append(plc_ssim)
    np.save("Errors/art_medium_low_ssim.npy", np.array(art_medium_low_ssim))
    del art_medium_low_ssim
    del art_medium_low

    art_medium_high = np.load("Recons/type_d_ac_recon_medium_high.npy")
    art_medium_high2 = np.load("Recons/other_ac_recon_medium_high.npy")
    art_medium_high = np.concatenate((art_medium_high, art_medium_high2), axis = 0)
    del art_medium_high2
    art_medium_high_rrmse = np.sum((true_c - art_medium_high)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/art_medium_high_rrmse.npy", art_medium_high_rrmse)
    del art_medium_high_rrmse
    art_medium_high_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], art_medium_high[j,0,:,:], data_range= 0.2)
        art_medium_high_ssim.append(plc_ssim)
    np.save("Errors/art_medium_high_ssim.npy", np.array(art_medium_high_ssim))
    del art_medium_high_ssim
    del art_medium_high


    dc_medium_low = np.load("Recons/type_d_dc_recon_medium_low.npy")
    dc_medium_low2 = np.load("Recons/other_dc_recon_medium_low.npy")
    dc_medium_low = np.concatenate((dc_medium_low, dc_medium_low2), axis = 0)
    del dc_medium_low2
    dc_medium_low_rrmse = np.sum((true_c - dc_medium_low)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dc_medium_low_rrmse.npy", dc_medium_low_rrmse)
    del dc_medium_low_rrmse
    dc_medium_low_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dc_medium_low[j,0,:,:], data_range= 0.2)
        dc_medium_low_ssim.append(plc_ssim)
    np.save("Errors/dc_medium_low_ssim.npy", np.array(dc_medium_low_ssim))
    del dc_medium_low_ssim
    del dc_medium_low


    dc_medium_high = np.load("Recons/type_d_dc_recon_medium_high.npy")
    dc_medium_high2 = np.load("Recons/other_dc_recon_medium_high.npy")
    dc_medium_high = np.concatenate((dc_medium_high, dc_medium_high2), axis = 0)
    del dc_medium_high2
    dc_medium_high_rrmse = np.sum((true_c - dc_medium_high)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dc_medium_high_rrmse.npy", dc_medium_high_rrmse)
    del dc_medium_high_rrmse
    dc_medium_high_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dc_medium_high[j,0,:,:], data_range= 0.2)
        dc_medium_high_ssim.append(plc_ssim)
    np.save("Errors/dc_medium_high_ssim.npy", np.array(dc_medium_high_ssim))
    del dc_medium_high_ssim
    del dc_medium_high


    dual_medium_low = np.load("Recons/type_d_dual_recon_medium_low.npy")
    dual_medium_low2 = np.load("Recons/other_dual_recon_medium_low.npy")
    dual_medium_low = np.concatenate((dual_medium_low, dual_medium_low2), axis = 0)
    del dual_medium_low2
    dual_medium_low_rrmse = np.sum((true_c - dual_medium_low)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dual_medium_low_rrmse.npy", dual_medium_low_rrmse)
    del dual_medium_low_rrmse
    dual_medium_low_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dual_medium_low[j,0,:,:], data_range= 0.2)
        dual_medium_low_ssim.append(plc_ssim)
    np.save("Errors/dual_medium_low_ssim.npy", np.array(dual_medium_low_ssim))
    del dual_medium_low_ssim
    del dual_medium_low


    dual_medium_high = np.load("Recons/type_d_dual_recon_medium_high.npy")
    dual_medium_high2 = np.load("Recons/other_dual_recon_medium_high.npy")
    dual_medium_high = np.concatenate((dual_medium_high, dual_medium_high2), axis = 0)
    del dual_medium_high2
    dual_medium_high_rrmse = np.sum((true_c - dual_medium_high)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dual_medium_high_rrmse.npy", dual_medium_high_rrmse)
    del dual_medium_high_rrmse
    dual_medium_high_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dual_medium_high[j,0,:,:], data_range= 0.2)
        dual_medium_high_ssim.append(plc_ssim)
    np.save("Errors/dual_medium_high_ssim.npy", np.array(dual_medium_high_ssim))
    del dual_medium_high_ssim
    del dual_medium_high



    inet_medium_low = np.load("Recons/type_d_inet_recon_medium_low.npy")
    inet_medium_low2 = np.load("Recons/other_inet_recon_medium_low.npy")
    inet_medium_low = np.concatenate((inet_medium_low, inet_medium_low2), axis = 0)
    del inet_medium_low2
    inet_medium_low_rrmse = np.sum((true_c - inet_medium_low)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/inet_medium_low_rrmse.npy", inet_medium_low_rrmse)
    del inet_medium_low_rrmse
    inet_medium_low_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], inet_medium_low[j,0,:,:], data_range= 0.2)
        inet_medium_low_ssim.append(plc_ssim)
    np.save("Errors/inet_medium_low_ssim.npy", np.array(inet_medium_low_ssim))
    del inet_medium_low_ssim
    del inet_medium_low



    inet_medium_high = np.load("Recons/type_d_inet_recon_medium_high.npy")
    inet_medium_high2 = np.load("Recons/other_inet_recon_medium_high.npy")
    inet_medium_high = np.concatenate((inet_medium_high, inet_medium_high2), axis = 0)
    del inet_medium_high2
    inet_medium_high_rrmse = np.sum((true_c - inet_medium_high)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/inet_medium_high_rrmse.npy", inet_medium_high_rrmse)
    del inet_medium_high_rrmse
    inet_medium_high_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], inet_medium_high[j,0,:,:], data_range= 0.2)
        inet_medium_high_ssim.append(plc_ssim)
    np.save("Errors/inet_medium_high_ssim.npy", np.array(inet_medium_high_ssim))
    del inet_medium_high_ssim
    del inet_medium_high


    art_high_low = np.load("Recons/type_d_ac_recon_high_low.npy")
    art_high_low2 = np.load("Recons/other_ac_recon_high_low.npy")
    art_high_low = np.concatenate((art_high_low, art_high_low2), axis = 0)
    del art_high_low2
    art_high_low_rrmse = np.sum((true_c - art_high_low)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/art_high_low_rrmse.npy", art_high_low_rrmse)
    del art_high_low_rrmse
    art_high_low_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], art_high_low[j,0,:,:], data_range= 0.2)
        art_high_low_ssim.append(plc_ssim)
    np.save("Errors/art_high_low_ssim.npy", np.array(art_high_low_ssim))
    del art_high_low_ssim
    del art_high_low


    art_high_medium = np.load("Recons/type_d_ac_recon_high_medium.npy")
    art_high_medium2 = np.load("Recons/other_ac_recon_high_medium.npy")
    art_high_medium = np.concatenate((art_high_medium, art_high_medium2), axis = 0)
    del art_high_medium2
    art_high_medium_rrmse = np.sum((true_c - art_high_medium)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/art_high_medium_rrmse.npy", art_high_medium_rrmse)
    del art_high_medium_rrmse
    art_high_medium_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], art_high_medium[j,0,:,:], data_range= 0.2)
        art_high_medium_ssim.append(plc_ssim)
    np.save("Errors/art_high_medium_ssim.npy", np.array(art_high_medium_ssim))
    del art_high_medium_ssim
    del art_high_medium

    dc_high_low = np.load("Recons/type_d_dc_recon_high_low.npy")
    dc_high_low2 = np.load("Recons/other_dc_recon_high_low.npy")
    dc_high_low = np.concatenate((dc_high_low, dc_high_low2), axis = 0)
    del dc_high_low2
    dc_high_low_rrmse = np.sum((true_c - dc_high_low)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dc_high_low_rrmse.npy", dc_high_low_rrmse)
    del dc_high_low_rrmse
    dc_high_low_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dc_high_low[j,0,:,:], data_range= 0.2)
        dc_high_low_ssim.append(plc_ssim)
    np.save("Errors/dc_high_low_ssim.npy", np.array(dc_high_low_ssim))
    del dc_high_low_ssim
    del dc_high_low


    dc_high_medium = np.load("Recons/type_d_dc_recon_high_medium.npy")
    dc_high_medium2 = np.load("Recons/other_dc_recon_high_medium.npy")
    dc_high_medium = np.concatenate((dc_high_medium, dc_high_medium2), axis = 0)
    del dc_high_medium2
    dc_high_medium_rrmse = np.sum((true_c - dc_high_medium)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dc_high_medium_rrmse.npy", dc_high_medium_rrmse)
    del dc_high_medium_rrmse
    dc_high_medium_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dc_high_medium[j,0,:,:], data_range= 0.2)
        dc_high_medium_ssim.append(plc_ssim)
    np.save("Errors/dc_high_medium_ssim.npy", np.array(dc_high_medium_ssim))
    del dc_high_medium_ssim
    del dc_high_medium

    dual_high_low = np.load("Recons/type_d_dual_recon_high_low.npy")
    dual_high_low2 = np.load("Recons/other_dual_recon_high_low.npy")
    dual_high_low = np.concatenate((dual_high_low, dual_high_low2), axis = 0)
    del dual_high_low2
    dual_high_low_rrmse = np.sum((true_c - dual_high_low)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dual_high_low_rrmse.npy", dual_high_low_rrmse)
    del dual_high_low_rrmse
    dual_high_low_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dual_high_low[j,0,:,:], data_range= 0.2)
        dual_high_low_ssim.append(plc_ssim)
    np.save("Errors/dual_high_low_ssim.npy", np.array(dual_high_low_ssim))
    del dual_high_low_ssim
    del dual_high_low

    dual_high_medium = np.load("Recons/type_d_dual_recon_high_medium.npy")
    dual_high_medium2 = np.load("Recons/other_dual_recon_high_medium.npy")
    dual_high_medium = np.concatenate((dual_high_medium, dual_high_medium2), axis = 0)
    del dual_high_medium2
    dual_high_medium_rrmse = np.sum((true_c - dual_high_medium)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dual_high_medium_rrmse.npy", dual_high_medium_rrmse)
    del dual_high_medium_rrmse
    dual_high_medium_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dual_high_medium[j,0,:,:], data_range= 0.2)
        dual_high_medium_ssim.append(plc_ssim)
    np.save("Errors/dual_high_medium_ssim.npy", np.array(dual_high_medium_ssim))
    del dual_high_medium_ssim
    del dual_high_medium

    inet_high_low = np.load("Recons/type_d_inet_recon_high_low.npy")
    inet_high_low2 = np.load("Recons/other_inet_recon_high_low.npy")
    inet_high_low = np.concatenate((inet_high_low, inet_high_low2), axis = 0)
    del inet_high_low2
    inet_high_low_rrmse = np.sum((true_c - inet_high_low)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/inet_high_low_rrmse.npy", inet_high_low_rrmse)
    del inet_high_low_rrmse
    inet_high_low_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], inet_high_low[j,0,:,:], data_range= 0.2)
        inet_high_low_ssim.append(plc_ssim)
    np.save("Errors/inet_high_low_ssim.npy", np.array(inet_high_low_ssim))
    del inet_high_low_ssim
    del inet_high_low

    inet_high_medium = np.load("Recons/type_d_inet_recon_high_medium.npy")
    inet_high_medium2 = np.load("Recons/other_inet_recon_high_medium.npy")
    inet_high_medium = np.concatenate((inet_high_medium, inet_high_medium2), axis = 0)
    del inet_high_medium2
    inet_high_medium_rrmse = np.sum((true_c - inet_high_medium)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/inet_high_medium_rrmse.npy", inet_high_medium_rrmse)
    del inet_high_medium_rrmse
    inet_high_medium_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], inet_high_medium[j,0,:,:], data_range= 0.2)
        inet_high_medium_ssim.append(plc_ssim)
    np.save("Errors/inet_high_medium_ssim.npy", np.array(inet_high_medium_ssim))
    del inet_high_medium_ssim
    del inet_high_medium