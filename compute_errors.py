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
    

    fwi_low_noise = np.load("Recons/type_d_fwi_recon_low_noise.npy")
    fwi_low_noise2 = np.load("Recons/other_fwi_recon_low_noise.npy")
    fwi_low_noise = np.concatenate((fwi_low_noise, fwi_low_noise2), axis = 0)
    del fwi_low_noise2
    fwi_low_noise_rrmse = np.sum((true_c - fwi_low_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/fwi_low_noise_rrmse.npy", fwi_low_noise_rrmse)
    del fwi_low_noise_rrmse
    fwi_low_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], fwi_low_noise[j,0,:,:], data_range= 0.2)
        fwi_low_noise_ssim.append(plc_ssim)
    np.save("Errors/fwi_low_noise_ssim.npy", np.array(fwi_low_noise_ssim))
    del fwi_low_noise_ssim
    del fwi_low_noise
        


    born_low_noise = np.load("Recons/type_d_born_recon_low_noise.npy")
    born_low_noise2 = np.load("Recons/other_born_recon_low_noise.npy")
    born_low_noise = np.concatenate((born_low_noise, born_low_noise2), axis = 0)
    del born_low_noise2
    born_low_noise_rrmse = np.sum((true_c - born_low_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/born_low_noise_rrmse.npy", born_low_noise_rrmse)
    del born_low_noise_rrmse
    born_low_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], born_low_noise[j,0,:,:], data_range= 0.2)
        born_low_noise_ssim.append(plc_ssim)
    np.save("Errors/born_low_noise_ssim.npy", np.array(born_low_noise_ssim))
    del born_low_noise_ssim
    del born_low_noise
        

    art_low_noise = np.load("Recons/type_d_ac_recon_low_noise.npy")
    art_low_noise2 = np.load("Recons/other_ac_recon_low_noise.npy")
    art_low_noise = np.concatenate((art_low_noise, art_low_noise2), axis = 0)
    del art_low_noise2
    art_low_noise_rrmse = np.sum((true_c - art_low_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/art_low_noise_rrmse.npy", art_low_noise_rrmse)
    del art_low_noise_rrmse
    art_low_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], art_low_noise[j,0,:,:], data_range= 0.2)
        art_low_noise_ssim.append(plc_ssim)
    np.save("Errors/art_low_noise_ssim.npy", np.array(art_low_noise_ssim))
    del art_low_noise_ssim
    del art_low_noise
        

    dc_low_noise = np.load("Recons/type_d_dc_recon_low_noise.npy")
    dc_low_noise2 = np.load("Recons/other_dc_recon_low_noise.npy")
    dc_low_noise = np.concatenate((dc_low_noise, dc_low_noise2), axis = 0)
    del dc_low_noise2
    dc_low_noise_rrmse = np.sum((true_c - dc_low_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dc_low_noise_rrmse.npy", dc_low_noise_rrmse)
    del dc_low_noise_rrmse
    dc_low_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dc_low_noise[j,0,:,:], data_range= 0.2)
        dc_low_noise_ssim.append(plc_ssim)
    np.save("Errors/dc_low_noise_ssim.npy", np.array(dc_low_noise_ssim))
    del dc_low_noise_ssim
    del dc_low_noise
        

    dual_low_noise = np.load("Recons/type_d_dual_recon_low_noise.npy")
    dual_low_noise2 = np.load("Recons/other_dual_recon_low_noise.npy")
    dual_low_noise = np.concatenate((dual_low_noise, dual_low_noise2), axis = 0)
    del dual_low_noise2
    dual_low_noise_rrmse = np.sum((true_c - dual_low_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dual_low_noise_rrmse.npy", dual_low_noise_rrmse)
    del dual_low_noise_rrmse
    dual_low_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dual_low_noise[j,0,:,:], data_range= 0.2)
        dual_low_noise_ssim.append(plc_ssim)
    np.save("Errors/dual_low_noise_ssim.npy", np.array(dual_low_noise_ssim))
    del dual_low_noise_ssim
    del dual_low_noise
        

    inet_low_noise = np.load("Recons/type_d_inet_recon_low_noise.npy")
    inet_low_noise2 = np.load("Recons/other_inet_recon_low_noise.npy")
    inet_low_noise = np.concatenate((inet_low_noise, inet_low_noise2), axis = 0)
    del inet_low_noise2
    inet_low_noise_rrmse = np.sum((true_c - inet_low_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/inet_low_noise_rrmse.npy", inet_low_noise_rrmse)
    del inet_low_noise_rrmse
    inet_low_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], inet_low_noise[j,0,:,:], data_range= 0.2)
        inet_low_noise_ssim.append(plc_ssim)
    np.save("Errors/inet_low_noise_ssim.npy", np.array(inet_low_noise_ssim))
    del inet_low_noise_ssim
    del inet_low_noise
        


    

    fwi_medium_noise = np.load("Recons/type_d_fwi_recon_medium_noise.npy")
    fwi_medium_noise2 = np.load("Recons/other_fwi_recon_medium_noise.npy")
    fwi_medium_noise = np.concatenate((fwi_medium_noise, fwi_medium_noise2), axis = 0)
    del fwi_medium_noise2
    fwi_medium_noise_rrmse = np.sum((true_c - fwi_medium_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/fwi_medium_noise_rrmse.npy", fwi_medium_noise_rrmse)
    del fwi_medium_noise_rrmse
    fwi_medium_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], fwi_medium_noise[j,0,:,:], data_range= 0.2)
        fwi_medium_noise_ssim.append(plc_ssim)
    np.save("Errors/fwi_medium_noise_ssim.npy", np.array(fwi_medium_noise_ssim))
    del fwi_medium_noise_ssim
    del fwi_medium_noise
        


    born_medium_noise = np.load("Recons/type_d_born_recon_medium_noise.npy")
    born_medium_noise2 = np.load("Recons/other_born_recon_medium_noise.npy")
    born_medium_noise = np.concatenate((born_medium_noise, born_medium_noise2), axis = 0)
    del born_medium_noise2
    born_medium_noise_rrmse = np.sum((true_c - born_medium_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/born_medium_noise_rrmse.npy", born_medium_noise_rrmse)
    del born_medium_noise_rrmse
    born_medium_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], born_medium_noise[j,0,:,:], data_range= 0.2)
        born_medium_noise_ssim.append(plc_ssim)
    np.save("Errors/born_medium_noise_ssim.npy", np.array(born_medium_noise_ssim))
    del born_medium_noise_ssim
    del born_medium_noise
        

    art_medium_noise = np.load("Recons/type_d_ac_recon_medium_noise.npy")
    art_medium_noise2 = np.load("Recons/other_ac_recon_medium_noise.npy")
    art_medium_noise = np.concatenate((art_medium_noise, art_medium_noise2), axis = 0)
    del art_medium_noise2
    art_medium_noise_rrmse = np.sum((true_c - art_medium_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/art_medium_noise_rrmse.npy", art_medium_noise_rrmse)
    del art_medium_noise_rrmse
    art_medium_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], art_medium_noise[j,0,:,:], data_range= 0.2)
        art_medium_noise_ssim.append(plc_ssim)
    np.save("Errors/art_medium_noise_ssim.npy", np.array(art_medium_noise_ssim))
    del art_medium_noise_ssim
    del art_medium_noise
        

    dc_medium_noise = np.load("Recons/type_d_dc_recon_medium_noise.npy")
    dc_medium_noise2 = np.load("Recons/other_dc_recon_medium_noise.npy")
    dc_medium_noise = np.concatenate((dc_medium_noise, dc_medium_noise2), axis = 0)
    del dc_medium_noise2
    dc_medium_noise_rrmse = np.sum((true_c - dc_medium_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dc_medium_noise_rrmse.npy", dc_medium_noise_rrmse)
    del dc_medium_noise_rrmse
    dc_medium_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dc_medium_noise[j,0,:,:], data_range= 0.2)
        dc_medium_noise_ssim.append(plc_ssim)
    np.save("Errors/dc_medium_noise_ssim.npy", np.array(dc_medium_noise_ssim))
    del dc_medium_noise_ssim
    del dc_medium_noise
        

    dual_medium_noise = np.load("Recons/type_d_dual_recon_medium_noise.npy")
    dual_medium_noise2 = np.load("Recons/other_dual_recon_medium_noise.npy")
    dual_medium_noise = np.concatenate((dual_medium_noise, dual_medium_noise2), axis = 0)
    del dual_medium_noise2
    dual_medium_noise_rrmse = np.sum((true_c - dual_medium_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dual_medium_noise_rrmse.npy", dual_medium_noise_rrmse)
    del dual_medium_noise_rrmse
    dual_medium_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dual_medium_noise[j,0,:,:], data_range= 0.2)
        dual_medium_noise_ssim.append(plc_ssim)
    np.save("Errors/dual_medium_noise_ssim.npy", np.array(dual_medium_noise_ssim))
    del dual_medium_noise_ssim
    del dual_medium_noise
        

    inet_medium_noise = np.load("Recons/type_d_inet_recon_medium_noise.npy")
    inet_medium_noise2 = np.load("Recons/other_inet_recon_medium_noise.npy")
    inet_medium_noise = np.concatenate((inet_medium_noise, inet_medium_noise2), axis = 0)
    del inet_medium_noise2
    inet_medium_noise_rrmse = np.sum((true_c - inet_medium_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/inet_medium_noise_rrmse.npy", inet_medium_noise_rrmse)
    del inet_medium_noise_rrmse
    inet_medium_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], inet_medium_noise[j,0,:,:], data_range= 0.2)
        inet_medium_noise_ssim.append(plc_ssim)
    np.save("Errors/inet_medium_noise_ssim.npy", np.array(inet_medium_noise_ssim))
    del inet_medium_noise_ssim
    del inet_medium_noise
        


    

    fwi_high_noise = np.load("Recons/type_d_fwi_recon_high_noise.npy")
    fwi_high_noise2 = np.load("Recons/other_fwi_recon_high_noise.npy")
    fwi_high_noise = np.concatenate((fwi_high_noise, fwi_high_noise2), axis = 0)
    del fwi_high_noise2
    fwi_high_noise_rrmse = np.sum((true_c - fwi_high_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/fwi_high_noise_rrmse.npy", fwi_high_noise_rrmse)
    del fwi_high_noise_rrmse
    fwi_high_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], fwi_high_noise[j,0,:,:], data_range= 0.2)
        fwi_high_noise_ssim.append(plc_ssim)
    np.save("Errors/fwi_high_noise_ssim.npy", np.array(fwi_high_noise_ssim))
    del fwi_high_noise_ssim
    del fwi_high_noise
        


    born_high_noise = np.load("Recons/type_d_born_recon_high_noise.npy")
    born_high_noise2 = np.load("Recons/other_born_recon_high_noise.npy")
    born_high_noise = np.concatenate((born_high_noise, born_high_noise2), axis = 0)
    del born_high_noise2
    born_high_noise_rrmse = np.sum((true_c - born_high_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/born_high_noise_rrmse.npy", born_high_noise_rrmse)
    del born_high_noise_rrmse
    born_high_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], born_high_noise[j,0,:,:], data_range= 0.2)
        born_high_noise_ssim.append(plc_ssim)
    np.save("Errors/born_high_noise_ssim.npy", np.array(born_high_noise_ssim))
    del born_high_noise_ssim
    del born_high_noise
        

    art_high_noise = np.load("Recons/type_d_ac_recon_high_noise.npy")
    art_high_noise2 = np.load("Recons/other_ac_recon_high_noise.npy")
    art_high_noise = np.concatenate((art_high_noise, art_high_noise2), axis = 0)
    del art_high_noise2
    art_high_noise_rrmse = np.sum((true_c - art_high_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/art_high_noise_rrmse.npy", art_high_noise_rrmse)
    del art_high_noise_rrmse
    art_high_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], art_high_noise[j,0,:,:], data_range= 0.2)
        art_high_noise_ssim.append(plc_ssim)
    np.save("Errors/art_high_noise_ssim.npy", np.array(art_high_noise_ssim))
    del art_high_noise_ssim
    del art_high_noise
        

    dc_high_noise = np.load("Recons/type_d_dc_recon_high_noise.npy")
    dc_high_noise2 = np.load("Recons/other_dc_recon_high_noise.npy")
    dc_high_noise = np.concatenate((dc_high_noise, dc_high_noise2), axis = 0)
    del dc_high_noise2
    dc_high_noise_rrmse = np.sum((true_c - dc_high_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dc_high_noise_rrmse.npy", dc_high_noise_rrmse)
    del dc_high_noise_rrmse
    dc_high_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dc_high_noise[j,0,:,:], data_range= 0.2)
        dc_high_noise_ssim.append(plc_ssim)
    np.save("Errors/dc_high_noise_ssim.npy", np.array(dc_high_noise_ssim))
    del dc_high_noise_ssim
    del dc_high_noise
        

    dual_high_noise = np.load("Recons/type_d_dual_recon_high_noise.npy")
    dual_high_noise2 = np.load("Recons/other_dual_recon_high_noise.npy")
    dual_high_noise = np.concatenate((dual_high_noise, dual_high_noise2), axis = 0)
    del dual_high_noise2
    dual_high_noise_rrmse = np.sum((true_c - dual_high_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/dual_high_noise_rrmse.npy", dual_high_noise_rrmse)
    del dual_high_noise_rrmse
    dual_high_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], dual_high_noise[j,0,:,:], data_range= 0.2)
        dual_high_noise_ssim.append(plc_ssim)
    np.save("Errors/dual_high_noise_ssim.npy", np.array(dual_high_noise_ssim))
    del dual_high_noise_ssim
    del dual_high_noise
        

    inet_high_noise = np.load("Recons/type_d_inet_recon_high_noise.npy")
    inet_high_noise2 = np.load("Recons/other_inet_recon_high_noise.npy")
    inet_high_noise = np.concatenate((inet_high_noise, inet_high_noise2), axis = 0)
    del inet_high_noise2
    inet_high_noise_rrmse = np.sum((true_c - inet_high_noise)**2, axis = (1,2,3))**(1/2)/true_c_norm
    np.save("Errors/inet_high_noise_rrmse.npy", inet_high_noise_rrmse)
    del inet_high_noise_rrmse
    inet_high_noise_ssim = []
    for j in range(true_c.shape[0]):
        plc_ssim = ssim(true_c[j,0,:,:], inet_high_noise[j,0,:,:], data_range= 0.2)
        inet_high_noise_ssim.append(plc_ssim)
    np.save("Errors/inet_high_noise_ssim.npy", np.array(inet_high_noise_ssim))
    del inet_high_noise_ssim
    del inet_high_noise
        

