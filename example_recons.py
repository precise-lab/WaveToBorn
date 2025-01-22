import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":

    obj = np.load("other_phantoms.npy")


    fwi = np.load("other_fwi_recon_low_noise.npy")
    inet = np.load("other_inet_recon_low_noise.npy")
    born = np.load("other_born_recon_low_noise.npy")
    ac = np.load("other_ac_recon_low_noise.npy")
    dc = np.load("other_dc_recon_low_noise.npy")
    dual = np.load("other_dual_recon_low_noise.npy")

    iid = -10



    f, a = plt.subplots(1,7)
    a[0].imshow(obj[iid,0,:,:], vmin = 1.4, vmax = 1.6)
    a[1].imshow(fwi[iid,0,:,:], vmin = 1.4, vmax = 1.6)
    a[2].imshow(inet[iid,0,:,:], vmin = 1.4, vmax = 1.6)
    a[3].imshow(born[iid,0,:,:], vmin = 1.4, vmax = 1.6)
    a[4].imshow(ac[iid,0,:,:], vmin = 1.4, vmax = 1.6)
    a[5].imshow(dc[iid,0,:,:], vmin = 1.4, vmax = 1.6)
    a[6].imshow(dual[iid,0,:,:], vmin = 1.4, vmax = 1.6)

    plt.savefig("Figures/example_low_noise.png")