import numpy as np
import torch
import matplotlib.pyplot as plt

from forward import *

if __name__ == "__main__":
    with torch.no_grad():
        dev = torch.device('cuda:0')
        nx = 214
        Nx = 360
        dx = 0.6
        dt = 0.2
        fc = 0.5
        sgma = 2
        t0 = 3.2
        nt = 800
        sR = 160

        sr = 4
        nm = 256

        F = BornFoward(nx, Nx, dx, nt, dt, nm, sr, sR, fc, sgma, t0, dev=dev)
        
        x_np = np.load("type_d_phantoms.npy")
        x = torch.from_numpy(x_np).to(dev)

        nb = 10

        p = np.zeros((x_np.shape[0], nm//sr, nm, nt))
        for b in range(nb):
            x = torch.from_numpy(x_np[b::nb, :,:,:]).to(dev)
            p[b::nb, :,:,:] = F(x).cpu().detach().numpy()
        np.save("type_d_born_offset.npy", p.astype(np.float32))
        del x


        nb = 50

        


        
        x_np = np.load("other_phantoms.npy")
        p = np.zeros((x_np.shape[0], nm//sr, nm, nt))

        for b in range(nb):
            x = torch.from_numpy(x_np[b::nb, :,:,:]).to(dev)
            p[b::nb, :,:,:] = F(x).cpu().detach().numpy()
        np.save("other_born_offset.npy", p.astype(np.float32))



