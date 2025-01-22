import numpy as np
import torch
import matplotlib.pyplot as plt

from forward import *

if __name__ == "__main__":
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

    std = 6e-5
    ni = 500

    F = SEWaveFoward(nx, Nx, dx, nt, dt, nm, sr, sR, fc, sgma, t0)
    p0 = np.load("isotropic_wave.npy")

    nb = 5
    xt_np = np.load("type_d_phantoms.npy")
    xtnorm = np.sum((xt_np - 1.5)**2)**(1/2)
    p_np = np.load("type_d_wave_offset.npy") + p0
    p_np += std*np.random.standard_normal(p_np.shape)
    x_np = np.load("type_d_initial_guess.npy")

    

    for b in range(nb):
        p = torch.from_numpy(p_np[b::nb,:,:,:]).to(dev)
        x = torch.from_numpy(x_np[b::nb,:,:,:]).to(dev)
        x.requires_grad = True
        opt = torch.optim.Adam([x], lr = 1e-3)
        for i in range(ni):
            print(b,i/ni)

            opt.zero_grad()
            se = torch.randn(nm//sr, device = dev)
            se = se/np.sqrt(nm//sr)#torch.norm(se)
            with torch.no_grad():
                pe = torch.unsqueeze(torch.einsum('ijkm, j->ikm', p, se), 1)
            pexp = F(x, se)
            loss = torch.sum((pe-pexp)**2)/2
            loss.backward()
            opt.step()
        x_np[b::nb,:,:,:] = x.cpu().detach().numpy()
        np.save("type_d_fwi_recon_medium_noise.npy", x_np)

    nb *= 8
    xt_np = np.load("other_phantoms.npy")
    xtnorm = np.sum((xt_np - 1.5)**2)**(1/2)
    p_np = np.load("other_wave_offset.npy") + p0
    p_np += std*np.random.standard_normal(p_np.shape)
    x_np = np.load("other_initial_guess.npy")

    

    for b in range(nb):
        p = torch.from_numpy(p_np[b::nb,:,:,:]).to(dev)
        x = torch.from_numpy(x_np[b::nb,:,:,:]).to(dev)
        x.requires_grad = True
        opt = torch.optim.Adam([x], lr = 1e-3)
        for i in range(ni):
            print(b,i/ni)

            opt.zero_grad()
            se = torch.randn(nm//sr, device = dev)
            se = se/np.sqrt(nm//sr)#torch.norm(se)
            with torch.no_grad():
                pe = torch.unsqueeze(torch.einsum('ijkm, j->ikm', p, se), 1)
            pexp = F(x, se)
            loss = torch.sum((pe-pexp)**2)/2
            loss.backward()
            opt.step()
        x_np[b::nb,:,:,:] = x.cpu().detach().numpy()
        np.save("other_fwi_recon_medium_noise.npy", x_np)
        
    

        
    
