import numpy as np
import torch
import matplotlib.pyplot as plt

from forward import *
from net import *

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

    
    ni = 300

    F = SEBornFoward(nx, Nx, dx, nt, dt, nm, sr, sR, fc, sgma, t0, dev = dev)
    bnet = BornNet(32, 64, 128, 128).to(dev)
    bnet.load_state_dict(torch.load("Networks/data_net_medium_noise.pth"))
    bnet.eval()

    """std = 6e-5
    xt_np = np.load("Phantoms/type_d_phantoms.npy")
    xtnorm = np.sum((xt_np - 1.5)**2)**(1/2)
    p_np = np.load("Measurements/type_d_wave_offset.npy")
    p_np += std*np.random.standard_normal(p_np.shape)
    x_np = np.load("Initial_Guesses/type_d_initial_guess.npy")
    mask_np = np.abs(x_np - 1.5) > 1e-5

    nb = 5
    for b in range(nb):
        with torch.no_grad():
            p = torch.from_numpy(p_np[b::nb,:,:,:]).to(dev)
            for c in range(p.shape[1]):
                p[:,c:c+1,:,-496:] =  p[:,c:c+1,:,-496:] +  1e-3*torch.roll(bnet( torch.roll(p[:,c:c+1,:,-496:], -4*(c-32), 2)), 4*(c-32), 2)
                p[:,c:c+1,:,:-496] = 0
        x = torch.from_numpy(x_np[b::nb,:,:,:]).to(dev)
        mask = torch.from_numpy(mask_np[b::nb,:,:,:]).to(dev)
        x.requires_grad = True
        opt = torch.optim.Adam([x], lr = 1e-3)
        for i in range(ni):
            print(b,i/ni)

            opt.zero_grad()
            se = torch.randn(nm//sr, device = dev)
            se = se/torch.norm(se)
            xc = torch.clamp( mask*(x-1.5)+ 1.5, 1.4,1.6)
            with torch.no_grad():
                pe = torch.unsqueeze(torch.einsum('ijkm, j->ikm', p, se), 1)
            pexp = F(xc, se)
            loss = torch.sum((pe-pexp)**2)/2
            loss.backward()
            opt.step()
            
        xc = torch.clamp( mask*(x-1.5)+ 1.5, 1.4,1.6)
        x_np[b::nb,:,:,:] = xc.cpu().detach().numpy()
        np.save("Recons/type_d_dc_recon_medium_noise.npy", x_np)

    nb *= 8
    xt_np = np.load("Phantoms/other_phantoms.npy")
    xtnorm = np.sum((xt_np - 1.5)**2)**(1/2)
    p_np = np.load("Measurements/other_wave_offset.npy")
    p_np += std*np.random.standard_normal(p_np.shape)
    x_np = np.load("Initial_Guesses/other_initial_guess.npy")
    mask_np = np.abs(x_np - 1.5) > 1e-5

    for b in range(nb):
        with torch.no_grad():
            p = torch.from_numpy(p_np[b::nb,:,:,:]).to(dev)
            for c in range(p.shape[1]):
                p[:,c:c+1,:,-496:] =  p[:,c:c+1,:,-496:] +  1e-3*torch.roll(bnet( torch.roll(p[:,c:c+1,:,-496:], -4*(c-32), 2)), 4*(c-32), 2)
                p[:,c:c+1,:,:-496] = 0
        x = torch.from_numpy(x_np[b::nb,:,:,:]).to(dev)
        mask = torch.from_numpy(mask_np[b::nb,:,:,:]).to(dev)
        x.requires_grad = True
        opt = torch.optim.Adam([x], lr = 1e-3)
        for i in range(ni):
            print(b,i/ni)

            opt.zero_grad()
            se = torch.randn(nm//sr, device = dev)
            se = se/torch.norm(se)
            xc = torch.clamp( mask*(x-1.5)+ 1.5, 1.4,1.6)
            with torch.no_grad():
                pe = torch.unsqueeze(torch.einsum('ijkm, j->ikm', p, se), 1)
            pexp = F(xc, se)
            loss = torch.sum((pe-pexp)**2)/2
            loss.backward()
            opt.step()
        
        xc = torch.clamp( mask*(x-1.5)+ 1.5, 1.4,1.6)
        x_np[b::nb,:,:,:] = xc.cpu().detach().numpy()
        np.save("Recons/other_dc_recon_medium_noise.npy", x_np)"""


    std = 3e-5
    xt_np = np.load("Phantoms/type_d_phantoms.npy")
    xtnorm = np.sum((xt_np - 1.5)**2)**(1/2)
    p_np = np.load("Measurements/type_d_wave_offset.npy")
    p_np += std*np.random.standard_normal(p_np.shape)
    x_np = np.load("Initial_Guesses/type_d_initial_guess.npy")
    mask_np = np.abs(x_np - 1.5) > 1e-5

    nb = 5
    for b in range(nb):
        with torch.no_grad():
            p = torch.from_numpy(p_np[b::nb,:,:,:]).to(dev)
            for c in range(p.shape[1]):
                p[:,c:c+1,:,-496:] =  p[:,c:c+1,:,-496:] +  1e-3*torch.roll(bnet( torch.roll(p[:,c:c+1,:,-496:], -4*(c-32), 2)), 4*(c-32), 2)
                p[:,c:c+1,:,:-496] = 0
        x = torch.from_numpy(x_np[b::nb,:,:,:]).to(dev)
        mask = torch.from_numpy(mask_np[b::nb,:,:,:]).to(dev)
        x.requires_grad = True
        opt = torch.optim.Adam([x], lr = 1e-3)
        for i in range(ni):
            print(b,i/ni)

            opt.zero_grad()
            se = torch.randn(nm//sr, device = dev)
            se = se/torch.norm(se)
            xc = torch.clamp( mask*(x-1.5)+ 1.5, 1.4,1.6)
            with torch.no_grad():
                pe = torch.unsqueeze(torch.einsum('ijkm, j->ikm', p, se), 1)
            pexp = F(xc, se)
            loss = torch.sum((pe-pexp)**2)/2
            loss.backward()
            opt.step()
            
        xc = torch.clamp( mask*(x-1.5)+ 1.5, 1.4,1.6)
        x_np[b::nb,:,:,:] = xc.cpu().detach().numpy()
        np.save("Recons/type_d_dc_recon_medium_low.npy", x_np)

    nb *= 8
    xt_np = np.load("Phantoms/other_phantoms.npy")
    xtnorm = np.sum((xt_np - 1.5)**2)**(1/2)
    p_np = np.load("Measurements/other_wave_offset.npy")
    p_np += std*np.random.standard_normal(p_np.shape)
    x_np = np.load("Initial_Guesses/other_initial_guess.npy")
    mask_np = np.abs(x_np - 1.5) > 1e-5

    for b in range(nb):
        with torch.no_grad():
            p = torch.from_numpy(p_np[b::nb,:,:,:]).to(dev)
            for c in range(p.shape[1]):
                p[:,c:c+1,:,-496:] =  p[:,c:c+1,:,-496:] +  1e-3*torch.roll(bnet( torch.roll(p[:,c:c+1,:,-496:], -4*(c-32), 2)), 4*(c-32), 2)
                p[:,c:c+1,:,:-496] = 0
        x = torch.from_numpy(x_np[b::nb,:,:,:]).to(dev)
        mask = torch.from_numpy(mask_np[b::nb,:,:,:]).to(dev)
        x.requires_grad = True
        opt = torch.optim.Adam([x], lr = 1e-3)
        for i in range(ni):
            print(b,i/ni)

            opt.zero_grad()
            se = torch.randn(nm//sr, device = dev)
            se = se/torch.norm(se)
            xc = torch.clamp( mask*(x-1.5)+ 1.5, 1.4,1.6)
            with torch.no_grad():
                pe = torch.unsqueeze(torch.einsum('ijkm, j->ikm', p, se), 1)
            pexp = F(xc, se)
            loss = torch.sum((pe-pexp)**2)/2
            loss.backward()
            opt.step()
        
        xc = torch.clamp( mask*(x-1.5)+ 1.5, 1.4,1.6)
        x_np[b::nb,:,:,:] = xc.cpu().detach().numpy()
        np.save("Recons/other_dc_recon_medium_low.npy", x_np)



    std = 15e-5
    xt_np = np.load("Phantoms/type_d_phantoms.npy")
    xtnorm = np.sum((xt_np - 1.5)**2)**(1/2)
    p_np = np.load("Measurements/type_d_wave_offset.npy")
    p_np += std*np.random.standard_normal(p_np.shape)
    x_np = np.load("Initial_Guesses/type_d_initial_guess.npy")
    mask_np = np.abs(x_np - 1.5) > 1e-5

    nb = 5
    for b in range(nb):
        with torch.no_grad():
            p = torch.from_numpy(p_np[b::nb,:,:,:]).to(dev)
            for c in range(p.shape[1]):
                p[:,c:c+1,:,-496:] =  p[:,c:c+1,:,-496:] +  1e-3*torch.roll(bnet( torch.roll(p[:,c:c+1,:,-496:], -4*(c-32), 2)), 4*(c-32), 2)
                p[:,c:c+1,:,:-496] = 0
        x = torch.from_numpy(x_np[b::nb,:,:,:]).to(dev)
        mask = torch.from_numpy(mask_np[b::nb,:,:,:]).to(dev)
        x.requires_grad = True
        opt = torch.optim.Adam([x], lr = 1e-3)
        for i in range(ni):
            print(b,i/ni)

            opt.zero_grad()
            se = torch.randn(nm//sr, device = dev)
            se = se/torch.norm(se)
            xc = torch.clamp( mask*(x-1.5)+ 1.5, 1.4,1.6)
            with torch.no_grad():
                pe = torch.unsqueeze(torch.einsum('ijkm, j->ikm', p, se), 1)
            pexp = F(xc, se)
            loss = torch.sum((pe-pexp)**2)/2
            loss.backward()
            opt.step()
            
        xc = torch.clamp( mask*(x-1.5)+ 1.5, 1.4,1.6)
        x_np[b::nb,:,:,:] = xc.cpu().detach().numpy()
        np.save("Recons/type_d_dc_recon_medium_high.npy", x_np)

    nb *= 8
    xt_np = np.load("Phantoms/other_phantoms.npy")
    xtnorm = np.sum((xt_np - 1.5)**2)**(1/2)
    p_np = np.load("Measurements/other_wave_offset.npy")
    p_np += std*np.random.standard_normal(p_np.shape)
    x_np = np.load("Initial_Guesses/other_initial_guess.npy")
    mask_np = np.abs(x_np - 1.5) > 1e-5

    for b in range(nb):
        with torch.no_grad():
            p = torch.from_numpy(p_np[b::nb,:,:,:]).to(dev)
            for c in range(p.shape[1]):
                p[:,c:c+1,:,-496:] =  p[:,c:c+1,:,-496:] +  1e-3*torch.roll(bnet( torch.roll(p[:,c:c+1,:,-496:], -4*(c-32), 2)), 4*(c-32), 2)
                p[:,c:c+1,:,:-496] = 0
        x = torch.from_numpy(x_np[b::nb,:,:,:]).to(dev)
        mask = torch.from_numpy(mask_np[b::nb,:,:,:]).to(dev)
        x.requires_grad = True
        opt = torch.optim.Adam([x], lr = 1e-3)
        for i in range(ni):
            print(b,i/ni)

            opt.zero_grad()
            se = torch.randn(nm//sr, device = dev)
            se = se/torch.norm(se)
            xc = torch.clamp( mask*(x-1.5)+ 1.5, 1.4,1.6)
            with torch.no_grad():
                pe = torch.unsqueeze(torch.einsum('ijkm, j->ikm', p, se), 1)
            pexp = F(xc, se)
            loss = torch.sum((pe-pexp)**2)/2
            loss.backward()
            opt.step()
        
        xc = torch.clamp( mask*(x-1.5)+ 1.5, 1.4,1.6)
        x_np[b::nb,:,:,:] = xc.cpu().detach().numpy()
        np.save("Recons/other_dc_recon_medium_high.npy", x_np)