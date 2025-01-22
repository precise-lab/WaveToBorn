import numpy as np
import torch
import matplotlib.pyplot as plt

from inversionnet import *
if __name__ == "__main__":
    with torch.no_grad():
        dev = torch.device('cuda:0')
        inet = InversionNet().to(dev)
        inet.load_state_dict(torch.load("Networks/inet_low_noise.pth"))
        inet.eval()

        std = 3e-5
        nb = 5
        p_np = np.load("Measurements/type_d_wave_offset.npy")
        p_np += std*np.random.standard_normal(p_np.shape)
        x_np = 0*np.load("Initial_Guesses/type_d_initial_guess.npy")+ 1.5

        for b in range(nb):
            p = torch.from_numpy(p_np[b::nb,:,:,-496:]).to(dev)
            x_np[b::nb, :, 3:-3, 3:-3] = inet(p).cpu().detach().numpy()
            np.save("Recons/type_d_inet_recon_low_noise.npy", x_np)
        nb *= 8
        p_np = np.load("Measurements/other_wave_offset.npy")
        p_np += std*np.random.standard_normal(p_np.shape)
        x_np = 0*np.load("Initial_Guesses/other_initial_guess.npy")+ 1.5
        
        for b in range(nb):
            p = torch.from_numpy(p_np[b::nb,:,:,-496:]).to(dev)
            x_np[b::nb,:, 3:-3, 3:-3] = inet(p).cpu().detach().numpy()
            np.save("Recons/other_inet_recon_low_noise.npy", x_np)


        std = 6e-5
        nb = 5
        p_np = np.load("Measurements/type_d_wave_offset.npy")
        p_np += std*np.random.standard_normal(p_np.shape)
        x_np = 0*np.load("Initial_Guesses/type_d_initial_guess.npy")+ 1.5

        for b in range(nb):
            p = torch.from_numpy(p_np[b::nb,:,:,-496:]).to(dev)
            x_np[b::nb, :, 3:-3, 3:-3] = inet(p).cpu().detach().numpy()
            np.save("Recons/type_d_inet_recon_low_medium.npy", x_np)
        nb *= 8
        p_np = np.load("Measurements/other_wave_offset.npy")
        p_np += std*np.random.standard_normal(p_np.shape)
        x_np = 0*np.load("Initial_Guesses/other_initial_guess.npy")+ 1.5
        
        for b in range(nb):
            p = torch.from_numpy(p_np[b::nb,:,:,-496:]).to(dev)
            x_np[b::nb,:, 3:-3, 3:-3] = inet(p).cpu().detach().numpy()
            np.save("Recons/other_inet_recon_low_medium.npy", x_np)


        std = 15e-5
        nb = 5
        p_np = np.load("Measurements/type_d_wave_offset.npy")
        p_np += std*np.random.standard_normal(p_np.shape)
        x_np = 0*np.load("Initial_Guesses/type_d_initial_guess.npy")+ 1.5

        for b in range(nb):
            p = torch.from_numpy(p_np[b::nb,:,:,-496:]).to(dev)
            x_np[b::nb, :, 3:-3, 3:-3] = inet(p).cpu().detach().numpy()
            np.save("Recons/type_d_inet_recon_low_high.npy", x_np)
        nb *= 8
        p_np = np.load("Measurements/other_wave_offset.npy")
        p_np += std*np.random.standard_normal(p_np.shape)
        x_np = 0*np.load("Initial_Guesses/other_initial_guess.npy")+ 1.5
        
        for b in range(nb):
            p = torch.from_numpy(p_np[b::nb,:,:,-496:]).to(dev)
            x_np[b::nb,:, 3:-3, 3:-3] = inet(p).cpu().detach().numpy()
            np.save("Recons/other_inet_recon_low_high.npy", x_np)
        






        
        






        
        






        