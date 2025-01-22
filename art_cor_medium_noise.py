import numpy as np
import torch
import matplotlib.pyplot as plt

from net import *
if __name__ == "__main__":
    dev = torch.device('cuda:0')

    anet = ArtNet(32, 64, 128, 128).to(dev)
    anet.load_state_dict(torch.load("Networks/art_net_medium_noise.pth"))
    anet.eval()

    
    x_np = np.load("Recons/type_d_born_recon_medium_noise.npy")
    y_np = 1.5*np.ones(x_np.shape)

    nb = 5
    for b in range(nb):
        X = torch.from_numpy(x_np[b::nb,:,3:-3,3:-3]).to(dev)
        Y = 1.5 + 0.1*anet(X)
        Y = torch.clamp(Y, 1.4, 1.6)
        y_np[b::nb,:,3:-3,3:-3] = Y.cpu().detach().numpy()
    np.save("Recons/type_d_ac_recon_medium_noise.npy", y_np)

    x_np = np.load("Recons/other_born_recon_medium_noise.npy")
    y_np = 1.5*np.ones(x_np.shape)

    nb = 40
    for b in range(nb):
        X = torch.from_numpy(x_np[b::nb,:,3:-3,3:-3]).to(dev)
        Y = 1.5 + 0.1*anet(X)
        Y = torch.clamp(Y, 1.4, 1.6)
        y_np[b::nb,:,3:-3,3:-3] = Y.cpu().detach().numpy()
    np.save("Recons/other_ac_recon_medium_noise.npy", y_np)

        
    x_np = np.load("Recons/type_d_born_recon_low_noise.npy")
    y_np = 1.5*np.ones(x_np.shape)

    nb = 5
    for b in range(nb):
        X = torch.from_numpy(x_np[b::nb,:,3:-3,3:-3]).to(dev)
        Y = 1.5 + 0.1*anet(X)
        Y = torch.clamp(Y, 1.4, 1.6)
        y_np[b::nb,:,3:-3,3:-3] = Y.cpu().detach().numpy()
    np.save("Recons/type_d_ac_recon_medium_low.npy", y_np)

    x_np = np.load("Recons/other_born_recon_low_noise.npy")
    y_np = 1.5*np.ones(x_np.shape)

    nb = 40
    for b in range(nb):
        X = torch.from_numpy(x_np[b::nb,:,3:-3,3:-3]).to(dev)
        Y = 1.5 + 0.1*anet(X)
        Y = torch.clamp(Y, 1.4, 1.6)
        y_np[b::nb,:,3:-3,3:-3] = Y.cpu().detach().numpy()
    np.save("Recons/other_ac_recon_medium_low.npy", y_np)


        
    x_np = np.load("Recons/type_d_born_recon_high_noise.npy")
    y_np = 1.5*np.ones(x_np.shape)

    nb = 5
    for b in range(nb):
        X = torch.from_numpy(x_np[b::nb,:,3:-3,3:-3]).to(dev)
        Y = 1.5 + 0.1*anet(X)
        Y = torch.clamp(Y, 1.4, 1.6)
        y_np[b::nb,:,3:-3,3:-3] = Y.cpu().detach().numpy()
    np.save("Recons/type_d_ac_recon_medium_high.npy", y_np)

    x_np = np.load("Recons/other_born_recon_high_noise.npy")
    y_np = 1.5*np.ones(x_np.shape)

    nb = 40
    for b in range(nb):
        X = torch.from_numpy(x_np[b::nb,:,3:-3,3:-3]).to(dev)
        Y = 1.5 + 0.1*anet(X)
        Y = torch.clamp(Y, 1.4, 1.6)
        y_np[b::nb,:,3:-3,3:-3] = Y.cpu().detach().numpy()
    np.save("Recons/other_ac_recon_medium_high.npy", y_np)
    
    