import numpy as np
import torch
import matplotlib.pyplot as plt

from net import *
if __name__ == "__main__":
    dev = torch.device('cuda:0')

    anet = ArtNet(32, 64, 128, 128).to(dev)
    """s = 0 
    for p in anet.parameters():
        s += p.numel()
    print(s)
    assert False"""
    anet.load_state_dict(torch.load("Networks/dual_net_high_noise.pth"))
    anet.eval()

    
    x_np = np.load("Recons/type_d_dc_recon_high_noise.npy")
    y_np = 1.5*np.ones(x_np.shape)

    nb = 5
    for b in range(nb):
        X = torch.from_numpy(x_np[b::nb,:,3:-3,3:-3]).to(dev)
        #Y = X - 0.2*anet(X)
        Y = 1.5 + 0.1*anet(X)
        Y = torch.clamp(Y, 1.4,1.6)
        y_np[b::nb,:,3:-3,3:-3] = Y.cpu().detach().numpy()
    np.save("Recons/type_d_dual_recon_high_noise.npy", y_np)

    x_np = np.load("Recons/other_dc_recon_high_noise.npy")
    y_np = 1.5*np.ones(x_np.shape)

    nb = 40
    for b in range(nb):
        X = torch.from_numpy(x_np[b::nb,:,3:-3,3:-3]).to(dev)
        #Y = X - 0.2*anet(X)
        Y = 1.5 + 0.1*anet(X)
        Y = torch.clamp(Y, 1.4,1.6)
        y_np[b::nb,:,3:-3,3:-3] = Y.cpu().detach().numpy()
    np.save("Recons/other_dual_recon_high_noise.npy", y_np)

    x_np = np.load("Recons/type_d_dc_recon_high_low.npy")
    y_np = 1.5*np.ones(x_np.shape)

    nb = 5
    for b in range(nb):
        X = torch.from_numpy(x_np[b::nb,:,3:-3,3:-3]).to(dev)
        #Y = X - 0.2*anet(X)
        Y = 1.5 + 0.1*anet(X)
        Y = torch.clamp(Y, 1.4,1.6)
        y_np[b::nb,:,3:-3,3:-3] = Y.cpu().detach().numpy()
    np.save("Recons/type_d_dual_recon_high_low.npy", y_np)

    x_np = np.load("Recons/other_dc_recon_high_low.npy")
    y_np = 1.5*np.ones(x_np.shape)

    nb = 40
    for b in range(nb):
        X = torch.from_numpy(x_np[b::nb,:,3:-3,3:-3]).to(dev)
        #Y = X - 0.2*anet(X)
        Y = 1.5 + 0.1*anet(X)
        Y = torch.clamp(Y, 1.4,1.6)
        y_np[b::nb,:,3:-3,3:-3] = Y.cpu().detach().numpy()
    np.save("Recons/other_dual_recon_high_low.npy", y_np)


    x_np = np.load("Recons/type_d_dc_recon_high_medium.npy")
    y_np = 1.5*np.ones(x_np.shape)

    nb = 5
    for b in range(nb):
        X = torch.from_numpy(x_np[b::nb,:,3:-3,3:-3]).to(dev)
        #Y = X - 0.2*anet(X)
        Y = 1.5 + 0.1*anet(X)
        Y = torch.clamp(Y, 1.4,1.6)
        y_np[b::nb,:,3:-3,3:-3] = Y.cpu().detach().numpy()
    np.save("Recons/type_d_dual_recon_high_medium.npy", y_np)

    x_np = np.load("Recons/other_dc_recon_high_medium.npy")
    y_np = 1.5*np.ones(x_np.shape)

    nb = 40
    for b in range(nb):
        X = torch.from_numpy(x_np[b::nb,:,3:-3,3:-3]).to(dev)
        #Y = X - 0.2*anet(X)
        Y = 1.5 + 0.1*anet(X)
        Y = torch.clamp(Y, 1.4,1.6)
        y_np[b::nb,:,3:-3,3:-3] = Y.cpu().detach().numpy()
    np.save("Recons/other_dual_recon_high_medium.npy", y_np)
    
    
    
    
    
    