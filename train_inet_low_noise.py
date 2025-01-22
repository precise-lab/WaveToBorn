import numpy as np
import torch
import matplotlib.pyplot as plt

from inversionnet import *

if __name__ == "__main__":
    dev = torch.device('cuda:0')

    inet = InversionNet().to(dev)
    train_ratio = 20/35
    p_np = np.load("type_d_wave_offset.npy")
    p_np = p_np[ :int((train_ratio*p_np.shape[0])), :,:, -496:]

    p_np2 = np.load("other_wave_offset.npy")
    p_np2 = p_np2[ :int((train_ratio*p_np2.shape[0])), :,:, -496:]
    p_np = np.concatenate((p_np, p_np2), axis = 0)
    del p_np2
    c_np = np.load("type_d_phantoms.npy")
    c_np = c_np[ :int((train_ratio*c_np.shape[0])), :,3:-3, 3:-3]
    c_np2 = np.load("other_phantoms.npy")
    c_np2 = c_np2[ :int((train_ratio*c_np2.shape[0])), :,3:-3, 3:-3]
    c_np = np.concatenate((c_np, c_np2), axis = 0)
    del c_np2

    nb = 10
    
    batch_inds = np.arange(nb)

    nepochs = 10**3
    opt = torch.optim.Adam(inet.parameters(), lr = 1e-4)
    std = 3e-5
    losses = []
    for e in range(nepochs):
        np.random.shuffle(batch_inds)
        for b in batch_inds:
            opt.zero_grad()
            P = torch.from_numpy(p_np[b::nb,:,:,:]).to(dev)
            P += std*torch.randn(P.size(), device = dev)
            C = torch.from_numpy(c_np[b::nb,:,:,:]).to(dev)

            E = inet(P)
            loss = torch.mean((C-E)**2)/2
            loss.backward()
            losses.append(loss.item())
            opt.step()
            print(e, loss.item())
        np.save("inet_low_noise_losses.npy", np.array(losses))
        plt.clf()
        plt.semilogy(np.linspace(0, e+1, len(losses)), losses)
        plt.savefig("losses.png")
        torch.save(inet.state_dict(), "inet_low_noise.pth")
    



    