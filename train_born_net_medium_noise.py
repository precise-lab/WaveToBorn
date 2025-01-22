import numpy as np
import torch
import matplotlib.pyplot as plt

from net import *

if __name__ == "__main__":
    dev = torch.device('cuda:0')

    bnet = BornNet(32, 64, 128, 128).to(dev)

    train_ratio = 20/35
    p_np = np.load("type_d_wave_offset.npy")
    p_np = p_np[ :int((train_ratio*p_np.shape[0])), :,:, -496:]
    b_np = np.load("type_d_born_offset.npy")
    b_np = b_np[ :int((train_ratio*b_np.shape[0])), :,:, -496:]

    #p_np = p_np[ :1, :,:, :]
    #b_np = b_np[ :1, :,:, :]

    p_np2 = np.load("other_wave_offset.npy")
    p_np2 = p_np2[ :int((train_ratio*p_np2.shape[0])), :,:, -496:]
    p_np = np.concatenate((p_np, p_np2), axis = 0)
    del p_np2
    b_np2 = np.load("other_born_offset.npy")
    b_np2 = b_np2[ :int((train_ratio*b_np2.shape[0])), :,:, -496:]
    b_np = np.concatenate((b_np, b_np2), axis = 0)
    del b_np2

    for c in range(b_np.shape[1]):
        p_np[:,c:c+1,:,:] = np.roll(p_np[:,c:c+1,:,:], -4*(c-32), 2)
        b_np[:,c:c+1,:,:] = np.roll(b_np[:,c:c+1,:,:], -4*(c-32), 2)
    p_np = p_np.reshape(p_np.shape[0]*p_np.shape[1], 1, p_np.shape[2], p_np.shape[3])
    b_np = b_np.reshape(b_np.shape[0]*b_np.shape[1], 1, b_np.shape[2], b_np.shape[3])

    nb = 600
    batch_inds = np.arange(nb)
    channel_inds = np.arange(b_np.shape[1])
    nepochs = 200

    opt = torch.optim.Adam(bnet.parameters(), lr = 1e-3)
    std = 6e-5

    losses = []

    torch.save(bnet.state_dict(), "data_net_medium_noise.pth")

    nd = 256
    nt = 296

    inds = np.arange(b_np.shape[0])
    for e in range(nepochs):
        np.random.shuffle(inds)
        for c in channel_inds:
            np.random.shuffle(batch_inds)
            for b in batch_inds:
                P = torch.from_numpy(p_np[inds[b::nb],:,:,:] + std*np.random.standard_normal(p_np[b::nb,:,:,:].shape)).float().to(dev)
                B = torch.from_numpy(b_np[inds[b::nb],:,:,:]).to(dev)
                Bn = 1e6*torch.mean(B**2)/2
                opt.zero_grad()
                out = P + 1e-3*bnet(P)
                loss =  1e6*torch.mean((out-B)**2)/2
                losses.append(loss.item())
                loss.backward()
                opt.step()
                print(e, loss.item(), (loss).item()**(1/2), (loss.item()/Bn.item())**(1/2))
        np.save("data_net_medium_noise_losses.npy", np.array(losses))
        plt.clf()
        plt.semilogy(np.linspace(0, e+1, len(losses)), losses)
        plt.savefig("losses.png")
        torch.save(bnet.state_dict(), "data_net_medium_noise.pth")


    