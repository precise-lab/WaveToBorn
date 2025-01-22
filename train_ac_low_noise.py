import numpy as np
import torch
import matplotlib.pyplot as plt

from net import *

if __name__ == "__main__":
    dev = torch.device('cuda:0')

    anet = ArtNet(32, 64, 128, 128).to(dev)
    train_ratio = 20/35
    c_np = np.load("Phantoms/type_d_phantoms.npy")
    c_np = c_np[ :int((train_ratio*c_np.shape[0])), :,3:-3, 3:-3]
    x_np = np.load("Recons/type_d_born_recon_low_noise.npy")
    x_np = x_np[ :int((train_ratio*x_np.shape[0])), :,3:-3, 3:-3]

    c_np2 = np.load("Phantoms/other_phantoms.npy")
    c_np2 = c_np2[ :int((train_ratio*c_np2.shape[0])), :,3:-3, 3:-3]
    c_np = np.concatenate((c_np, c_np2), axis = 0)
    del c_np2
    x_np2 = np.load("Recons/other_born_recon_low_noise.npy")
    x_np2 = x_np2[ :int((train_ratio*x_np2.shape[0])), :,3:-3, 3:-3]
    x_np = np.concatenate((x_np, x_np2), axis = 0)
    del x_np2

    nb = 4
    batch_inds = np.arange(nb)
    nepochs = 10**4
    opt = torch.optim.Adam(anet.parameters(), lr = 1e-4)

    losses = []
    n_checkpoint = 100
    inds = np.arange(x_np.shape[0])
    for e in range(nepochs):
        np.random.shuffle(inds)
        np.random.shuffle(batch_inds)
        for b in batch_inds:

            nrots = np.random.randint(4)
            nflips0 = np.random.randint(2)
            nflips1 = np.random.randint(2)

            cs = c_np[inds[b::nb],:,:,:]
            xs = x_np[inds[b::nb],:,:,:]

            if nflips0 > 0:
                cs = np.flip(cs, axis = -1)
                xs = np.flip(xs, axis = -1)
            if nflips1 > 0:
                cs = np.flip(cs, axis = -2)
                xs = np.flip(xs, axis = -2)
            cs = np.rot90(cs, k = nrots, axes = (-1, -2))
            xs = np.rot90(xs, k = nrots, axes = (-1, -2))

            C = torch.from_numpy(cs.copy()).to(dev)
            X = torch.from_numpy(xs.copy()).to(dev)

            opt.zero_grad()
            Y = 1.5 + 0.1*anet(X)
            loss = torch.mean((C- Y)**2)/2
            losses.append(loss.item())
            loss.backward()
            opt.step()
            print(e, loss.item())
        if e > 0 and e%n_checkpoint == 0:
            np.save("Losses/art_net_low_noise_losses.npy", np.array(losses))
            plt.clf()
            plt.semilogy(np.linspace(0, e+1, len(losses)), losses)
            plt.savefig("losses.png")
            torch.save(anet.state_dict(), "Networks/art_net_low_noise.pth")
    np.save("Losses/art_net_low_noise_losses.npy", np.array(losses))
    plt.clf()
    plt.semilogy(np.linspace(0, e+1, len(losses)), losses)
    plt.savefig("losses.png")
    torch.save(anet.state_dict(), "Networks/art_net_low_noise.pth")
  
