import numpy as np
import torch
import matplotlib.pyplot as plt

from net import *

if __name__ == "__main__":
    dev = torch.device('cuda:0')
    #tnet = TumorNet(32, 64, 128, 128).to(dev)
    tnet = TumorNet(8, 16, 32, 64).to(dev)
    train_ratio = 20/35


    c_np = np.load("Recons/type_d_dc_recon_medium_noise.npy")
    c_np = c_np[ :int((train_ratio*c_np.shape[0])), :,3:-3, 3:-3]
    t_np = np.load("Phantoms/type_d_tumors.npy")
    t_np = t_np[ :int((train_ratio*t_np.shape[0])), :,3:-3, 3:-3]

    c_np2 = np.load("Recons/other_dc_recon_medium_noise.npy")
    c_np2 = c_np2[ :int((train_ratio*c_np2.shape[0])), :,3:-3, 3:-3]
    c_np = np.concatenate((c_np, c_np2), axis = 0)
    del c_np2
    t_np2 = np.load("Phantoms/other_tumors.npy")
    t_np2 = t_np2[ :int((train_ratio*t_np2.shape[0])), :,3:-3, 3:-3]
    t_np = np.concatenate((t_np, t_np2), axis = 0)
    del t_np2


    nb = 1
    batch_inds = np.arange(nb)
    nepochs = 2*10**3
    opt = torch.optim.Adam(tnet.parameters(), lr = 1e-3)

    losses = []
    n_checkpoint = 100
    inds = np.arange(t_np.shape[0])
    eps = 1e-4
    for e in range(nepochs):
        np.random.shuffle(inds)
        np.random.shuffle(batch_inds)
        for b in batch_inds:
            nrots = np.random.randint(4)
            nflips0 = np.random.randint(2)
            nflips1 = np.random.randint(2)

            cs = c_np[inds[b::nb],:,:,:]
            ts = t_np[inds[b::nb],:,:,:]
            if nflips0 > 0:
                cs = np.flip(cs, axis = -1)
                ts = np.flip(ts, axis = -1)
            if nflips1 > 0:
                cs = np.flip(cs, axis = -2)
                ts = np.flip(ts, axis = -2)
            cs = np.rot90(cs, k = nrots, axes = (-1, -2))
            ts = np.rot90(ts, k = nrots, axes = (-1, -2))
            
            C = torch.from_numpy(cs.copy()).float().to(dev)
            T = torch.from_numpy(ts.copy()).float().to(dev)
            opt.zero_grad()

            Y = tnet(C)
            #loss = torch.mean((T- Y)**2)/2
            #loss = torch.mean(torch.abs(T- Y))

            loss = -torch.mean( T*torch.log(Y + eps ) + (1-T)*torch.log(1-Y + eps))


            losses.append(loss.item())
            loss.backward()
            opt.step()
            print(e, loss.item())

        if e > 0 and e%n_checkpoint == 0:
            #np.save("Losses/art_net_medium_noise_losses.npy", np.array(losses))
            plt.clf()
            plt.semilogy(np.linspace(0, e+1, len(losses)), losses)
            plt.savefig("losses.png")
            torch.save(tnet.state_dict(), "Tumor_Segmenters/dc_net_medium_noise.pth")
    torch.save(tnet.state_dict(), "Tumor_Segmenters/dc_net_medium_noise.pth")
