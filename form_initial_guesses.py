import numpy as np
import torch
import matplotlib.pyplot as plt

from forward import *

if __name__ == "__main__":
    dev = torch.device('cuda:0')
    sgma = 10

    nx = 214
    

    grid_mrks = torch.arange(nx+1)
    X,Y = torch.meshgrid(grid_mrks, grid_mrks)
    blur = torch.exp( -(X-(nx+1)/2)**2/(2*sgma**2) -(Y-(nx+1)/2)**2/(2*sgma**2))
    blur = blur/torch.sum(blur)

    blur = blur/torch.sum(blur)


    """blur0 = torch.zeros((nx+1, nx+1))
    blur0[nx//2,nx//2] = 1

    plt.clf()
    plt.imshow((blur-blur0).cpu().detach().numpy())
    plt.colorbar()
    plt.savefig("blur_test.png")

    

    assert False"""
    blur = blur.reshape((1,1, nx+1,nx+1)).to(dev)

    x_np = np.load("type_d_phantoms.npy")
    x = torch.from_numpy(x_np).to(dev)
    y = torch.nn.functional.conv2d(x-1.5, blur, padding = nx//2) + 1.5
    y_np = y.cpu().detach().numpy()

    print( (np.sum((x_np - y_np)**2)/np.sum((x_np-1.5)**2))**(1/2))
    plt.clf()
    plt.imshow(x_np[0,0,:,:])
    plt.colorbar()
    plt.savefig("initial_image.png")
    plt.clf()
    plt.imshow(y_np[0,0,:,:])
    plt.colorbar()
    plt.savefig("blur_image.png")



    np.save("type_d_initial_guess.npy", y_np)
    assert False

    x_np = np.load("other_phantoms.npy")
    x = torch.from_numpy(x_np).to(dev)
    y = torch.nn.functional.conv2d(x-1.5, blur, padding = nx//2) + 1.5
    y_np = y.cpu().detach().numpy()

    np.save("other_initial_guess.npy", y_np)

   