import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from os import walk

if __name__ == "__main__":
    phantom_folder = "../2D_acoustic_phantoms/"
    filenames = next(walk(phantom_folder), (None, None, []))[2]  # [] if no file
    count = 0
    sos_end = "_sos.mat"
    sos_files = []
    for f in filenames:
        if f[-len(sos_end):] == sos_end:
            sos_files.append(f)
    atten_files = [""]*len(sos_files)
    atten_end = "_atten.mat"
    for f in filenames:
        if f[-len(atten_end):] == atten_end:
            for j,g in enumerate(sos_files):
                if f[:-len(atten_end)] == g[:-len(sos_end)]:
                    atten_files[j] = f
    tst = io.loadmat(phantom_folder + atten_files[0])


    tumr_thresh = 0.15
    bck_thresh = 0.01

    ams = []
    for af in atten_files:
        aa = io.loadmat(phantom_folder + af)["aa"]
        ams.append( aa[(aa > bck_thresh)*(aa < tumr_thresh)].mean())
    ams = np.array(ams)
    typed_thresh = 0.0629

    print(aa.shape)
    ssf = 3
    nx = 214
    wx = (aa.shape[0] - ssf*nx)//2

    ntd = np.sum(ams > typed_thresh )
    nto = len(sos_files) - ntd

    typeo = np.zeros((nto, 1, nx,nx))
    typed = np.zeros((ntd, 1, nx,nx))
    tumoo = np.zeros((nto, 1, nx,nx))
    tumod = np.zeros((ntd, 1, nx, nx))

    ocount = 0
    dcount = 0

    for j, am in enumerate(ams):
        c = io.loadmat(phantom_folder + sos_files[j])["sos"]
        aa = io.loadmat(phantom_folder + atten_files[j])["aa"]
        if am > typed_thresh:
            typed[dcount,0,:,:] = c[wx:-wx:ssf, wx:-wx:ssf]
            tumod[dcount,0,:,:] = aa[wx:-wx:ssf, wx:-wx:ssf] > tumr_thresh
            dcount += 1
        else:
            typeo[ocount,0,:,:] = c[wx:-wx:ssf, wx:-wx:ssf]
            tumoo[ocount,0,:,:] = aa[wx:-wx:ssf, wx:-wx:ssf] > tumr_thresh
            ocount += 1
    np.save("type_d_phantoms.npy", typed.astype(np.float32))
    np.save("type_d_tumors.npy", tumod.astype(int))

    np.save("other_phantoms.npy", typeo.astype(np.float32))
    np.save("other_tumors.npy", tumoo.astype(int))
    

    




    


