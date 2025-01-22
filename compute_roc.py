import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as io

from net import *

def compute_roc(threshes, t_np, mask_np, y_np):
        detection_rate = [1.]
        false_positive_rate = [1.]
        for thresh in threshes:
            texp = y_np > thresh
            detection_rate.append(np.sum(texp*t_np*mask_np)/true_den)
            #false_positive_rate.append(np.sum((texp)*(1-t_np)*mask_np)/false_den)
            if np.sum(texp*mask_np) > 0:
                false_positive_rate.append(np.sum((texp)*(1-t_np)*mask_np)/np.sum(texp*mask_np))
            else:
                false_positive_rate.append(0.)
                 
            t_np = t_np[texp]
            y_np = y_np[texp]
            mask_np = mask_np[texp]
            

        detection_rate.append(0.)
        false_positive_rate.append(0.)

        detection_rate = np.array(detection_rate)
        false_positive_rate = np.array(false_positive_rate)
        auc = np.sum(0.5*(detection_rate[:-1] + detection_rate[1:] )*(false_positive_rate[:-1] - false_positive_rate[1:]))
        return false_positive_rate, detection_rate, auc



if __name__ == "__main__":
    with torch.no_grad():
        train_ratio = 20/35
        dev = torch.device('cuda:0')
        #tnet = TumorNet(32, 64, 128, 128).to(dev)
        tnet = TumorNet(8, 16, 32, 64).to(dev)

        t_np = np.load("Phantoms/type_d_tumors.npy")
        #t_np = t_np[ int((train_ratio*t_np.shape[0])):, :,:, :]
        t_np = t_np[ int((train_ratio*t_np.shape[0])):, :,3:-3, 3:-3]
        t_np2 = np.load("Phantoms/other_tumors.npy")
        #t_np2 = t_np2[ int((train_ratio*t_np2.shape[0])):, :,:, :]
        t_np2 = t_np2[ int((train_ratio*t_np2.shape[0])):, :,3:-3, 3:-3]
        t_np = np.concatenate((t_np, t_np2), axis = 0)
        del t_np2

        x_np = np.load("Initial_Guesses/type_d_initial_guess.npy")
        mask_np = np.abs(x_np - 1.5) > 1e-5
        mask_np = mask_np[ int((train_ratio*mask_np.shape[0])):, :,3:-3, 3:-3]
        x_np = np.load("Initial_Guesses/other_initial_guess.npy")
        mask_np2 = np.abs(x_np - 1.5) > 1e-5
        mask_np2 = mask_np2[ int((train_ratio*mask_np2.shape[0])):, :,3:-3, 3:-3]
        mask_np = np.concatenate((mask_np, mask_np2), axis = 0)
        del mask_np2
        




        nthreshes0 = 10**3
        nthreshes1 = 10**3
        thresh_disc = 1e-3
        threshes0 = thresh_disc*np.linspace(0,1,nthreshes0)
        threshes1 = np.linspace(thresh_disc,1,nthreshes1)[1:]
        threshes = np.append(threshes0, threshes1)



        true_den = np.sum(t_np*mask_np)
        false_den = np.sum((1-t_np)*mask_np)
        N = true_den + false_den

        net_names = ["dc_net_low_noise", "ac_net_low_noise", "dual_net_low_noise", "inet_net_low_noise", \
                     "dc_net_medium_noise", "ac_net_medium_noise", "dual_net_medium_noise", "inet_net_medium_noise", \
                     "dc_net_high_noise", "ac_net_high_noise", "dual_net_high_noise", "inet_net_high_noise"]
        recon_names0 = ["dc_recon_low_noise", "ac_recon_low_noise", "dual_recon_low_noise", "inet_recon_low_noise", \
                     "dc_recon_medium_noise", "ac_recon_medium_noise", "dual_recon_medium_noise", "inet_recon_medium_noise", \
                     "dc_recon_high_noise", "ac_recon_high_noise", "dual_recon_high_noise", "inet_recon_high_noise"]
        recon_names1 = ["dc_recon_low_medium", "ac_recon_low_medium", "dual_recon_low_medium", "inet_recon_low_medium", \
                     "dc_recon_medium_low", "ac_recon_medium_low", "dual_recon_medium_low", "inet_recon_medium_low", \
                     "dc_recon_high_low", "ac_recon_high_low", "dual_recon_high_low", "inet_recon_high_low"]
        recon_names2 = ["dc_recon_low_high", "ac_recon_low_high", "dual_recon_low_high", "inet_recon_low_high", \
                     "dc_recon_medium_high", "ac_recon_medium_high", "dual_recon_medium_high", "inet_recon_medium_high", \
                     "dc_recon_high_medium", "ac_recon_high_medium", "dual_recon_high_medium", "inet_recon_high_medium"]
        

        aucs = {}
        for net_name, recon_name in zip(net_names, recon_names0):
            print(recon_name)
            c_np = np.load("Recons/type_d_" + recon_name + ".npy")
            c_np = c_np[ int((train_ratio*c_np.shape[0])):, :,3:-3, 3:-3]
            c_np2 = np.load("Recons/other_" + recon_name + ".npy")
            c_np2 = c_np2[ int((train_ratio*c_np2.shape[0])):, :,3:-3, 3:-3]
            c_np = np.concatenate((c_np, c_np2), axis = 0)
            del c_np2

            tnet.load_state_dict(torch.load("Tumor_Segmenters/" + net_name +".pth"))
            tnet.eval()
            y_np = tnet( torch.from_numpy(c_np).float().to(dev)).cpu().detach().numpy()
            false_positive_rate, detection_rate, auc = compute_roc(threshes, t_np, mask_np, y_np)
            aucs[recon_name] = auc
            
        for net_name, recon_name in zip(net_names, recon_names1):
            print(recon_name)
            c_np = np.load("Recons/type_d_" + recon_name + ".npy")
            c_np = c_np[ int((train_ratio*c_np.shape[0])):, :,3:-3, 3:-3]
            c_np2 = np.load("Recons/other_" + recon_name + ".npy")
            c_np2 = c_np2[ int((train_ratio*c_np2.shape[0])):, :,3:-3, 3:-3]
            c_np = np.concatenate((c_np, c_np2), axis = 0)
            del c_np2

            tnet.load_state_dict(torch.load("Tumor_Segmenters/" + net_name +".pth"))
            tnet.eval()
            y_np = tnet( torch.from_numpy(c_np).float().to(dev)).cpu().detach().numpy()
            false_positive_rate, detection_rate, auc = compute_roc(threshes, t_np, mask_np, y_np)
            aucs[recon_name] = auc

        for net_name, recon_name in zip(net_names, recon_names2):
            print(recon_name)
            c_np = np.load("Recons/type_d_" + recon_name + ".npy")
            c_np = c_np[ int((train_ratio*c_np.shape[0])):, :,3:-3, 3:-3]
            c_np2 = np.load("Recons/other_" + recon_name + ".npy")
            c_np2 = c_np2[ int((train_ratio*c_np2.shape[0])):, :,3:-3, 3:-3]
            c_np = np.concatenate((c_np, c_np2), axis = 0)
            del c_np2

            tnet.load_state_dict(torch.load("Tumor_Segmenters/" + net_name +".pth"))
            tnet.eval()
            y_np = tnet( torch.from_numpy(c_np).float().to(dev)).cpu().detach().numpy()
            false_positive_rate, detection_rate, auc = compute_roc(threshes, t_np, mask_np, y_np)
            aucs[recon_name] = auc
        io.savemat('aucs.mat', aucs)




        
