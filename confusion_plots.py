import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as io

import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

if __name__ == "__main__":
    nd = np.load("Phantoms/type_d_phantoms.npy").shape[0]
    no = np.load("Phantoms/other_phantoms.npy").shape[0]

    train_ratio = 20/35

    vmax_rrmse = 0.8
    vmin_rrmse = 0.36

    art_low_noise_rrmse = np.load("Errors/art_low_noise_rrmse.npy")
    art_low_noise_rrmse_train = art_low_noise_rrmse[:int(train_ratio*nd)]
    art_low_noise_rrmse_train = np.concatenate((art_low_noise_rrmse_train, art_low_noise_rrmse[nd:nd + int(train_ratio*no)]))
    art_low_noise_rrmse_test = art_low_noise_rrmse[int(train_ratio*nd):nd]
    art_low_noise_rrmse_test = np.concatenate((art_low_noise_rrmse_test, art_low_noise_rrmse[nd + int(train_ratio*no):]))

    art_low_medium_rrmse = np.load("Errors/art_low_medium_rrmse.npy")
    art_low_medium_rrmse_train = art_low_medium_rrmse[:int(train_ratio*nd)]
    art_low_medium_rrmse_train = np.concatenate((art_low_medium_rrmse_train, art_low_medium_rrmse[nd:nd + int(train_ratio*no)]))
    art_low_medium_rrmse_test = art_low_medium_rrmse[int(train_ratio*nd):nd]
    art_low_medium_rrmse_test = np.concatenate((art_low_medium_rrmse_test, art_low_medium_rrmse[nd + int(train_ratio*no):]))

    art_low_high_rrmse = np.load("Errors/art_low_high_rrmse.npy")
    art_low_high_rrmse_train = art_low_high_rrmse[:int(train_ratio*nd)]
    art_low_high_rrmse_train = np.concatenate((art_low_high_rrmse_train, art_low_high_rrmse[nd:nd + int(train_ratio*no)]))
    art_low_high_rrmse_test = art_low_high_rrmse[int(train_ratio*nd):nd]
    art_low_high_rrmse_test = np.concatenate((art_low_high_rrmse_test, art_low_high_rrmse[nd + int(train_ratio*no):]))

    art_medium_noise_rrmse = np.load("Errors/art_medium_noise_rrmse.npy")
    art_medium_noise_rrmse_train = art_medium_noise_rrmse[:int(train_ratio*nd)]
    art_medium_noise_rrmse_train = np.concatenate((art_medium_noise_rrmse_train, art_medium_noise_rrmse[nd:nd + int(train_ratio*no)]))
    art_medium_noise_rrmse_test = art_medium_noise_rrmse[int(train_ratio*nd):nd]
    art_medium_noise_rrmse_test = np.concatenate((art_medium_noise_rrmse_test, art_medium_noise_rrmse[nd + int(train_ratio*no):]))

    art_medium_low_rrmse = np.load("Errors/art_medium_low_rrmse.npy")
    art_medium_low_rrmse_train = art_medium_low_rrmse[:int(train_ratio*nd)]
    art_medium_low_rrmse_train = np.concatenate((art_medium_low_rrmse_train, art_medium_low_rrmse[nd:nd + int(train_ratio*no)]))
    art_medium_low_rrmse_test = art_medium_low_rrmse[int(train_ratio*nd):nd]
    art_medium_low_rrmse_test = np.concatenate((art_medium_low_rrmse_test, art_medium_low_rrmse[nd + int(train_ratio*no):]))

    art_medium_high_rrmse = np.load("Errors/art_medium_high_rrmse.npy")
    art_medium_high_rrmse_train = art_medium_high_rrmse[:int(train_ratio*nd)]
    art_medium_high_rrmse_train = np.concatenate((art_medium_high_rrmse_train, art_medium_high_rrmse[nd:nd + int(train_ratio*no)]))
    art_medium_high_rrmse_test = art_medium_high_rrmse[int(train_ratio*nd):nd]
    art_medium_high_rrmse_test = np.concatenate((art_medium_high_rrmse_test, art_medium_high_rrmse[nd + int(train_ratio*no):]))

    art_high_noise_rrmse = np.load("Errors/art_high_noise_rrmse.npy")
    art_high_noise_rrmse_train = art_high_noise_rrmse[:int(train_ratio*nd)]
    art_high_noise_rrmse_train = np.concatenate((art_high_noise_rrmse_train, art_high_noise_rrmse[nd:nd + int(train_ratio*no)]))
    art_high_noise_rrmse_test = art_high_noise_rrmse[int(train_ratio*nd):nd]
    art_high_noise_rrmse_test = np.concatenate((art_high_noise_rrmse_test, art_high_noise_rrmse[nd + int(train_ratio*no):]))

    art_high_low_rrmse = np.load("Errors/art_high_low_rrmse.npy")
    art_high_low_rrmse_train = art_high_low_rrmse[:int(train_ratio*nd)]
    art_high_low_rrmse_train = np.concatenate((art_high_low_rrmse_train, art_high_low_rrmse[nd:nd + int(train_ratio*no)]))
    art_high_low_rrmse_test = art_high_low_rrmse[int(train_ratio*nd):nd]
    art_high_low_rrmse_test = np.concatenate((art_high_low_rrmse_test, art_high_low_rrmse[nd + int(train_ratio*no):]))

    art_high_medium_rrmse = np.load("Errors/art_high_medium_rrmse.npy")
    art_high_medium_rrmse_train = art_high_medium_rrmse[:int(train_ratio*nd)]
    art_high_medium_rrmse_train = np.concatenate((art_high_medium_rrmse_train, art_high_medium_rrmse[nd:nd + int(train_ratio*no)]))
    art_high_medium_rrmse_test = art_high_medium_rrmse[int(train_ratio*nd):nd]
    art_high_medium_rrmse_test = np.concatenate((art_high_medium_rrmse_test, art_high_medium_rrmse[nd + int(train_ratio*no):]))
    
    art_low_noise_rrmse_ci = mean_confidence_interval(art_low_noise_rrmse_test)
    art_low_medium_rrmse_ci = mean_confidence_interval(art_low_medium_rrmse_test)
    art_low_high_rrmse_ci = mean_confidence_interval(art_low_high_rrmse_test)
    art_medium_noise_rrmse_ci = mean_confidence_interval(art_medium_noise_rrmse_test)
    art_medium_low_rrmse_ci = mean_confidence_interval(art_medium_low_rrmse_test)
    art_medium_high_rrmse_ci = mean_confidence_interval(art_medium_high_rrmse_test)
    art_high_noise_rrmse_ci = mean_confidence_interval(art_high_noise_rrmse_test)
    art_high_low_rrmse_ci = mean_confidence_interval(art_high_low_rrmse_test)
    art_high_medium_rrmse_ci = mean_confidence_interval(art_high_medium_rrmse_test)
    
    art_map = np.array([[art_low_noise_rrmse_ci[0], art_low_medium_rrmse_ci[0], art_low_high_rrmse_ci[0]],
               [art_medium_low_rrmse_ci[0], art_medium_noise_rrmse_ci[0], art_medium_high_rrmse_ci[0]],
               [art_high_low_rrmse_ci[0], art_high_medium_rrmse_ci[0], art_high_noise_rrmse_ci[0]]])
    art_map = np.transpose(art_map)

    art_h = np.array([[art_low_noise_rrmse_ci[1], art_low_medium_rrmse_ci[1], art_low_high_rrmse_ci[1]],
               [art_medium_low_rrmse_ci[1], art_medium_noise_rrmse_ci[1], art_medium_high_rrmse_ci[1]],
               [art_high_low_rrmse_ci[1], art_high_medium_rrmse_ci[1], art_high_noise_rrmse_ci[1]]])
    art_h = np.transpose(art_h)
    
    noise_labels = ["Low", "Medium", "High"]
    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(art_map, vmin = vmin_rrmse, vmax = vmax_rrmse, cmap = 'Oranges')

    art_map = np.round(100*art_map)/100
    art_h = np.round(100*art_h)/100
    ax.set_xticks(range(3), noise_labels)
    ax.set_yticks(range(3), noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax.text(i, j, str(art_map[j, i]) + r"$\pm$" + str(art_h[j, i]),
                       ha="center", va="center", color="black")
    ax.set_ylabel("Testing Noise", fontsize = 18)
    ax.set_xlabel("Training Noise", fontsize = 18)
    ax.set_title("Artifact Correction Confusion",  fontsize = 24)
    plt.savefig("Figures/art_rrmse_confusion.png", bbox_inches='tight')


    dc_low_noise_rrmse = np.load("Errors/dc_low_noise_rrmse.npy")
    dc_low_noise_rrmse_train = dc_low_noise_rrmse[:int(train_ratio*nd)]
    dc_low_noise_rrmse_train = np.concatenate((dc_low_noise_rrmse_train, dc_low_noise_rrmse[nd:nd + int(train_ratio*no)]))
    dc_low_noise_rrmse_test = dc_low_noise_rrmse[int(train_ratio*nd):nd]
    dc_low_noise_rrmse_test = np.concatenate((dc_low_noise_rrmse_test, dc_low_noise_rrmse[nd + int(train_ratio*no):]))

    dc_low_medium_rrmse = np.load("Errors/dc_low_medium_rrmse.npy")
    dc_low_medium_rrmse_train = dc_low_medium_rrmse[:int(train_ratio*nd)]
    dc_low_medium_rrmse_train = np.concatenate((dc_low_medium_rrmse_train, dc_low_medium_rrmse[nd:nd + int(train_ratio*no)]))
    dc_low_medium_rrmse_test = dc_low_medium_rrmse[int(train_ratio*nd):nd]
    dc_low_medium_rrmse_test = np.concatenate((dc_low_medium_rrmse_test, dc_low_medium_rrmse[nd + int(train_ratio*no):]))

    dc_low_high_rrmse = np.load("Errors/dc_low_high_rrmse.npy")
    dc_low_high_rrmse_train = dc_low_high_rrmse[:int(train_ratio*nd)]
    dc_low_high_rrmse_train = np.concatenate((dc_low_high_rrmse_train, dc_low_high_rrmse[nd:nd + int(train_ratio*no)]))
    dc_low_high_rrmse_test = dc_low_high_rrmse[int(train_ratio*nd):nd]
    dc_low_high_rrmse_test = np.concatenate((dc_low_high_rrmse_test, dc_low_high_rrmse[nd + int(train_ratio*no):]))

    dc_medium_noise_rrmse = np.load("Errors/dc_medium_noise_rrmse.npy")
    dc_medium_noise_rrmse_train = dc_medium_noise_rrmse[:int(train_ratio*nd)]
    dc_medium_noise_rrmse_train = np.concatenate((dc_medium_noise_rrmse_train, dc_medium_noise_rrmse[nd:nd + int(train_ratio*no)]))
    dc_medium_noise_rrmse_test = dc_medium_noise_rrmse[int(train_ratio*nd):nd]
    dc_medium_noise_rrmse_test = np.concatenate((dc_medium_noise_rrmse_test, dc_medium_noise_rrmse[nd + int(train_ratio*no):]))

    dc_medium_low_rrmse = np.load("Errors/dc_medium_low_rrmse.npy")
    dc_medium_low_rrmse_train = dc_medium_low_rrmse[:int(train_ratio*nd)]
    dc_medium_low_rrmse_train = np.concatenate((dc_medium_low_rrmse_train, dc_medium_low_rrmse[nd:nd + int(train_ratio*no)]))
    dc_medium_low_rrmse_test = dc_medium_low_rrmse[int(train_ratio*nd):nd]
    dc_medium_low_rrmse_test = np.concatenate((dc_medium_low_rrmse_test, dc_medium_low_rrmse[nd + int(train_ratio*no):]))

    dc_medium_high_rrmse = np.load("Errors/dc_medium_high_rrmse.npy")
    dc_medium_high_rrmse_train = dc_medium_high_rrmse[:int(train_ratio*nd)]
    dc_medium_high_rrmse_train = np.concatenate((dc_medium_high_rrmse_train, dc_medium_high_rrmse[nd:nd + int(train_ratio*no)]))
    dc_medium_high_rrmse_test = dc_medium_high_rrmse[int(train_ratio*nd):nd]
    dc_medium_high_rrmse_test = np.concatenate((dc_medium_high_rrmse_test, dc_medium_high_rrmse[nd + int(train_ratio*no):]))

    dc_high_noise_rrmse = np.load("Errors/dc_high_noise_rrmse.npy")
    dc_high_noise_rrmse_train = dc_high_noise_rrmse[:int(train_ratio*nd)]
    dc_high_noise_rrmse_train = np.concatenate((dc_high_noise_rrmse_train, dc_high_noise_rrmse[nd:nd + int(train_ratio*no)]))
    dc_high_noise_rrmse_test = dc_high_noise_rrmse[int(train_ratio*nd):nd]
    dc_high_noise_rrmse_test = np.concatenate((dc_high_noise_rrmse_test, dc_high_noise_rrmse[nd + int(train_ratio*no):]))

    dc_high_low_rrmse = np.load("Errors/dc_high_low_rrmse.npy")
    dc_high_low_rrmse_train = dc_high_low_rrmse[:int(train_ratio*nd)]
    dc_high_low_rrmse_train = np.concatenate((dc_high_low_rrmse_train, dc_high_low_rrmse[nd:nd + int(train_ratio*no)]))
    dc_high_low_rrmse_test = dc_high_low_rrmse[int(train_ratio*nd):nd]
    dc_high_low_rrmse_test = np.concatenate((dc_high_low_rrmse_test, dc_high_low_rrmse[nd + int(train_ratio*no):]))

    dc_high_medium_rrmse = np.load("Errors/dc_high_medium_rrmse.npy")
    dc_high_medium_rrmse_train = dc_high_medium_rrmse[:int(train_ratio*nd)]
    dc_high_medium_rrmse_train = np.concatenate((dc_high_medium_rrmse_train, dc_high_medium_rrmse[nd:nd + int(train_ratio*no)]))
    dc_high_medium_rrmse_test = dc_high_medium_rrmse[int(train_ratio*nd):nd]
    dc_high_medium_rrmse_test = np.concatenate((dc_high_medium_rrmse_test, dc_high_medium_rrmse[nd + int(train_ratio*no):]))
    
    dc_low_noise_rrmse_ci = mean_confidence_interval(dc_low_noise_rrmse_test)
    dc_low_medium_rrmse_ci = mean_confidence_interval(dc_low_medium_rrmse_test)
    dc_low_high_rrmse_ci = mean_confidence_interval(dc_low_high_rrmse_test)
    dc_medium_noise_rrmse_ci = mean_confidence_interval(dc_medium_noise_rrmse_test)
    dc_medium_low_rrmse_ci = mean_confidence_interval(dc_medium_low_rrmse_test)
    dc_medium_high_rrmse_ci = mean_confidence_interval(dc_medium_high_rrmse_test)
    dc_high_noise_rrmse_ci = mean_confidence_interval(dc_high_noise_rrmse_test)
    dc_high_low_rrmse_ci = mean_confidence_interval(dc_high_low_rrmse_test)
    dc_high_medium_rrmse_ci = mean_confidence_interval(dc_high_medium_rrmse_test)
    
    dc_map = np.array([[dc_low_noise_rrmse_ci[0], dc_low_medium_rrmse_ci[0], dc_low_high_rrmse_ci[0]],
               [dc_medium_low_rrmse_ci[0], dc_medium_noise_rrmse_ci[0], dc_medium_high_rrmse_ci[0]],
               [dc_high_low_rrmse_ci[0], dc_high_medium_rrmse_ci[0], dc_high_noise_rrmse_ci[0]]])
    dc_map = np.transpose(dc_map)

    dc_h = np.array([[dc_low_noise_rrmse_ci[1], dc_low_medium_rrmse_ci[1], dc_low_high_rrmse_ci[1]],
               [dc_medium_low_rrmse_ci[1], dc_medium_noise_rrmse_ci[1], dc_medium_high_rrmse_ci[1]],
               [dc_high_low_rrmse_ci[1], dc_high_medium_rrmse_ci[1], dc_high_noise_rrmse_ci[1]]])
    dc_h = np.transpose(dc_h)
    
    
    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(dc_map, vmin = vmin_rrmse, vmax = vmax_rrmse, cmap = 'Oranges')

    dc_map = np.round(100*dc_map)/100
    dc_h = np.round(100*dc_h)/100
    ax.set_xticks(range(3), noise_labels)
    ax.set_yticks(range(3), noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax.text(i, j, str(dc_map[j, i]) + r"$\pm$" + str(dc_h[j, i]),
                       ha="center", va="center", color="black")
    ax.set_ylabel("Testing Noise", fontsize = 18)
    ax.set_xlabel("Training Noise", fontsize = 18)
    ax.set_title("Artifact Correction Confusion",  fontsize = 24)
    plt.savefig("Figures/dc_rrmse_confusion.png", bbox_inches='tight')

    dual_low_noise_rrmse = np.load("Errors/dual_low_noise_rrmse.npy")
    dual_low_noise_rrmse_train = dual_low_noise_rrmse[:int(train_ratio*nd)]
    dual_low_noise_rrmse_train = np.concatenate((dual_low_noise_rrmse_train, dual_low_noise_rrmse[nd:nd + int(train_ratio*no)]))
    dual_low_noise_rrmse_test = dual_low_noise_rrmse[int(train_ratio*nd):nd]
    dual_low_noise_rrmse_test = np.concatenate((dual_low_noise_rrmse_test, dual_low_noise_rrmse[nd + int(train_ratio*no):]))

    dual_low_medium_rrmse = np.load("Errors/dual_low_medium_rrmse.npy")
    dual_low_medium_rrmse_train = dual_low_medium_rrmse[:int(train_ratio*nd)]
    dual_low_medium_rrmse_train = np.concatenate((dual_low_medium_rrmse_train, dual_low_medium_rrmse[nd:nd + int(train_ratio*no)]))
    dual_low_medium_rrmse_test = dual_low_medium_rrmse[int(train_ratio*nd):nd]
    dual_low_medium_rrmse_test = np.concatenate((dual_low_medium_rrmse_test, dual_low_medium_rrmse[nd + int(train_ratio*no):]))

    dual_low_high_rrmse = np.load("Errors/dual_low_high_rrmse.npy")
    dual_low_high_rrmse_train = dual_low_high_rrmse[:int(train_ratio*nd)]
    dual_low_high_rrmse_train = np.concatenate((dual_low_high_rrmse_train, dual_low_high_rrmse[nd:nd + int(train_ratio*no)]))
    dual_low_high_rrmse_test = dual_low_high_rrmse[int(train_ratio*nd):nd]
    dual_low_high_rrmse_test = np.concatenate((dual_low_high_rrmse_test, dual_low_high_rrmse[nd + int(train_ratio*no):]))

    dual_medium_noise_rrmse = np.load("Errors/dual_medium_noise_rrmse.npy")
    dual_medium_noise_rrmse_train = dual_medium_noise_rrmse[:int(train_ratio*nd)]
    dual_medium_noise_rrmse_train = np.concatenate((dual_medium_noise_rrmse_train, dual_medium_noise_rrmse[nd:nd + int(train_ratio*no)]))
    dual_medium_noise_rrmse_test = dual_medium_noise_rrmse[int(train_ratio*nd):nd]
    dual_medium_noise_rrmse_test = np.concatenate((dual_medium_noise_rrmse_test, dual_medium_noise_rrmse[nd + int(train_ratio*no):]))

    dual_medium_low_rrmse = np.load("Errors/dual_medium_low_rrmse.npy")
    dual_medium_low_rrmse_train = dual_medium_low_rrmse[:int(train_ratio*nd)]
    dual_medium_low_rrmse_train = np.concatenate((dual_medium_low_rrmse_train, dual_medium_low_rrmse[nd:nd + int(train_ratio*no)]))
    dual_medium_low_rrmse_test = dual_medium_low_rrmse[int(train_ratio*nd):nd]
    dual_medium_low_rrmse_test = np.concatenate((dual_medium_low_rrmse_test, dual_medium_low_rrmse[nd + int(train_ratio*no):]))

    dual_medium_high_rrmse = np.load("Errors/dual_medium_high_rrmse.npy")
    dual_medium_high_rrmse_train = dual_medium_high_rrmse[:int(train_ratio*nd)]
    dual_medium_high_rrmse_train = np.concatenate((dual_medium_high_rrmse_train, dual_medium_high_rrmse[nd:nd + int(train_ratio*no)]))
    dual_medium_high_rrmse_test = dual_medium_high_rrmse[int(train_ratio*nd):nd]
    dual_medium_high_rrmse_test = np.concatenate((dual_medium_high_rrmse_test, dual_medium_high_rrmse[nd + int(train_ratio*no):]))

    dual_high_noise_rrmse = np.load("Errors/dual_high_noise_rrmse.npy")
    dual_high_noise_rrmse_train = dual_high_noise_rrmse[:int(train_ratio*nd)]
    dual_high_noise_rrmse_train = np.concatenate((dual_high_noise_rrmse_train, dual_high_noise_rrmse[nd:nd + int(train_ratio*no)]))
    dual_high_noise_rrmse_test = dual_high_noise_rrmse[int(train_ratio*nd):nd]
    dual_high_noise_rrmse_test = np.concatenate((dual_high_noise_rrmse_test, dual_high_noise_rrmse[nd + int(train_ratio*no):]))

    dual_high_low_rrmse = np.load("Errors/dual_high_low_rrmse.npy")
    dual_high_low_rrmse_train = dual_high_low_rrmse[:int(train_ratio*nd)]
    dual_high_low_rrmse_train = np.concatenate((dual_high_low_rrmse_train, dual_high_low_rrmse[nd:nd + int(train_ratio*no)]))
    dual_high_low_rrmse_test = dual_high_low_rrmse[int(train_ratio*nd):nd]
    dual_high_low_rrmse_test = np.concatenate((dual_high_low_rrmse_test, dual_high_low_rrmse[nd + int(train_ratio*no):]))

    dual_high_medium_rrmse = np.load("Errors/dual_high_medium_rrmse.npy")
    dual_high_medium_rrmse_train = dual_high_medium_rrmse[:int(train_ratio*nd)]
    dual_high_medium_rrmse_train = np.concatenate((dual_high_medium_rrmse_train, dual_high_medium_rrmse[nd:nd + int(train_ratio*no)]))
    dual_high_medium_rrmse_test = dual_high_medium_rrmse[int(train_ratio*nd):nd]
    dual_high_medium_rrmse_test = np.concatenate((dual_high_medium_rrmse_test, dual_high_medium_rrmse[nd + int(train_ratio*no):]))
    
    dual_low_noise_rrmse_ci = mean_confidence_interval(dual_low_noise_rrmse_test)
    dual_low_medium_rrmse_ci = mean_confidence_interval(dual_low_medium_rrmse_test)
    dual_low_high_rrmse_ci = mean_confidence_interval(dual_low_high_rrmse_test)
    dual_medium_noise_rrmse_ci = mean_confidence_interval(dual_medium_noise_rrmse_test)
    dual_medium_low_rrmse_ci = mean_confidence_interval(dual_medium_low_rrmse_test)
    dual_medium_high_rrmse_ci = mean_confidence_interval(dual_medium_high_rrmse_test)
    dual_high_noise_rrmse_ci = mean_confidence_interval(dual_high_noise_rrmse_test)
    dual_high_low_rrmse_ci = mean_confidence_interval(dual_high_low_rrmse_test)
    dual_high_medium_rrmse_ci = mean_confidence_interval(dual_high_medium_rrmse_test)
    
    dual_map = np.array([[dual_low_noise_rrmse_ci[0], dual_low_medium_rrmse_ci[0], dual_low_high_rrmse_ci[0]],
               [dual_medium_low_rrmse_ci[0], dual_medium_noise_rrmse_ci[0], dual_medium_high_rrmse_ci[0]],
               [dual_high_low_rrmse_ci[0], dual_high_medium_rrmse_ci[0], dual_high_noise_rrmse_ci[0]]])
    dual_map = np.transpose(dual_map)

    dual_h = np.array([[dual_low_noise_rrmse_ci[1], dual_low_medium_rrmse_ci[1], dual_low_high_rrmse_ci[1]],
               [dual_medium_low_rrmse_ci[1], dual_medium_noise_rrmse_ci[1], dual_medium_high_rrmse_ci[1]],
               [dual_high_low_rrmse_ci[1], dual_high_medium_rrmse_ci[1], dual_high_noise_rrmse_ci[1]]])
    dual_h = np.transpose(dual_h)
    
    
    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(dual_map, vmin = vmin_rrmse, vmax = vmax_rrmse, cmap = 'Oranges')

    dual_map = np.round(100*dual_map)/100
    dual_h = np.round(100*dual_h)/100
    ax.set_xticks(range(3), noise_labels)
    ax.set_yticks(range(3), noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax.text(i, j, str(dual_map[j, i]) + r"$\pm$" + str(dual_h[j, i]),
                       ha="center", va="center", color="black")
    ax.set_ylabel("Testing Noise", fontsize = 18)
    ax.set_xlabel("Training Noise", fontsize = 18)
    ax.set_title("Artifact Correction Confusion",  fontsize = 24)
    plt.savefig("Figures/dual_rrmse_confusion.png", bbox_inches='tight')

    inet_low_noise_rrmse = np.load("Errors/inet_low_noise_rrmse.npy")
    inet_low_noise_rrmse_train = inet_low_noise_rrmse[:int(train_ratio*nd)]
    inet_low_noise_rrmse_train = np.concatenate((inet_low_noise_rrmse_train, inet_low_noise_rrmse[nd:nd + int(train_ratio*no)]))
    inet_low_noise_rrmse_test = inet_low_noise_rrmse[int(train_ratio*nd):nd]
    inet_low_noise_rrmse_test = np.concatenate((inet_low_noise_rrmse_test, inet_low_noise_rrmse[nd + int(train_ratio*no):]))

    inet_low_medium_rrmse = np.load("Errors/inet_low_medium_rrmse.npy")
    inet_low_medium_rrmse_train = inet_low_medium_rrmse[:int(train_ratio*nd)]
    inet_low_medium_rrmse_train = np.concatenate((inet_low_medium_rrmse_train, inet_low_medium_rrmse[nd:nd + int(train_ratio*no)]))
    inet_low_medium_rrmse_test = inet_low_medium_rrmse[int(train_ratio*nd):nd]
    inet_low_medium_rrmse_test = np.concatenate((inet_low_medium_rrmse_test, inet_low_medium_rrmse[nd + int(train_ratio*no):]))

    inet_low_high_rrmse = np.load("Errors/inet_low_high_rrmse.npy")
    inet_low_high_rrmse_train = inet_low_high_rrmse[:int(train_ratio*nd)]
    inet_low_high_rrmse_train = np.concatenate((inet_low_high_rrmse_train, inet_low_high_rrmse[nd:nd + int(train_ratio*no)]))
    inet_low_high_rrmse_test = inet_low_high_rrmse[int(train_ratio*nd):nd]
    inet_low_high_rrmse_test = np.concatenate((inet_low_high_rrmse_test, inet_low_high_rrmse[nd + int(train_ratio*no):]))

    inet_medium_noise_rrmse = np.load("Errors/inet_medium_noise_rrmse.npy")
    inet_medium_noise_rrmse_train = inet_medium_noise_rrmse[:int(train_ratio*nd)]
    inet_medium_noise_rrmse_train = np.concatenate((inet_medium_noise_rrmse_train, inet_medium_noise_rrmse[nd:nd + int(train_ratio*no)]))
    inet_medium_noise_rrmse_test = inet_medium_noise_rrmse[int(train_ratio*nd):nd]
    inet_medium_noise_rrmse_test = np.concatenate((inet_medium_noise_rrmse_test, inet_medium_noise_rrmse[nd + int(train_ratio*no):]))

    inet_medium_low_rrmse = np.load("Errors/inet_medium_low_rrmse.npy")
    inet_medium_low_rrmse_train = inet_medium_low_rrmse[:int(train_ratio*nd)]
    inet_medium_low_rrmse_train = np.concatenate((inet_medium_low_rrmse_train, inet_medium_low_rrmse[nd:nd + int(train_ratio*no)]))
    inet_medium_low_rrmse_test = inet_medium_low_rrmse[int(train_ratio*nd):nd]
    inet_medium_low_rrmse_test = np.concatenate((inet_medium_low_rrmse_test, inet_medium_low_rrmse[nd + int(train_ratio*no):]))

    inet_medium_high_rrmse = np.load("Errors/inet_medium_high_rrmse.npy")
    inet_medium_high_rrmse_train = inet_medium_high_rrmse[:int(train_ratio*nd)]
    inet_medium_high_rrmse_train = np.concatenate((inet_medium_high_rrmse_train, inet_medium_high_rrmse[nd:nd + int(train_ratio*no)]))
    inet_medium_high_rrmse_test = inet_medium_high_rrmse[int(train_ratio*nd):nd]
    inet_medium_high_rrmse_test = np.concatenate((inet_medium_high_rrmse_test, inet_medium_high_rrmse[nd + int(train_ratio*no):]))

    inet_high_noise_rrmse = np.load("Errors/inet_high_noise_rrmse.npy")
    inet_high_noise_rrmse_train = inet_high_noise_rrmse[:int(train_ratio*nd)]
    inet_high_noise_rrmse_train = np.concatenate((inet_high_noise_rrmse_train, inet_high_noise_rrmse[nd:nd + int(train_ratio*no)]))
    inet_high_noise_rrmse_test = inet_high_noise_rrmse[int(train_ratio*nd):nd]
    inet_high_noise_rrmse_test = np.concatenate((inet_high_noise_rrmse_test, inet_high_noise_rrmse[nd + int(train_ratio*no):]))

    inet_high_low_rrmse = np.load("Errors/inet_high_low_rrmse.npy")
    inet_high_low_rrmse_train = inet_high_low_rrmse[:int(train_ratio*nd)]
    inet_high_low_rrmse_train = np.concatenate((inet_high_low_rrmse_train, inet_high_low_rrmse[nd:nd + int(train_ratio*no)]))
    inet_high_low_rrmse_test = inet_high_low_rrmse[int(train_ratio*nd):nd]
    inet_high_low_rrmse_test = np.concatenate((inet_high_low_rrmse_test, inet_high_low_rrmse[nd + int(train_ratio*no):]))

    inet_high_medium_rrmse = np.load("Errors/inet_high_medium_rrmse.npy")
    inet_high_medium_rrmse_train = inet_high_medium_rrmse[:int(train_ratio*nd)]
    inet_high_medium_rrmse_train = np.concatenate((inet_high_medium_rrmse_train, inet_high_medium_rrmse[nd:nd + int(train_ratio*no)]))
    inet_high_medium_rrmse_test = inet_high_medium_rrmse[int(train_ratio*nd):nd]
    inet_high_medium_rrmse_test = np.concatenate((inet_high_medium_rrmse_test, inet_high_medium_rrmse[nd + int(train_ratio*no):]))
    
    inet_low_noise_rrmse_ci = mean_confidence_interval(inet_low_noise_rrmse_test)
    inet_low_medium_rrmse_ci = mean_confidence_interval(inet_low_medium_rrmse_test)
    inet_low_high_rrmse_ci = mean_confidence_interval(inet_low_high_rrmse_test)
    inet_medium_noise_rrmse_ci = mean_confidence_interval(inet_medium_noise_rrmse_test)
    inet_medium_low_rrmse_ci = mean_confidence_interval(inet_medium_low_rrmse_test)
    inet_medium_high_rrmse_ci = mean_confidence_interval(inet_medium_high_rrmse_test)
    inet_high_noise_rrmse_ci = mean_confidence_interval(inet_high_noise_rrmse_test)
    inet_high_low_rrmse_ci = mean_confidence_interval(inet_high_low_rrmse_test)
    inet_high_medium_rrmse_ci = mean_confidence_interval(inet_high_medium_rrmse_test)
    
    inet_map = np.array([[inet_low_noise_rrmse_ci[0], inet_low_medium_rrmse_ci[0], inet_low_high_rrmse_ci[0]],
               [inet_medium_low_rrmse_ci[0], inet_medium_noise_rrmse_ci[0], inet_medium_high_rrmse_ci[0]],
               [inet_high_low_rrmse_ci[0], inet_high_medium_rrmse_ci[0], inet_high_noise_rrmse_ci[0]]])
    inet_map = np.transpose(inet_map)

    inet_h = np.array([[inet_low_noise_rrmse_ci[1], inet_low_medium_rrmse_ci[1], inet_low_high_rrmse_ci[1]],
               [inet_medium_low_rrmse_ci[1], inet_medium_noise_rrmse_ci[1], inet_medium_high_rrmse_ci[1]],
               [inet_high_low_rrmse_ci[1], inet_high_medium_rrmse_ci[1], inet_high_noise_rrmse_ci[1]]])
    inet_h = np.transpose(inet_h)
    
    
    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(inet_map, vmin = vmin_rrmse, vmax = vmax_rrmse, cmap = 'Oranges')

    inet_map = np.round(100*inet_map)/100
    inet_h = np.round(100*inet_h)/100
    ax.set_xticks(range(3), noise_labels)
    ax.set_yticks(range(3), noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax.text(i, j, str(inet_map[j, i]) + r"$\pm$" + str(inet_h[j, i]),
                       ha="center", va="center", color="black")
    ax.set_ylabel("Testing Noise", fontsize = 18)
    ax.set_xlabel("Training Noise", fontsize = 18)
    ax.set_title("Artifact Correction Confusion",  fontsize = 24)
    plt.savefig("Figures/inet_rrmse_confusion.png", bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots(1,4)
    fig.set_figheight(4)
    fig.set_figwidth(16)
    ax[0].imshow(inet_map, vmin = vmin_rrmse, vmax = vmax_rrmse, cmap = 'Oranges')
    ax[0].set_xticks(range(3), noise_labels)
    ax[0].set_yticks(range(3), noise_labels, rotation = 90)
    for i in range(3):
        for j in range(3):
            text = ax[0].text(i, j, str(inet_map[j, i]) + r"$\pm$" + str(inet_h[j, i]),
                       ha="center", va="center", color="white")
    ax[0].set_ylabel("Testing Noise", fontsize = 18)
    #ax[0].set_xlabel("Training Noise", fontsize = 18)
    ax[0].set_title("InversionNet",  fontsize = 20)


    ax[1].imshow(art_map, vmin = vmin_rrmse, vmax = vmax_rrmse, cmap = 'Oranges')
    ax[1].set_xticks(range(3), noise_labels)
    ax[1].set_yticks(range(3), [""]*3)#noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax[1].text(i, j, str(art_map[j, i]) + r"$\pm$" + str(art_h[j, i]),
                       ha="center", va="center", color="black")
    #ax[1].set_ylabel("Testing Noise", fontsize = 18)
    #ax[1].set_xlabel("Training Noise", fontsize = 18)
    ax[1].set_title("Artifact Correction",  fontsize = 20)


    ax[2].imshow(dc_map, vmin = vmin_rrmse, vmax = vmax_rrmse, cmap = 'Oranges')
    ax[2].set_xticks(range(3), noise_labels)
    ax[2].set_yticks(range(3), [""]*3)#noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax[2].text(i, j, str(dc_map[j, i]) + r"$\pm$" + str(dc_h[j, i]),
                       ha="center", va="center", color="black")
    #ax.set_ylabel("Testing Noise", fontsize = 18)
    #ax[2].set_xlabel("Training Noise", fontsize = 18)
    ax[2].set_title("Data Correction",  fontsize = 20)

    im = ax[3].imshow(dual_map, vmin = vmin_rrmse, vmax = vmax_rrmse, cmap = 'Oranges')
    ax[3].set_xticks(range(3), noise_labels)
    ax[3].set_yticks(range(3), [""]*3)#noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax[3].text(i, j, str(dual_map[j, i]) + r"$\pm$" + str(dual_h[j, i]),
                       ha="center", va="center", color="black")
    #ax[3].set_ylabel("Testing Noise", fontsize = 18)
    #ax[3].set_xlabel("Training Noise", fontsize = 18)
    ax[3].set_title("Dual Correction",  fontsize = 20)
    fig.subplots_adjust(right=0.84)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)



    #fig.x("Training Noise", fontsize = 18)
    #fig.text(0.5, 0.03, 'Training Noise', ha='center', fontsize = 18)
    fig.text(0.5, 0.03, ' ', ha='center', fontsize = 18)
    fig.text(0.08, 0.43, 'RRMSE', ha='center', fontsize = 18, rotation = 90)
    #fig.title("RRMSE Confusion")
    plt.subplots_adjust(wspace=0.05)
    plt.savefig("Figures/combined_rrmse_confusion.png", bbox_inches='tight')


    vmax_ssim = 0.9
    vmin_ssim = 0.6

    art_low_noise_ssim = np.load("Errors/art_low_noise_ssim.npy")
    art_low_noise_ssim_train = art_low_noise_ssim[:int(train_ratio*nd)]
    art_low_noise_ssim_train = np.concatenate((art_low_noise_ssim_train, art_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    art_low_noise_ssim_test = art_low_noise_ssim[int(train_ratio*nd):nd]
    art_low_noise_ssim_test = np.concatenate((art_low_noise_ssim_test, art_low_noise_ssim[nd + int(train_ratio*no):]))

    art_low_medium_ssim = np.load("Errors/art_low_medium_ssim.npy")
    art_low_medium_ssim_train = art_low_medium_ssim[:int(train_ratio*nd)]
    art_low_medium_ssim_train = np.concatenate((art_low_medium_ssim_train, art_low_medium_ssim[nd:nd + int(train_ratio*no)]))
    art_low_medium_ssim_test = art_low_medium_ssim[int(train_ratio*nd):nd]
    art_low_medium_ssim_test = np.concatenate((art_low_medium_ssim_test, art_low_medium_ssim[nd + int(train_ratio*no):]))

    art_low_high_ssim = np.load("Errors/art_low_high_ssim.npy")
    art_low_high_ssim_train = art_low_high_ssim[:int(train_ratio*nd)]
    art_low_high_ssim_train = np.concatenate((art_low_high_ssim_train, art_low_high_ssim[nd:nd + int(train_ratio*no)]))
    art_low_high_ssim_test = art_low_high_ssim[int(train_ratio*nd):nd]
    art_low_high_ssim_test = np.concatenate((art_low_high_ssim_test, art_low_high_ssim[nd + int(train_ratio*no):]))

    art_medium_noise_ssim = np.load("Errors/art_medium_noise_ssim.npy")
    art_medium_noise_ssim_train = art_medium_noise_ssim[:int(train_ratio*nd)]
    art_medium_noise_ssim_train = np.concatenate((art_medium_noise_ssim_train, art_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    art_medium_noise_ssim_test = art_medium_noise_ssim[int(train_ratio*nd):nd]
    art_medium_noise_ssim_test = np.concatenate((art_medium_noise_ssim_test, art_medium_noise_ssim[nd + int(train_ratio*no):]))

    art_medium_low_ssim = np.load("Errors/art_medium_low_ssim.npy")
    art_medium_low_ssim_train = art_medium_low_ssim[:int(train_ratio*nd)]
    art_medium_low_ssim_train = np.concatenate((art_medium_low_ssim_train, art_medium_low_ssim[nd:nd + int(train_ratio*no)]))
    art_medium_low_ssim_test = art_medium_low_ssim[int(train_ratio*nd):nd]
    art_medium_low_ssim_test = np.concatenate((art_medium_low_ssim_test, art_medium_low_ssim[nd + int(train_ratio*no):]))

    art_medium_high_ssim = np.load("Errors/art_medium_high_ssim.npy")
    art_medium_high_ssim_train = art_medium_high_ssim[:int(train_ratio*nd)]
    art_medium_high_ssim_train = np.concatenate((art_medium_high_ssim_train, art_medium_high_ssim[nd:nd + int(train_ratio*no)]))
    art_medium_high_ssim_test = art_medium_high_ssim[int(train_ratio*nd):nd]
    art_medium_high_ssim_test = np.concatenate((art_medium_high_ssim_test, art_medium_high_ssim[nd + int(train_ratio*no):]))

    art_high_noise_ssim = np.load("Errors/art_high_noise_ssim.npy")
    art_high_noise_ssim_train = art_high_noise_ssim[:int(train_ratio*nd)]
    art_high_noise_ssim_train = np.concatenate((art_high_noise_ssim_train, art_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    art_high_noise_ssim_test = art_high_noise_ssim[int(train_ratio*nd):nd]
    art_high_noise_ssim_test = np.concatenate((art_high_noise_ssim_test, art_high_noise_ssim[nd + int(train_ratio*no):]))

    art_high_low_ssim = np.load("Errors/art_high_low_ssim.npy")
    art_high_low_ssim_train = art_high_low_ssim[:int(train_ratio*nd)]
    art_high_low_ssim_train = np.concatenate((art_high_low_ssim_train, art_high_low_ssim[nd:nd + int(train_ratio*no)]))
    art_high_low_ssim_test = art_high_low_ssim[int(train_ratio*nd):nd]
    art_high_low_ssim_test = np.concatenate((art_high_low_ssim_test, art_high_low_ssim[nd + int(train_ratio*no):]))

    art_high_medium_ssim = np.load("Errors/art_high_medium_ssim.npy")
    art_high_medium_ssim_train = art_high_medium_ssim[:int(train_ratio*nd)]
    art_high_medium_ssim_train = np.concatenate((art_high_medium_ssim_train, art_high_medium_ssim[nd:nd + int(train_ratio*no)]))
    art_high_medium_ssim_test = art_high_medium_ssim[int(train_ratio*nd):nd]
    art_high_medium_ssim_test = np.concatenate((art_high_medium_ssim_test, art_high_medium_ssim[nd + int(train_ratio*no):]))
    
    art_low_noise_ssim_ci = mean_confidence_interval(art_low_noise_ssim_test)
    art_low_medium_ssim_ci = mean_confidence_interval(art_low_medium_ssim_test)
    art_low_high_ssim_ci = mean_confidence_interval(art_low_high_ssim_test)
    art_medium_noise_ssim_ci = mean_confidence_interval(art_medium_noise_ssim_test)
    art_medium_low_ssim_ci = mean_confidence_interval(art_medium_low_ssim_test)
    art_medium_high_ssim_ci = mean_confidence_interval(art_medium_high_ssim_test)
    art_high_noise_ssim_ci = mean_confidence_interval(art_high_noise_ssim_test)
    art_high_low_ssim_ci = mean_confidence_interval(art_high_low_ssim_test)
    art_high_medium_ssim_ci = mean_confidence_interval(art_high_medium_ssim_test)
    
    art_map = np.array([[art_low_noise_ssim_ci[0], art_low_medium_ssim_ci[0], art_low_high_ssim_ci[0]],
               [art_medium_low_ssim_ci[0], art_medium_noise_ssim_ci[0], art_medium_high_ssim_ci[0]],
               [art_high_low_ssim_ci[0], art_high_medium_ssim_ci[0], art_high_noise_ssim_ci[0]]])
    art_map = np.transpose(art_map)

    art_h = np.array([[art_low_noise_ssim_ci[1], art_low_medium_ssim_ci[1], art_low_high_ssim_ci[1]],
               [art_medium_low_ssim_ci[1], art_medium_noise_ssim_ci[1], art_medium_high_ssim_ci[1]],
               [art_high_low_ssim_ci[1], art_high_medium_ssim_ci[1], art_high_noise_ssim_ci[1]]])
    art_h = np.transpose(art_h)
    
    
    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(art_map, vmin = vmin_ssim, vmax = vmax_ssim, cmap = 'Purples_r')

    art_map = np.round(100*art_map)/100
    art_h = np.round(100*art_h)/100
    ax.set_xticks(range(3), noise_labels)
    ax.set_yticks(range(3), noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax.text(i, j, str(art_map[j, i]) + r"$\pm$" + str(art_h[j, i]),
                       ha="center", va="center", color="black")
    ax.set_ylabel("Testing Noise", fontsize = 18)
    ax.set_xlabel("Training Noise", fontsize = 18)
    ax.set_title("Artifact Correction Confusion",  fontsize = 24)
    plt.savefig("Figures/art_ssim_confusion.png", bbox_inches='tight')


    dc_low_noise_ssim = np.load("Errors/dc_low_noise_ssim.npy")
    dc_low_noise_ssim_train = dc_low_noise_ssim[:int(train_ratio*nd)]
    dc_low_noise_ssim_train = np.concatenate((dc_low_noise_ssim_train, dc_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    dc_low_noise_ssim_test = dc_low_noise_ssim[int(train_ratio*nd):nd]
    dc_low_noise_ssim_test = np.concatenate((dc_low_noise_ssim_test, dc_low_noise_ssim[nd + int(train_ratio*no):]))

    dc_low_medium_ssim = np.load("Errors/dc_low_medium_ssim.npy")
    dc_low_medium_ssim_train = dc_low_medium_ssim[:int(train_ratio*nd)]
    dc_low_medium_ssim_train = np.concatenate((dc_low_medium_ssim_train, dc_low_medium_ssim[nd:nd + int(train_ratio*no)]))
    dc_low_medium_ssim_test = dc_low_medium_ssim[int(train_ratio*nd):nd]
    dc_low_medium_ssim_test = np.concatenate((dc_low_medium_ssim_test, dc_low_medium_ssim[nd + int(train_ratio*no):]))

    dc_low_high_ssim = np.load("Errors/dc_low_high_ssim.npy")
    dc_low_high_ssim_train = dc_low_high_ssim[:int(train_ratio*nd)]
    dc_low_high_ssim_train = np.concatenate((dc_low_high_ssim_train, dc_low_high_ssim[nd:nd + int(train_ratio*no)]))
    dc_low_high_ssim_test = dc_low_high_ssim[int(train_ratio*nd):nd]
    dc_low_high_ssim_test = np.concatenate((dc_low_high_ssim_test, dc_low_high_ssim[nd + int(train_ratio*no):]))

    dc_medium_noise_ssim = np.load("Errors/dc_medium_noise_ssim.npy")
    dc_medium_noise_ssim_train = dc_medium_noise_ssim[:int(train_ratio*nd)]
    dc_medium_noise_ssim_train = np.concatenate((dc_medium_noise_ssim_train, dc_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    dc_medium_noise_ssim_test = dc_medium_noise_ssim[int(train_ratio*nd):nd]
    dc_medium_noise_ssim_test = np.concatenate((dc_medium_noise_ssim_test, dc_medium_noise_ssim[nd + int(train_ratio*no):]))

    dc_medium_low_ssim = np.load("Errors/dc_medium_low_ssim.npy")
    dc_medium_low_ssim_train = dc_medium_low_ssim[:int(train_ratio*nd)]
    dc_medium_low_ssim_train = np.concatenate((dc_medium_low_ssim_train, dc_medium_low_ssim[nd:nd + int(train_ratio*no)]))
    dc_medium_low_ssim_test = dc_medium_low_ssim[int(train_ratio*nd):nd]
    dc_medium_low_ssim_test = np.concatenate((dc_medium_low_ssim_test, dc_medium_low_ssim[nd + int(train_ratio*no):]))

    dc_medium_high_ssim = np.load("Errors/dc_medium_high_ssim.npy")
    dc_medium_high_ssim_train = dc_medium_high_ssim[:int(train_ratio*nd)]
    dc_medium_high_ssim_train = np.concatenate((dc_medium_high_ssim_train, dc_medium_high_ssim[nd:nd + int(train_ratio*no)]))
    dc_medium_high_ssim_test = dc_medium_high_ssim[int(train_ratio*nd):nd]
    dc_medium_high_ssim_test = np.concatenate((dc_medium_high_ssim_test, dc_medium_high_ssim[nd + int(train_ratio*no):]))

    dc_high_noise_ssim = np.load("Errors/dc_high_noise_ssim.npy")
    dc_high_noise_ssim_train = dc_high_noise_ssim[:int(train_ratio*nd)]
    dc_high_noise_ssim_train = np.concatenate((dc_high_noise_ssim_train, dc_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    dc_high_noise_ssim_test = dc_high_noise_ssim[int(train_ratio*nd):nd]
    dc_high_noise_ssim_test = np.concatenate((dc_high_noise_ssim_test, dc_high_noise_ssim[nd + int(train_ratio*no):]))

    dc_high_low_ssim = np.load("Errors/dc_high_low_ssim.npy")
    dc_high_low_ssim_train = dc_high_low_ssim[:int(train_ratio*nd)]
    dc_high_low_ssim_train = np.concatenate((dc_high_low_ssim_train, dc_high_low_ssim[nd:nd + int(train_ratio*no)]))
    dc_high_low_ssim_test = dc_high_low_ssim[int(train_ratio*nd):nd]
    dc_high_low_ssim_test = np.concatenate((dc_high_low_ssim_test, dc_high_low_ssim[nd + int(train_ratio*no):]))

    dc_high_medium_ssim = np.load("Errors/dc_high_medium_ssim.npy")
    dc_high_medium_ssim_train = dc_high_medium_ssim[:int(train_ratio*nd)]
    dc_high_medium_ssim_train = np.concatenate((dc_high_medium_ssim_train, dc_high_medium_ssim[nd:nd + int(train_ratio*no)]))
    dc_high_medium_ssim_test = dc_high_medium_ssim[int(train_ratio*nd):nd]
    dc_high_medium_ssim_test = np.concatenate((dc_high_medium_ssim_test, dc_high_medium_ssim[nd + int(train_ratio*no):]))
    
    dc_low_noise_ssim_ci = mean_confidence_interval(dc_low_noise_ssim_test)
    dc_low_medium_ssim_ci = mean_confidence_interval(dc_low_medium_ssim_test)
    dc_low_high_ssim_ci = mean_confidence_interval(dc_low_high_ssim_test)
    dc_medium_noise_ssim_ci = mean_confidence_interval(dc_medium_noise_ssim_test)
    dc_medium_low_ssim_ci = mean_confidence_interval(dc_medium_low_ssim_test)
    dc_medium_high_ssim_ci = mean_confidence_interval(dc_medium_high_ssim_test)
    dc_high_noise_ssim_ci = mean_confidence_interval(dc_high_noise_ssim_test)
    dc_high_low_ssim_ci = mean_confidence_interval(dc_high_low_ssim_test)
    dc_high_medium_ssim_ci = mean_confidence_interval(dc_high_medium_ssim_test)
    
    dc_map = np.array([[dc_low_noise_ssim_ci[0], dc_low_medium_ssim_ci[0], dc_low_high_ssim_ci[0]],
               [dc_medium_low_ssim_ci[0], dc_medium_noise_ssim_ci[0], dc_medium_high_ssim_ci[0]],
               [dc_high_low_ssim_ci[0], dc_high_medium_ssim_ci[0], dc_high_noise_ssim_ci[0]]])
    dc_map = np.transpose(dc_map)

    dc_h = np.array([[dc_low_noise_ssim_ci[1], dc_low_medium_ssim_ci[1], dc_low_high_ssim_ci[1]],
               [dc_medium_low_ssim_ci[1], dc_medium_noise_ssim_ci[1], dc_medium_high_ssim_ci[1]],
               [dc_high_low_ssim_ci[1], dc_high_medium_ssim_ci[1], dc_high_noise_ssim_ci[1]]])
    dc_h = np.transpose(dc_h)
    
    
    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(dc_map, vmin = vmin_ssim, vmax = vmax_ssim, cmap = 'Purples_r')

    dc_map = np.round(100*dc_map)/100
    dc_h = np.round(100*dc_h)/100
    ax.set_xticks(range(3), noise_labels)
    ax.set_yticks(range(3), noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax.text(i, j, str(dc_map[j, i]) + r"$\pm$" + str(dc_h[j, i]),
                       ha="center", va="center", color="black")
    ax.set_ylabel("Testing Noise", fontsize = 18)
    ax.set_xlabel("Training Noise", fontsize = 18)
    ax.set_title("Artifact Correction Confusion",  fontsize = 24)
    plt.savefig("Figures/dc_ssim_confusion.png", bbox_inches='tight')

    dual_low_noise_ssim = np.load("Errors/dual_low_noise_ssim.npy")
    dual_low_noise_ssim_train = dual_low_noise_ssim[:int(train_ratio*nd)]
    dual_low_noise_ssim_train = np.concatenate((dual_low_noise_ssim_train, dual_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    dual_low_noise_ssim_test = dual_low_noise_ssim[int(train_ratio*nd):nd]
    dual_low_noise_ssim_test = np.concatenate((dual_low_noise_ssim_test, dual_low_noise_ssim[nd + int(train_ratio*no):]))

    dual_low_medium_ssim = np.load("Errors/dual_low_medium_ssim.npy")
    dual_low_medium_ssim_train = dual_low_medium_ssim[:int(train_ratio*nd)]
    dual_low_medium_ssim_train = np.concatenate((dual_low_medium_ssim_train, dual_low_medium_ssim[nd:nd + int(train_ratio*no)]))
    dual_low_medium_ssim_test = dual_low_medium_ssim[int(train_ratio*nd):nd]
    dual_low_medium_ssim_test = np.concatenate((dual_low_medium_ssim_test, dual_low_medium_ssim[nd + int(train_ratio*no):]))

    dual_low_high_ssim = np.load("Errors/dual_low_high_ssim.npy")
    dual_low_high_ssim_train = dual_low_high_ssim[:int(train_ratio*nd)]
    dual_low_high_ssim_train = np.concatenate((dual_low_high_ssim_train, dual_low_high_ssim[nd:nd + int(train_ratio*no)]))
    dual_low_high_ssim_test = dual_low_high_ssim[int(train_ratio*nd):nd]
    dual_low_high_ssim_test = np.concatenate((dual_low_high_ssim_test, dual_low_high_ssim[nd + int(train_ratio*no):]))

    dual_medium_noise_ssim = np.load("Errors/dual_medium_noise_ssim.npy")
    dual_medium_noise_ssim_train = dual_medium_noise_ssim[:int(train_ratio*nd)]
    dual_medium_noise_ssim_train = np.concatenate((dual_medium_noise_ssim_train, dual_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    dual_medium_noise_ssim_test = dual_medium_noise_ssim[int(train_ratio*nd):nd]
    dual_medium_noise_ssim_test = np.concatenate((dual_medium_noise_ssim_test, dual_medium_noise_ssim[nd + int(train_ratio*no):]))

    dual_medium_low_ssim = np.load("Errors/dual_medium_low_ssim.npy")
    dual_medium_low_ssim_train = dual_medium_low_ssim[:int(train_ratio*nd)]
    dual_medium_low_ssim_train = np.concatenate((dual_medium_low_ssim_train, dual_medium_low_ssim[nd:nd + int(train_ratio*no)]))
    dual_medium_low_ssim_test = dual_medium_low_ssim[int(train_ratio*nd):nd]
    dual_medium_low_ssim_test = np.concatenate((dual_medium_low_ssim_test, dual_medium_low_ssim[nd + int(train_ratio*no):]))

    dual_medium_high_ssim = np.load("Errors/dual_medium_high_ssim.npy")
    dual_medium_high_ssim_train = dual_medium_high_ssim[:int(train_ratio*nd)]
    dual_medium_high_ssim_train = np.concatenate((dual_medium_high_ssim_train, dual_medium_high_ssim[nd:nd + int(train_ratio*no)]))
    dual_medium_high_ssim_test = dual_medium_high_ssim[int(train_ratio*nd):nd]
    dual_medium_high_ssim_test = np.concatenate((dual_medium_high_ssim_test, dual_medium_high_ssim[nd + int(train_ratio*no):]))

    dual_high_noise_ssim = np.load("Errors/dual_high_noise_ssim.npy")
    dual_high_noise_ssim_train = dual_high_noise_ssim[:int(train_ratio*nd)]
    dual_high_noise_ssim_train = np.concatenate((dual_high_noise_ssim_train, dual_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    dual_high_noise_ssim_test = dual_high_noise_ssim[int(train_ratio*nd):nd]
    dual_high_noise_ssim_test = np.concatenate((dual_high_noise_ssim_test, dual_high_noise_ssim[nd + int(train_ratio*no):]))

    dual_high_low_ssim = np.load("Errors/dual_high_low_ssim.npy")
    dual_high_low_ssim_train = dual_high_low_ssim[:int(train_ratio*nd)]
    dual_high_low_ssim_train = np.concatenate((dual_high_low_ssim_train, dual_high_low_ssim[nd:nd + int(train_ratio*no)]))
    dual_high_low_ssim_test = dual_high_low_ssim[int(train_ratio*nd):nd]
    dual_high_low_ssim_test = np.concatenate((dual_high_low_ssim_test, dual_high_low_ssim[nd + int(train_ratio*no):]))

    dual_high_medium_ssim = np.load("Errors/dual_high_medium_ssim.npy")
    dual_high_medium_ssim_train = dual_high_medium_ssim[:int(train_ratio*nd)]
    dual_high_medium_ssim_train = np.concatenate((dual_high_medium_ssim_train, dual_high_medium_ssim[nd:nd + int(train_ratio*no)]))
    dual_high_medium_ssim_test = dual_high_medium_ssim[int(train_ratio*nd):nd]
    dual_high_medium_ssim_test = np.concatenate((dual_high_medium_ssim_test, dual_high_medium_ssim[nd + int(train_ratio*no):]))
    
    dual_low_noise_ssim_ci = mean_confidence_interval(dual_low_noise_ssim_test)
    dual_low_medium_ssim_ci = mean_confidence_interval(dual_low_medium_ssim_test)
    dual_low_high_ssim_ci = mean_confidence_interval(dual_low_high_ssim_test)
    dual_medium_noise_ssim_ci = mean_confidence_interval(dual_medium_noise_ssim_test)
    dual_medium_low_ssim_ci = mean_confidence_interval(dual_medium_low_ssim_test)
    dual_medium_high_ssim_ci = mean_confidence_interval(dual_medium_high_ssim_test)
    dual_high_noise_ssim_ci = mean_confidence_interval(dual_high_noise_ssim_test)
    dual_high_low_ssim_ci = mean_confidence_interval(dual_high_low_ssim_test)
    dual_high_medium_ssim_ci = mean_confidence_interval(dual_high_medium_ssim_test)
    
    dual_map = np.array([[dual_low_noise_ssim_ci[0], dual_low_medium_ssim_ci[0], dual_low_high_ssim_ci[0]],
               [dual_medium_low_ssim_ci[0], dual_medium_noise_ssim_ci[0], dual_medium_high_ssim_ci[0]],
               [dual_high_low_ssim_ci[0], dual_high_medium_ssim_ci[0], dual_high_noise_ssim_ci[0]]])
    dual_map = np.transpose(dual_map)

    dual_h = np.array([[dual_low_noise_ssim_ci[1], dual_low_medium_ssim_ci[1], dual_low_high_ssim_ci[1]],
               [dual_medium_low_ssim_ci[1], dual_medium_noise_ssim_ci[1], dual_medium_high_ssim_ci[1]],
               [dual_high_low_ssim_ci[1], dual_high_medium_ssim_ci[1], dual_high_noise_ssim_ci[1]]])
    dual_h = np.transpose(dual_h)
    
    
    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(dual_map, vmin = vmin_ssim, vmax = vmax_ssim, cmap = 'Purples_r')

    dual_map = np.round(100*dual_map)/100
    dual_h = np.round(100*dual_h)/100
    ax.set_xticks(range(3), noise_labels)
    ax.set_yticks(range(3), noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax.text(i, j, str(dual_map[j, i]) + r"$\pm$" + str(dual_h[j, i]),
                       ha="center", va="center", color="black")
    ax.set_ylabel("Testing Noise", fontsize = 18)
    ax.set_xlabel("Training Noise", fontsize = 18)
    ax.set_title("Artifact Correction Confusion",  fontsize = 24)
    plt.savefig("Figures/dual_ssim_confusion.png", bbox_inches='tight')

    inet_low_noise_ssim = np.load("Errors/inet_low_noise_ssim.npy")
    inet_low_noise_ssim_train = inet_low_noise_ssim[:int(train_ratio*nd)]
    inet_low_noise_ssim_train = np.concatenate((inet_low_noise_ssim_train, inet_low_noise_ssim[nd:nd + int(train_ratio*no)]))
    inet_low_noise_ssim_test = inet_low_noise_ssim[int(train_ratio*nd):nd]
    inet_low_noise_ssim_test = np.concatenate((inet_low_noise_ssim_test, inet_low_noise_ssim[nd + int(train_ratio*no):]))

    inet_low_medium_ssim = np.load("Errors/inet_low_medium_ssim.npy")
    inet_low_medium_ssim_train = inet_low_medium_ssim[:int(train_ratio*nd)]
    inet_low_medium_ssim_train = np.concatenate((inet_low_medium_ssim_train, inet_low_medium_ssim[nd:nd + int(train_ratio*no)]))
    inet_low_medium_ssim_test = inet_low_medium_ssim[int(train_ratio*nd):nd]
    inet_low_medium_ssim_test = np.concatenate((inet_low_medium_ssim_test, inet_low_medium_ssim[nd + int(train_ratio*no):]))

    inet_low_high_ssim = np.load("Errors/inet_low_high_ssim.npy")
    inet_low_high_ssim_train = inet_low_high_ssim[:int(train_ratio*nd)]
    inet_low_high_ssim_train = np.concatenate((inet_low_high_ssim_train, inet_low_high_ssim[nd:nd + int(train_ratio*no)]))
    inet_low_high_ssim_test = inet_low_high_ssim[int(train_ratio*nd):nd]
    inet_low_high_ssim_test = np.concatenate((inet_low_high_ssim_test, inet_low_high_ssim[nd + int(train_ratio*no):]))

    inet_medium_noise_ssim = np.load("Errors/inet_medium_noise_ssim.npy")
    inet_medium_noise_ssim_train = inet_medium_noise_ssim[:int(train_ratio*nd)]
    inet_medium_noise_ssim_train = np.concatenate((inet_medium_noise_ssim_train, inet_medium_noise_ssim[nd:nd + int(train_ratio*no)]))
    inet_medium_noise_ssim_test = inet_medium_noise_ssim[int(train_ratio*nd):nd]
    inet_medium_noise_ssim_test = np.concatenate((inet_medium_noise_ssim_test, inet_medium_noise_ssim[nd + int(train_ratio*no):]))

    inet_medium_low_ssim = np.load("Errors/inet_medium_low_ssim.npy")
    inet_medium_low_ssim_train = inet_medium_low_ssim[:int(train_ratio*nd)]
    inet_medium_low_ssim_train = np.concatenate((inet_medium_low_ssim_train, inet_medium_low_ssim[nd:nd + int(train_ratio*no)]))
    inet_medium_low_ssim_test = inet_medium_low_ssim[int(train_ratio*nd):nd]
    inet_medium_low_ssim_test = np.concatenate((inet_medium_low_ssim_test, inet_medium_low_ssim[nd + int(train_ratio*no):]))

    inet_medium_high_ssim = np.load("Errors/inet_medium_high_ssim.npy")
    inet_medium_high_ssim_train = inet_medium_high_ssim[:int(train_ratio*nd)]
    inet_medium_high_ssim_train = np.concatenate((inet_medium_high_ssim_train, inet_medium_high_ssim[nd:nd + int(train_ratio*no)]))
    inet_medium_high_ssim_test = inet_medium_high_ssim[int(train_ratio*nd):nd]
    inet_medium_high_ssim_test = np.concatenate((inet_medium_high_ssim_test, inet_medium_high_ssim[nd + int(train_ratio*no):]))

    inet_high_noise_ssim = np.load("Errors/inet_high_noise_ssim.npy")
    inet_high_noise_ssim_train = inet_high_noise_ssim[:int(train_ratio*nd)]
    inet_high_noise_ssim_train = np.concatenate((inet_high_noise_ssim_train, inet_high_noise_ssim[nd:nd + int(train_ratio*no)]))
    inet_high_noise_ssim_test = inet_high_noise_ssim[int(train_ratio*nd):nd]
    inet_high_noise_ssim_test = np.concatenate((inet_high_noise_ssim_test, inet_high_noise_ssim[nd + int(train_ratio*no):]))

    inet_high_low_ssim = np.load("Errors/inet_high_low_ssim.npy")
    inet_high_low_ssim_train = inet_high_low_ssim[:int(train_ratio*nd)]
    inet_high_low_ssim_train = np.concatenate((inet_high_low_ssim_train, inet_high_low_ssim[nd:nd + int(train_ratio*no)]))
    inet_high_low_ssim_test = inet_high_low_ssim[int(train_ratio*nd):nd]
    inet_high_low_ssim_test = np.concatenate((inet_high_low_ssim_test, inet_high_low_ssim[nd + int(train_ratio*no):]))

    inet_high_medium_ssim = np.load("Errors/inet_high_medium_ssim.npy")
    inet_high_medium_ssim_train = inet_high_medium_ssim[:int(train_ratio*nd)]
    inet_high_medium_ssim_train = np.concatenate((inet_high_medium_ssim_train, inet_high_medium_ssim[nd:nd + int(train_ratio*no)]))
    inet_high_medium_ssim_test = inet_high_medium_ssim[int(train_ratio*nd):nd]
    inet_high_medium_ssim_test = np.concatenate((inet_high_medium_ssim_test, inet_high_medium_ssim[nd + int(train_ratio*no):]))
    
    inet_low_noise_ssim_ci = mean_confidence_interval(inet_low_noise_ssim_test)
    inet_low_medium_ssim_ci = mean_confidence_interval(inet_low_medium_ssim_test)
    inet_low_high_ssim_ci = mean_confidence_interval(inet_low_high_ssim_test)
    inet_medium_noise_ssim_ci = mean_confidence_interval(inet_medium_noise_ssim_test)
    inet_medium_low_ssim_ci = mean_confidence_interval(inet_medium_low_ssim_test)
    inet_medium_high_ssim_ci = mean_confidence_interval(inet_medium_high_ssim_test)
    inet_high_noise_ssim_ci = mean_confidence_interval(inet_high_noise_ssim_test)
    inet_high_low_ssim_ci = mean_confidence_interval(inet_high_low_ssim_test)
    inet_high_medium_ssim_ci = mean_confidence_interval(inet_high_medium_ssim_test)
    
    inet_map = np.array([[inet_low_noise_ssim_ci[0], inet_low_medium_ssim_ci[0], inet_low_high_ssim_ci[0]],
               [inet_medium_low_ssim_ci[0], inet_medium_noise_ssim_ci[0], inet_medium_high_ssim_ci[0]],
               [inet_high_low_ssim_ci[0], inet_high_medium_ssim_ci[0], inet_high_noise_ssim_ci[0]]])
    inet_map = np.transpose(inet_map)

    inet_h = np.array([[inet_low_noise_ssim_ci[1], inet_low_medium_ssim_ci[1], inet_low_high_ssim_ci[1]],
               [inet_medium_low_ssim_ci[1], inet_medium_noise_ssim_ci[1], inet_medium_high_ssim_ci[1]],
               [inet_high_low_ssim_ci[1], inet_high_medium_ssim_ci[1], inet_high_noise_ssim_ci[1]]])
    inet_h = np.transpose(inet_h)
    
    
    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(inet_map, vmin = vmin_ssim, vmax = vmax_ssim, cmap = 'Purples_r')

    inet_map = np.round(100*inet_map)/100
    inet_h = np.round(100*inet_h)/100
    ax.set_xticks(range(3), noise_labels)
    ax.set_yticks(range(3), noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax.text(i, j, str(inet_map[j, i]) + r"$\pm$" + str(inet_h[j, i]),
                       ha="center", va="center", color="black")
    ax.set_ylabel("Testing Noise", fontsize = 18)
    ax.set_xlabel("Training Noise", fontsize = 18)
    ax.set_title("Artifact Correction Confusion",  fontsize = 24)
    plt.savefig("Figures/inet_ssim_confusion.png", bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots(1,4)
    fig.set_figheight(4)
    fig.set_figwidth(16)
    ax[0].imshow(inet_map, vmin = vmin_ssim, vmax = vmax_ssim, cmap = 'Purples_r')
    ax[0].set_xticks(range(3), noise_labels)
    ax[0].set_yticks(range(3), noise_labels, rotation = 90)
    for i in range(3):
        for j in range(3):
            text = ax[0].text(i, j, str(inet_map[j, i]) + r"$\pm$" + str(inet_h[j, i]),
                       ha="center", va="center", color="white")
    ax[0].set_ylabel("Testing Noise", fontsize = 18)
    #ax[0].set_xlabel("Training Noise", fontsize = 18)
    #ax[0].set_title("InversionNet",  fontsize = 20)
    ax[0].set_title(" ",  fontsize = 20)


    ax[1].imshow(art_map, vmin = vmin_ssim, vmax = vmax_ssim, cmap = 'Purples_r')
    ax[1].set_xticks(range(3), noise_labels)
    ax[1].set_yticks(range(3), [""]*3)#noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax[1].text(i, j, str(art_map[j, i]) + r"$\pm$" + str(art_h[j, i]),
                       ha="center", va="center", color="black")
    #ax[1].set_ylabel("Testing Noise", fontsize = 18)
    #ax[1].set_xlabel("Training Noise", fontsize = 18)
    #ax[1].set_title("Artifact Correction",  fontsize = 20)
    ax[1].set_title(" ",  fontsize = 20)


    ax[2].imshow(dc_map, vmin = vmin_ssim, vmax = vmax_ssim, cmap = 'Purples_r')
    ax[2].set_xticks(range(3), noise_labels)
    ax[2].set_yticks(range(3), [""]*3)#noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax[2].text(i, j, str(dc_map[j, i]) + r"$\pm$" + str(dc_h[j, i]),
                       ha="center", va="center", color="white")
    #ax.set_ylabel("Testing Noise", fontsize = 18)
    #ax[2].set_xlabel("Training Noise", fontsize = 18)
    #ax[2].set_title("Data Correction",  fontsize = 20)
    ax[2].set_title(" ",  fontsize = 20)

    im = ax[3].imshow(dual_map, vmin = vmin_ssim, vmax = vmax_ssim, cmap = 'Purples_r')
    ax[3].set_xticks(range(3), noise_labels)
    ax[3].set_yticks(range(3), [""]*3)#noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax[3].text(i, j, str(dual_map[j, i]) + r"$\pm$" + str(dual_h[j, i]),
                       ha="center", va="center", color="black")
    #ax[3].set_ylabel("Testing Noise", fontsize = 18)
    #ax[3].set_xlabel("Training Noise", fontsize = 18)
    #ax[3].set_title("Dual Correction",  fontsize = 20)
    ax[3].set_title(" ",  fontsize = 20)
    fig.subplots_adjust(right=0.84)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)



    #fig.x("Training Noise", fontsize = 18)
    #fig.text(0.5, 0.03, 'Training Noise', ha='center', fontsize = 18)
    fig.text(0.5, 0.03, ' ', ha='center', fontsize = 18)
    fig.text(0.08, 0.44, 'SSIM', ha='center', fontsize = 18, rotation = 90)
    #fig.title("SSIM Confusion")
    plt.subplots_adjust(wspace=0.05)
    plt.savefig("Figures/combined_ssim_confusion.png", bbox_inches='tight')


    vmax_auc = 1.0
    vmin_auc = 0.7
    
    aucs = io.loadmat('aucs.mat')
    art_map = np.array([[aucs['ac_recon_low_noise'][0,0], aucs['ac_recon_low_medium'][0,0], aucs['ac_recon_low_high'][0,0]],
                        [aucs['ac_recon_medium_low'][0,0], aucs['ac_recon_medium_noise'][0,0], aucs['ac_recon_medium_high'][0,0]],
                        [aucs['ac_recon_high_low'][0,0], aucs['ac_recon_high_medium'][0,0], aucs['ac_recon_high_noise'][0,0]]])
    art_map = np.transpose(art_map)
    
    
    
    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(art_map, vmin = vmin_auc, vmax = vmax_auc, cmap = 'Greens_r')

    art_map = np.round(1000*art_map)/1000
    ax.set_xticks(range(3), noise_labels)
    ax.set_yticks(range(3), noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax.text(i, j, str(art_map[j, i]),
                       ha="center", va="center", color="black")
    ax.set_ylabel("Testing Noise", fontsize = 18)
    ax.set_xlabel("Training Noise", fontsize = 18)
    ax.set_title("Artifact Correction Confusion",  fontsize = 24)
    plt.savefig("Figures/art_auc_confusion.png", bbox_inches='tight')
    
    dc_map = np.array([[aucs['dc_recon_low_noise'][0,0], aucs['dc_recon_low_medium'][0,0], aucs['dc_recon_low_high'][0,0]],
                        [aucs['dc_recon_medium_low'][0,0], aucs['dc_recon_medium_noise'][0,0], aucs['dc_recon_medium_high'][0,0]],
                        [aucs['dc_recon_high_low'][0,0], aucs['dc_recon_high_medium'][0,0], aucs['dc_recon_high_noise'][0,0]]])

    dc_map = np.transpose(dc_map)
    
    
    
    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(dc_map, vmin = vmin_auc, vmax = vmax_auc, cmap = 'Greens_r')

    dc_map = np.round(1000*dc_map)/1000
    ax.set_xticks(range(3), noise_labels)
    ax.set_yticks(range(3), noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax.text(i, j, str(dc_map[j, i]),
                       ha="center", va="center", color="black")
    ax.set_ylabel("Testing Noise", fontsize = 18)
    ax.set_xlabel("Training Noise", fontsize = 18)
    ax.set_title("Artifact Correction Confusion",  fontsize = 24)
    plt.savefig("Figures/dc_auc_confusion.png", bbox_inches='tight')

    
    
    dual_map = np.array([[aucs['dual_recon_low_noise'][0,0], aucs['dual_recon_low_medium'][0,0], aucs['dual_recon_low_high'][0,0]],
                        [aucs['dual_recon_medium_low'][0,0], aucs['dual_recon_medium_noise'][0,0], aucs['dual_recon_medium_high'][0,0]],
                        [aucs['dual_recon_high_low'][0,0], aucs['dual_recon_high_medium'][0,0], aucs['dual_recon_high_noise'][0,0]]])

    dual_map = np.transpose(dual_map)

    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(dual_map, vmin = vmin_auc, vmax = vmax_auc, cmap = 'Greens_r')

    dual_map = np.round(1000*dual_map)/1000
    ax.set_xticks(range(3), noise_labels)
    ax.set_yticks(range(3), noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax.text(i, j, str(dual_map[j, i]),
                       ha="center", va="center", color="black")
    ax.set_ylabel("Testing Noise", fontsize = 18)
    ax.set_xlabel("Training Noise", fontsize = 18)
    ax.set_title("Artifact Correction Confusion",  fontsize = 24)
    plt.savefig("Figures/dual_auc_confusion.png", bbox_inches='tight')

    
    inet_map = np.array([[aucs['inet_recon_low_noise'][0,0], aucs['inet_recon_low_medium'][0,0], aucs['inet_recon_low_high'][0,0]],    
                        [aucs['inet_recon_medium_low'][0,0], aucs['inet_recon_medium_noise'][0,0], aucs['inet_recon_medium_high'][0,0]],
                        [aucs['inet_recon_high_low'][0,0], aucs['inet_recon_high_medium'][0,0], aucs['inet_recon_high_noise'][0,0]]])
    inet_map = np.transpose(inet_map)

    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(inet_map, vmin = vmin_auc, vmax = vmax_auc, cmap = 'Greens_r')

    inet_map = np.round(1000*inet_map)/1000
    ax.set_xticks(range(3), noise_labels)
    ax.set_yticks(range(3), noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax.text(i, j, str(inet_map[j, i]),
                       ha="center", va="center", color="black")
    ax.set_ylabel("Testing Noise", fontsize = 18)
    ax.set_xlabel("Training Noise", fontsize = 18)
    ax.set_title("Artifact Correction Confusion",  fontsize = 24)
    plt.savefig("Figures/inet_auc_confusion.png", bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots(1,4)
    fig.set_figheight(4)
    fig.set_figwidth(16)
    ax[0].imshow(inet_map, vmin = vmin_auc, vmax = vmax_auc, cmap = 'Greens_r')
    ax[0].set_xticks(range(3), noise_labels)
    ax[0].set_yticks(range(3), noise_labels, rotation = 90)
    for i in range(3):
        for j in range(3):
            text = ax[0].text(i, j, str(inet_map[j, i]),
                       ha="center", va="center", color="black")
    ax[0].set_ylabel("Testing Noise", fontsize = 18)
    #ax[0].set_xlabel("Training Noise", fontsize = 18)
    #ax[0].set_title("InversionNet",  fontsize = 20)
    ax[0].set_title(" ",  fontsize = 20)


    ax[1].imshow(art_map, vmin = vmin_auc, vmax = vmax_auc, cmap = 'Greens_r')
    ax[1].set_xticks(range(3), noise_labels)
    ax[1].set_yticks(range(3), [""]*3)#noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax[1].text(i, j, str(art_map[j, i]),
                       ha="center", va="center", color="black")
    #ax[1].set_ylabel("Testing Noise", fontsize = 18)
    #ax[1].set_xlabel("Training Noise", fontsize = 18)
    #ax[1].set_title("Artifact Correction",  fontsize = 20)
    ax[1].set_title(" ",  fontsize = 20)


    ax[2].imshow(dc_map, vmin = vmin_auc, vmax = vmax_auc, cmap = 'Greens_r')
    ax[2].set_xticks(range(3), noise_labels)
    ax[2].set_yticks(range(3), [""]*3)#noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax[2].text(i, j, str(dc_map[j, i]),
                       ha="center", va="center", color="black")
    #ax.set_ylabel("Testing Noise", fontsize = 18)
    #ax[2].set_xlabel("Training Noise", fontsize = 18)
    #ax[2].set_title("Data Correction",  fontsize = 20)
    ax[2].set_title(" ",  fontsize = 20)

    im = ax[3].imshow(dual_map, vmin = vmin_auc, vmax = vmax_auc, cmap = 'Greens_r')
    ax[3].set_xticks(range(3), noise_labels)
    ax[3].set_yticks(range(3), [""]*3)#noise_labels)
    for i in range(3):
        for j in range(3):
            text = ax[3].text(i, j, str(dual_map[j, i]),
                       ha="center", va="center", color="black")
    #ax[3].set_ylabel("Testing Noise", fontsize = 18)
    #ax[3].set_xlabel("Training Noise", fontsize = 18)
    #ax[3].set_title("Dual Correction",  fontsize = 20)
    ax[3].set_title(" ",  fontsize = 20)
    fig.subplots_adjust(right=0.84)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)



    #fig.x("Training Noise", fontsize = 18)
    fig.text(0.5, 0.03, 'Training Noise', ha='center', fontsize = 18)
    fig.text(0.08, 0.46, 'AUC', ha='center', fontsize = 18, rotation = 90)
    #fig.title("SSIM Confusion")
    plt.subplots_adjust(wspace=0.05)
    plt.savefig("Figures/combined_auc_confusion.png", bbox_inches='tight')
    
    