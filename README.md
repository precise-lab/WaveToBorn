# Learned Correction Methods for USCT Imaging 

Companion code for 
> L. Lozenski, H. Wang, F. Li, M. Anastasio, B. Wohlberg, Y. Lin, U. Villa. _Learned Correction Methods for Ultrasound Computed Tomography Imaging Using Simplified Physics Models_, arXiv ([preprint](https://arxiv.org/abs/2205.05585?context=eess(https://arxiv.org/abs/2502.09546)))

Learned correction methods for applying the Born approximation for USCT image reconstruciton governed by the acoustic wave equation. This repository provides two approaches for correction, artifact correction and data correction. 
Artifact correciton applies the Born approximation for reconstruction from USCT measurements generated utilizing the the acoustic wave equation then utilizes a convolutional neural network (CNN) to correct artifacts due to model mismatch. 
Measurement correction utilizes a CNN to preprocess wave equation data so that it is more compatible with the Born approximation then applies the Born approximation for inversion. 
These two correction approaches are also combined in a dual correction approach which incorporates artifact correction and dual correction together. 

This repository provides code for training and performing multiple numerical expirements utilizing these correction aproaches with comparison to a traditional full waveform inversion (FWI) method for USCT utilizing the wave equation, an uncorrected method employing the Born approximation, and a data-driven learned reconstruciton method utilizing the InversionNet architecture.
Each of these reconstruction methods is performed at three different noise levels and two different distributions of images.
Image accuracy can then be computed in terms of relative root mean square error (RRMSE), structural similarity index measure (SSIM), and a task based assesment of accuracy in tumor segmentation. 
This task based metric is constructed by training a U-net tumor segmenter on each set of training reconstructions, then performing a receiver operator characteristic (ROC) analysis for pixelwise tumor detection on the testing set reconstructions. The area under the curve (AUC) for the ROC plot then serves as a single metric of task performance.


# Dependencies 

`PyTorch`: open source machine learning framework that accelerates the path from research prototyping to production deployment.
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

`scikit-image`: collection of algorithms for image processing
```bash
conda install scikit-image
```


