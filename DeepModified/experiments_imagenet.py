# -*- coding: utf-8 -*-
"""
From this script, experiments for ImageNet pictures can be started.
See "configuration" below for the different possible settings.
The results are saved automatically to the folder ./results

It is recommended to run caffe in gpu mode when overlapping is set
to True, otherwise the calculation will take a very long time.

@author: Luisa M Zintgraf
"""

# the following is needed to avoid some error that can be thrown when 
# using matplotlib.pyplot in a linux shell
import matplotlib 
matplotlib.use('Agg')   

# standard imports
import numpy as np
import time
import os

# most important script - relevance estimator
from prediction_difference_analysis import PredDiffAnalyser

# utilities
import utils_classifiers as utlC
import utils_data as utlD
import utils_sampling as utlS
import utils_visualise as utlV
import sensitivity_analysis_caffe as SA


import torch
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

model = models.vgg16_bn(pretrained=True)
model.eval() # congela i gradienti (salva memoria e velocizza le prestazioni)
print(model)

test_indices = None
print('controllo accesso cuda')
HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False

if HAS_CUDA:
    gpu_id = 0
    model.cuda(gpu_id)

# window size (i.e., the size of the pixel patch that is marginalised out in each step)
win_size = 10               # k in alg 1 (see paper)

# indicate whether windows should be overlapping or not
overlapping = False

# settings for sampling 
sampl_style = 'marginal' # choose: conditional / marginal
num_samples = 4
padding_size = 2            # important for conditional sampling,
                            # l = win_size+2*padding_size in alg 1
                            # (see paper)

# set the batch size - the larger, the faster computation will be
# (if caffe crashes with memory error, reduce the batch size)
batch_size = 8


# ------------------------ SET-UP ------------------------


# get the data
X_test, X_test_im, X_filenames = utlD.get_imagenet_data(net=model)

# get the label names of the 1000 ImageNet classes
classnames = utlD.get_imagenet_classnames()

if not test_indices:
    test_indices = [i for i in range(X_test.shape[0])]      

# make folder for saving the results if it doesn't exist
path_results = './results/'
if not os.path.exists(path_results):
    os.makedirs(path_results)          
          
# ------------------------ EXPERIMENTS ------------------------

# change the batch size of the network to the given value
#net.blobs['data'].reshape(batch_size, X_test.shape[1], X_test.shape[2], X_test.shape[3])

# target function (mapping input features to output probabilities)
target_func = lambda x: utlC.forward_pass(model, x, None,HAS_CUDA=HAS_CUDA)

# for the given test indices, do the prediction difference analysis
for test_idx in test_indices:
      
    # get the specific image (preprocessed, can be used as input to the target function)
    x_test = X_test[test_idx]
    # get the image for plotting (not preprocessed)
    x_test_im = X_test_im[test_idx]
    # prediction of the network
    y_pred = np.argmax(utlC.forward_pass(model, x_test, ['prob'],HAS_CUDA=HAS_CUDA))
    y_pred_label = classnames[y_pred]
    
    print("{}:{}".format(X_filenames[test_idx],y_pred_label))                     
    # get the path for saving the results
    if sampl_style == 'conditional':
        save_path = path_results+'{}_{}_winSize{}_condSampl_numSampl{}_paddSize{}_{}'.format(X_filenames[test_idx],y_pred_label,win_size,num_samples,padding_size,netname)
    elif sampl_style == 'marginal':
        save_path = path_results+'{}_{}_winSize{}_margSampl_numSampl{}_{}'.format(X_filenames[test_idx],y_pred_label,win_size,num_samples,"test")

    if os.path.exists(save_path+'.npz'):
        print('Results for ', X_filenames[test_idx], ' exist, will move to the next image. ')
        continue
                 
    print("doing test...", "file :", X_filenames[test_idx], ", net:", "netname", ", win_size:", win_size, ", sampling: ", sampl_style)

    # compute the sensitivity map
    #layer_name = net.blobs.keys()[-2] # look at penultimate layer (like in Simonyan et al. (2013))
    #sensMap = SA.get_sens_map(net, x_test[np.newaxis], layer_name, np.argmax(target_func(x_test)[-1][0]))                 

    start_time = time.time()
    
    if sampl_style == 'conditional':
        sampler = utlS.cond_sampler_imagenet(win_size=win_size, padding_size=padding_size, image_dims=[224,224])
    elif sampl_style == 'marginal':
        sampler = utlS.marg_sampler_imagenet(X_test)
        
    pda = PredDiffAnalyser(x_test, target_func, sampler, num_samples=num_samples, batch_size=batch_size)
    pred_diff = pda.get_rel_vect(win_size=win_size, overlap=overlapping)
    #print(len(pred_diff))
    # plot and save the results
    utlV.plot_results(x_test, x_test_im, None, pred_diff[0], target_func, classnames, test_idx, save_path)
    np.savez(save_path, *pred_diff)
    print("--- Total computation took {:.4f} minutes ---".format((time.time() - start_time)/60))
    
        


        
