![image](https://github.com/Bei-Jin/STMFANet/blob/master/.idea/network-2.png)
## Introduction
--------------------------------------------------------------------------------------------
This repository is the implementation of "Exploring Spatial-Temporal Multi-Frequency Analysis for High-Fidelity and Temporal-Consistency Video Prediction". (arXiv report [here](https://arxiv.org/pdf/2002.09905.pdf)).

Video prediction is a pixel-wise dense prediction task to infer future frames based on past frames. Missing appearance details and motion blur are still two major problems for current models, leading to image distortion and temporal inconsistency. We point out the necessity of exploring multi-frequency analysis to deal with the two problems. Inspired by the frequency band decomposition characteristic of Human Vision System (HVS), we propose a video prediction network based on multi-level wavelet analysis to uniformly deal with spatial and temporal information. Specifically, multi-level spatial discrete wavelet transform decomposes each video frame into anisotropic sub-bands with multiple frequencies, helping to enrich structural information and reserve fine details. On the other hand, multi-level temporal discrete wavelet transform which operates on time axis decomposes the frame sequence into sub-band groups of different frequencies to accurately capture multi-frequency motions under fixed frame rate. Extensive experiments on diverse datasets demonstrate that our model shows significant improvements on fidelity and temporal consistency over state-of-the-art works.


## Result show
--------------------------------------------------------------------------------------------

![video](https://github.com/Bei-Jin/STMFANet/blob/master/videos/6645~11.mp4)



Code will come soon.
