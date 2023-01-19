# Deep Gaussian Processes for classification with multiple noisy annotators. Application to breast cancer tissue classification



### Code of Deep Gaussian Processes for crowdsourcing 
#### Citation
~~~
@article{lopez2023deep,
  author={López-Pérez, Miguel and Morales-Álvarez, Pablo and Cooper, Lee A. D. and Molina, Rafael and Katsaggelos, Aggelos K.},
  journal={IEEE Access}, 
  title={Deep Gaussian Processes for classification with multiple noisy annotators. {A}pplication to breast cancer tissue classification}, 
  year={2023},
  volume={},
  number={},
  pages={}}
~~~

## Repo
Find the features, preprocessing and all the info about the data in the [repo](https://github.com/wizmik12/crowdsourcing-digital-pathology-GPs) of our previous work.

To use Deep Gaussian Processes, you need this [repo](https://github.com/UCL-SML/Doubly-Stochastic-DGP) installed.

## Abstract
Machine learning (ML) methods often require large volumes of labeled data to achieve
meaningful performance. The expertise necessary for labeling data in medical applications like pathology
presents a significant challenge in developing clinical-grade tools. Crowdsourcing approaches address this
challenge by collecting labels from multiple annotators with varying degrees of expertise. In recent years,
multiple methods have been adapted to learn from noisy crowdsourced labels. Among them, Gaussian
Processes (GPs) have achieved excellent performance due to their ability to model uncertainty. Deep
Gaussian Processes (DGPs) address the limitations of GPs using multiple layers to enable the learning
of more complex representations. In this work, we develop Deep Gaussian Processes for Crowdsourcing
(DGPCR) to model the crowdsourcing problem with DGPs for the first time. DGPCR models the (unknown)
underlying true labels, and the behavior of each annotator is modeled with a confusion matrix among classes.
We use end-to-end variational inference to estimate both DGPCR parameters and annotator biases. Using
annotations from 25 pathologists and medical trainees, we show that DGPCR is competitive or superior
to Scalable Gaussian Processes for Crowdsourcing (SVGPCR) and other state-of-the-art deep-learning
crowdsourcing methods for breast cancer classification. Also, we observe that DGPCR with noisy labels
obtains better results (F1 = 81.91%) than GPs (F1 = 81.57%) and deep learning methods (F1 = 80.88%)
with true labels curated by experts. Finally, we show an improved estimation of annotators’ behavior.
