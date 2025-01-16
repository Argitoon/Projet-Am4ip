
You can find the starting code at `the gitlab repository am4ip-lab3 <https://gitlab.com/am4ip/am4ip-lab3>`_

Exercise - Implementation of CBD Network
========================================
The goal of this lab is to implement Convolutional Blind Denoising (CBD) Network from `this arXiv paper <https://arxiv.org/pdf/1807.04686v2.pdf>`_
The am4ip library is composed of the following modules:

::

    am4ip
    │   datasets.py
    │   losses.py
    │   metrics.py
    │   models.py
    │   trainer.py
    │   utils.py
    │   utils.py

The role of `datasets.py` is to prepare the data. This includes data loading and pre-processing.

`losses.py` contains loss functions to train the proposed denoising network.

`metrics.py` which will contain IQ metrics implemented during lab 1.

`models.py` will contain all implemented models, as well as a code skeleton for CBD Network.

`trainer.py` contains a utility function to perform the training.

`train.py` contains a skeleton of the code, including data loading into batches, the overall training procedure, etc.

Finally, `utils.py` contains a set of tools and utility function that are helpful, but not mandatory to used.

Activity 1 - Basic auto-encoder pipeline
----------------------------------------

Extend the python script `train.py` in the script folder (or start with a notebook from scratch) as follows:

1. Load TID2013 dataset
2. Build a simple auto-encoder
3. Add the reconstruction loss (MSE)
4. Perform the training
6. Show some generated images

Activity 2 - CBD Network architecture
-------------------------------------

Replace the auto-encoder by the CBD Network:
1. Estimate the per-pixel noise variance
2. Generate the noisy image according the noise model
3. Modify the auto-encoder to match the CBD Network architecture
4. Test the pipeline (train + denoising quality)

Activity 3 - Losses
-------------------

1. Implement the Total Variation loss function
2. Implement the asymmetric loss function
3. Test the pipeline (train + denoising quality)

Additional Exercises
====================
1. Implement IQ metrics not implemented during lab 1 to evaluated the generated images
2. Modify existing code of TID2013 to be split between training and evaluation
3. Change the architecture of CBD Network and try to improve performances
4. Compute Inception Scores and/or Frechet Inception Distance

