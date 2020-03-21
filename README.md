# NT_Xent_loss_tensorflow

This code is the tensorflow implementation of the NT-Xent loss from SimCLR paper by Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton.
The pytorch implementation of the proposed algorithm is available here: https://github.com/mdiephuis/SimCLR/.
In this repo I just adapted the loss function to be used as a custom loss layer in Keras models.

This project contains two files:

contrastive_loss.py
main.py

The main file implements a simple siamese-like model in Keras and demonstrates how the custom contrastive (NT_Xent) loss can be plugged in.
