# dual_IB
This repository contains the implementation of the dual information bottleneck framework.
In the dual ib, we replace the distortion of the IB with the reverse one.

This git contains the variational implementation of the dual IB.
You need to specify three types of networks: the encoder (Wide ResNet, or simple covnet), the decoder (one softmax layer as a mean Gaussian in our case), and the reverse decoder.

The trai.py is the main file for training with different networks and objectives
For training with VdualIB run train.py file with run_model=dual_ib parameter. 

