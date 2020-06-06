# dual_IB
This is the implemntaion of the dual information bottleneck framework.
In the dual ib we replace the distoration of the IB wit the reverse one.

This git contains the varitional implenation of the dual IB.
There are 3 types of networks that you need to specipie - the encoder (Wide ResNet, or simple covnet), the decoder (one softmax layer as a mean Gaussian in our case) and the reverse decoder.

The trai.py is the main file for training with diffrenet netwroks and objectvives
For training with VdualIB run train.py file with run_model=dual_ib paramter. 

