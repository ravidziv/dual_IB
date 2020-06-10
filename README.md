# dual_IB
This git contains the variational implementation of the dual IB.
 https://128.84.21.199/abs/2006.04641ץ

You need to specify three types of networks: the encoder (Wide ResNet, or simple covnet), the decoder (one softmax layer as a mean Gaussian in our case), and the reverse decoder.

The notebook [VdualIB_MNIST](VdualIB_MNIST.ipynb) contains an example how to run VdualIB network vs [CEB](https://arxiv.org/abs/2002.05380) network on MNIST dataset with simple encoder, decoder and reverese decoder.
For more extensive training, you can find the file [train](train.py). In this file, you can train with different networks, objectives and datasets.
