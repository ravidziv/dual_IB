# The Dual Information Bottleneck

This git contains the variational TensorFlow 2 implementation of the dual Information Bottleneck.
 https://128.84.21.199/abs/2006.04641
 
 ![Information Plane](docs/inf_plane_git)

You need to specify three types of networks: the encoder (Wide ResNet, or simple covnet), the decoder (one softmax layer as a mean Gaussian in our case), and the reverse decoder.

The notebook [VdualIB_MNIST](VdualIB_MNIST.ipynb) contains an example of how to run the VdualIB network vs. [CEB](https://arxiv.org/abs/2002.05380) network on MNIST dataset with simple encoder, decoder, and the reverse decoder.
For more extensive training, you can find the file [train](train.py). In this file, you can train with different networks, objectives, and datasets.

The requirements file for needed packages is under doc directory, which also contains some config files for running [mlflow](https://mlflow.org/) framework
