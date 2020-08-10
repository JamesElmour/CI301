# Resobust Neural Network Powered Cartoon Restoration

For my final year dissertation/project, I demonstrated that super resolution GANs can be employed to upscale images while restoring quality to them. In form of reducing production and video encoding artifacts on stills taken from said videos. By adopting an enhanced deep residual network super resolution (EDSR) network, altering its architecture and applying it to a real-world problem. Scaling images 4x their initial resolution, then restoring clarity and detail to the stills. Despite knowing nothing about machine learning and neural networks prior to undertaking this project.

## Results

These materials would be taken from; Woody Woodpecker in Pantry Panic (public domain) and Oggy &
the Cockroaches (Creative Commons Attribution). Transformed into 320px by 240px and encoded
into an MP4 file at 512kb/s prior to upscaling 4x and having quality restored via the neural network.

**Left - upscaled result from the neural network after being trained on cartoons, right - input data.**
![Woody Woodpecker in Pantry Panic](https://i.imgur.com/0MMxEzT.png "Woody Woodpecker in Pantry Panic")
![Oggy and the Cockroaches](https://i.imgur.com/jOzAedw.png "Oggy and the Cockroaches")

## Batch Normalization
Despite how common the usage of batch normalization is in modern model creation. The understanding of its effects is tenuous, at best. Despite the originating paper stating that batch normalization reduces the internal covariate shift (Loffe & Szegedy, 2015). Follow-up research from MIT indicated that its effect on the internal covariate shift fluctuates in accordance to the context of the network. It goes to later imply that batch normalization impacts the Lipschitzness of the network’s loss and gradients (Santurkar, Tsipras, Ilyas, & Madry, 2018). Thus, making them smoother, less erratic, and allowing for higher learning rates. Increasing the performance of the network.

The regularization offered by batch normalization can be utilized instead of other methods, such as dropout. Dropout randomly sets the activity of neurons within a network to zero. This method was widely popular before the introduction of batch normalization as a way to reduce overfitting (Srivastava, Hinton, Krizhevsky, Sutskever, & Salakhutdinov, 2014). Whilst dropout has resulted in substantial performance increases on models consisting of fully connected layers (Pham, Bluche, Kermorvant, & Louradour, 2014). Due to the inclusion of skip connections via residual blocks, the layers are no longer fully connected. Thus, the usefulness of dropout in the network will be limited. It could reduce performance by interfering with the identity matrix used throughout.

## Residual Blocks
Due to the deep nature of the generator network, containing many different layers. Several undesirable side-effects start to appear, such as vanishing gradient and, more so, the degradation problem. Which is the phenomenon whereby a shallower network seemingly outperforms a deeper one. As the accuracy of the deep network saturates, then degrades during training (Sahoo, 2018). Despite the potential of the deeper network outstripping the shallow counterpart. Residual blocks are designed to overcome these issues.

It does this by skipping layers and making use of an identity matrix, then combining the output of a residual block to the identity. By adding the identity to the residual block output, the block’s learning objective changes. Instead of focusing on learning how to manipulate the input data to the desired output, it focuses on reducing the residual between them.

## Sub-Pixel Convolution
A key feature of the generator model is sub-pixel convolution. Initial versions of a generator did not make use of this feature. By doing so, the scope of error within the generated images were on the super-pixel scale. Not only did this produce drastically inadequate results, but also greatly limit the network’s ability to understand the complex relationships housed within the image data.

Fortunately, the Tensorflow API provides a built-in method to reshape the depth of convolutional layer output into the desired space. Utilizing the depth_to_space method converts convolutional depth into spatial resolution. This is key for the generator’s accuracy. Previous GANs and other neural networks did not make use of sub-pixel manipulation. Thus, were significantly less accurate.

### Appendix
Loffe, S., & Szegedy, C. (2015, March 2). Batch normalization: Accelerating Deep Network Training By Reducing Internal Covariate Shift.

Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How Does Batch Normalization Help Optimization? Advances in Neural Information Processing Systems, pp. 2483-2493.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014, June 14). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, pp. 929-1958.

Sahoo, S. (2018, November 27). Residual blocks — Building blocks of ResNet. Retrieved from https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
